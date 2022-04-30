import torch.nn as nn
from HeadNeRFOptions import BaseOptions
from RenderUtils import ExtractLandMarkPosition, SoftSimpleShader
import torch
import torch.nn.functional as F
import FaceModels

from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    PerspectiveCameras, RasterizationSettings, TexturesVertex, PointLights, blending, MeshRenderer, MeshRasterizer, HardPhongShader)

import numpy as np



class NL3DMMRenderer(nn.Module):
    def __init__(self, img_size, opt: BaseOptions):
        super().__init__()
        
        self.opt = opt
        self.img_h = img_size
        self.img_w = img_size
        
        self.build_info()
        self.build_nl3dmm()
        self.build_tool_funcs()
        self.set_3dmmdecoder_eval()
        

    def build_nl3dmm(self):
        self.decoder_3dmm = FaceModels.Linear_3DMM(self.opt)
        self.decoder_nl3dmm_new = FaceModels.NonLinear_3DMM(self.opt)
        

    def build_info(self):
        topo_info = np.load("ConfigFiles/nl_3dmm_topo_info.npz")
        tris = torch.as_tensor(topo_info['fv_indices']).long()
        vert_tris = torch.as_tensor(topo_info['corr_vf_indices']).long()
        
        self.register_buffer("tris", tris)
        self.register_buffer("corr_vf_indices", vert_tris)
        
        self.a0 = np.pi
        self.a1 = 2 * np.pi / np.sqrt(3.0)
        self.a2 = 2 * np.pi / np.sqrt(8.0)
        self.c0 = 1 / np.sqrt(4 * np.pi)
        self.c1 = np.sqrt(3.0) / np.sqrt(4 * np.pi)
        self.c2 = 3 * np.sqrt(5.0) / np.sqrt(12 * np.pi)
        self.d0 = 0.5/ np.sqrt(3.0)
        

    def build_tool_funcs(self):
        self.extract_lm3d_func = ExtractLandMarkPosition()


    def set_3dmmdecoder_eval(self):
        self.decoder_3dmm.eval()
        self.decoder_nl3dmm_new.eval()
        
        
    def train(self, mode=True):
        r"""Sets the module in training mode."""      
        self.training = mode
        for module in self.children():
            module.train(mode)

        self.set_3dmmdecoder_eval()
        return self
    

    def calc_geometry_Albedo(self, iden_codes, text_codes, expr_codes):
        batch_vps = self.decoder_nl3dmm_new(iden_codes, expr_codes)
        batch_vcs = self.decoder_3dmm(text_codes)
        
        return batch_vps, batch_vcs


    def calc_normal(self, geometry):

        vert_1 = geometry[:, self.tris[:, 0], :]
        vert_2 = geometry[:, self.tris[:, 1], :]
        vert_3 = geometry[:, self.tris[:, 2], :]
        
        nnorm = torch.cross(vert_2 - vert_1, vert_3 - vert_1, 2)
        tri_normal = F.normalize(nnorm, dim=2)
        tri_normal = F.pad(tri_normal, [0, 0, 0, 1, 0, 0], mode="constant", value=0)
                
        v_norm = tri_normal[:, self.corr_vf_indices, :].sum(2)
        vert_normal = F.normalize(v_norm, dim=-1)

        return vert_normal


    def build_color(self, batch_vcolor, batch_norm, batch_gamma):
        """
        batch_vcolor: [1, n_v, 3]
        batch_norm: [B, n_v, 3]
        batch_gamma: [B, 27]
        """
        # n_b, num_vertex, _ = batch_vcolor.size()
        n_b, num_vertex, _  = batch_norm.size()
       
        gamma = batch_gamma.view(-1, 9, 3)
        
        norm = batch_norm.view(-1, 3)
        nx, ny, nz = norm[:, 0], norm[:, 1], norm[:, 2]
        Y0 = torch.ones_like(nx) * self.a0 * self.c0

        arrH = []

        arrH.append(Y0)
        arrH.append(-self.a1 * self.c1 * ny)
        arrH.append(self.a1 * self.c1 * nz)
        arrH.append(-self.a1 * self.c1 * nx)
        arrH.append(self.a2 * self.c2 * nx * ny)
        arrH.append(-self.a2 * self.c2 * ny * nz)
        arrH.append(self.a2 * self.c2 * self.d0 * (3 * nz.pow(2) - 1))
        arrH.append(-self.a2 * self.c2 * nx * nz)
        arrH.append(self.a2 * self.c2 * 0.5 * (nx.pow(2) - ny.pow(2)))

        H = torch.stack(arrH, 1)
        Y = H.view(n_b, num_vertex, 9)
        lighting = Y.bmm(gamma)

        face_color = batch_vcolor * lighting
        return face_color


    def calc_ProjUV(self, cam_vps, batch_inmat):
        tv = cam_vps[:, :, 2:3] + 1e-7
        temp_uvs = cam_vps / tv
        uv = torch.bmm(temp_uvs, batch_inmat.permute(0, 2, 1))
        # uv = bmm_self_define_dim3(temp_uvs, batch_inmat, mat_2_is_trans=True)
        
        return uv[:, :, :2]
    

    def generate_renderer(self, batch_inmats):
        cur_device = batch_inmats.device
        batch_size = batch_inmats.size(0)
        cur_dtype = batch_inmats.dtype
        
        #cameras:
        half_w = self.img_w * 0.5
        half_h = self.img_h * 0.5
        focal_info = torch.stack([batch_inmats[:, 0, 0] / half_w, batch_inmats[:, 1, 1] / half_w], dim=-1)
        center_info = torch.stack([batch_inmats[:, 0, 2] / half_w - 1.0, batch_inmats[:, 1, 2] / half_h - 1.0], dim=-1)

        iden_mat = torch.eye(3)
        iden_mat[0, 0] = -1.0
        iden_mat[1, 1] = -1.0
        
        temp_Rmat = iden_mat.unsqueeze(0).expand(batch_size, -1, -1)
        temp_Vec = torch.zeros((batch_size, 3), dtype=cur_dtype)
        
        cameras = PerspectiveCameras(
                    focal_length=focal_info, 
                    principal_point=center_info,
                    R=temp_Rmat, 
                    T=temp_Vec,
                    device=cur_device
                )

        # focal_info = torch.stack([batch_inmats[:, 0, 0], batch_inmats[:, 1, 1]], dim=-1)
        # center_info = torch.stack([batch_inmats[:, 0, 2], batch_inmats[:, 1, 2]], dim=-1)

        # iden_mat = torch.eye(3)
        # iden_mat[0, 0] = -1.0
        # iden_mat[1, 1] = -1.0
        
        # temp_Rmat = iden_mat.unsqueeze(0).expand(batch_size, -1, -1)
        # temp_Vec = torch.zeros((batch_size, 3), dtype=cur_dtype)
        
        # cameras = PerspectiveCameras(
        #             focal_length=focal_info, 
        #             principal_point=center_info,
        #             R=temp_Rmat, 
        #             T=temp_Vec,
        #             in_ndc=False,
        #             image_size = [[self.img_h, self.img_w] * batch_size],
        #             device=cur_device
        #         )

        # light
        lights = PointLights(
            location=[[0.0, 0.0, 1e5]], 
            ambient_color=[[1, 1, 1]], 
            specular_color=[[0., 0., 0.]], 
            diffuse_color=[[0., 0., 0.]], device=cur_device
        )
        
        raster_settings = RasterizationSettings(
            image_size=(self.img_h, self.img_w),
            # blur_radius=0.000001,
            # faces_per_pixel=10,
            blur_radius=0,
            faces_per_pixel=1,
        )

        blend_params = blending.BlendParams(background_color=[0, 0, 0])

        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                raster_settings=raster_settings, 
                cameras=cameras
            ),
            shader=SoftSimpleShader(
                lights=lights,
                blend_params=blend_params,
                cameras=cameras
            ),
        ).to(cur_device)
        
        return renderer


    def render_img(self,
                    batch_vps, batch_vcs, illu_sh, 
                    c2l_Scales, c2l_Rmats, c2l_Tvecs, 
                    batch_Rmats, batch_Tvecs, batch_inmats,
                   ):

        batch_size = batch_vps.size(0)
        
        live_vps = torch.bmm(c2l_Scales * batch_vps, c2l_Rmats.permute(0, 2, 1)) + c2l_Tvecs.view(-1, 1, 3)
        cam_vps = torch.bmm(live_vps, batch_Rmats.permute(0, 2, 1)) + batch_Tvecs.view(-1, 1, 3)

        vns = self.calc_normal(cam_vps)     
        sh_vcs = self.build_color(batch_vcs, vns, illu_sh)
        
        face_color = TexturesVertex(sh_vcs)
        meshes = Meshes(cam_vps, self.tris.unsqueeze(0).expand(batch_size, -1, -1), face_color)

        cur_renderer = self.generate_renderer(batch_inmats)
        rendered_res = cur_renderer(meshes)
        rendered_res /= 255.0
        
        mask_c3b = (rendered_res[:, :, :, 3:]).detach().expand(-1, -1, -1, 3) > 0.0001
        rendered_img = rendered_res[:, :, :, :3]
        rendered_img = torch.clamp(rendered_img, min=0.0, max=1.0) 
        lm_3d_posi = self.extract_lm3d_func(cam_vps)
        proj_lm2d = self.calc_ProjUV(lm_3d_posi, batch_inmats)
        
        return rendered_img, mask_c3b, proj_lm2d, sh_vcs
    

    def generate_renderer_for_eval(self, batch_inmats):
        cur_device = batch_inmats.device
        batch_size = batch_inmats.size(0)
        cur_dtype = batch_inmats.dtype
        
        #cameras:
        # half_w = self.img_w * 0.5
        # half_h = self.img_h * 0.5
        focal_info = torch.stack([batch_inmats[:, 0, 0], batch_inmats[:, 1, 1]], dim=-1)
        center_info = torch.stack([batch_inmats[:, 0, 2], batch_inmats[:, 1, 2]], dim=-1)

        iden_mat = torch.eye(3)
        iden_mat[0, 0] = -1.0
        iden_mat[1, 1] = -1.0
        
        temp_Rmat = iden_mat.unsqueeze(0).expand(batch_size, -1, -1)
        temp_Vec = torch.zeros((batch_size, 3), dtype=cur_dtype)
        
        cameras = PerspectiveCameras(
                    focal_length=focal_info, 
                    principal_point=center_info,
                    R=temp_Rmat, 
                    T=temp_Vec,
                    in_ndc=False,
                    image_size = [[self.img_h, self.img_w] * batch_size],
                    device=cur_device
                )
        
        # light
        lights = PointLights(
            location=[[0.0, 0.0, 1e5]], 
            ambient_color=[[1, 1, 1]], 
            specular_color=[[0., 0., 0.]], 
            diffuse_color=[[0., 0., 0.]], device=cur_device
        )
        
        raster_settings = RasterizationSettings(
            image_size=(self.img_h, self.img_w),
            # blur_radius=0.000001,
            # faces_per_pixel=10,
            blur_radius=0,
            faces_per_pixel=1,
        )

        blend_params = blending.BlendParams(background_color=[0, 0, 0])

        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                raster_settings=raster_settings, 
                cameras=cameras
            ),
            shader=SoftSimpleShader(
                lights=lights,
                blend_params=blend_params,
                cameras=cameras
            ),
        ).to(cur_device)


        lights_phong = PointLights(
            location=[[0.0, 0.0, -1e5]], 
            ambient_color=[[0.5, 0.5, 0.5]], 
            specular_color=[[0.2, 0.2, 0.2]], 
            diffuse_color=[[0.3, 0.3, 0.3]], device=cur_device
        )

        renderer_phong = MeshRenderer(
            rasterizer=MeshRasterizer(
                raster_settings=raster_settings, 
                cameras=cameras
            ),
            shader=HardPhongShader(
                lights=lights_phong,
                blend_params=blend_params,
                cameras=cameras
            ),
        ).to(cur_device)

        return renderer, renderer_phong


    def render_img_for_eval(self,
                            batch_vps, batch_vcs, illu_sh, 
                            batch_Rmats, batch_Tvecs, batch_inmats
                        ):

        batch_size = batch_vps.size(0)
                
        cam_vps = torch.bmm(batch_vps, batch_Rmats.permute(0, 2, 1)) + batch_Tvecs.view(-1, 1, 3)
        vns = self.calc_normal(cam_vps)     
        sh_vcs = self.build_color(batch_vcs, vns, illu_sh)
        
        face_color = TexturesVertex(sh_vcs)
        meshes = Meshes(cam_vps, self.tris.unsqueeze(0).expand(batch_size, -1, -1), face_color)

        cur_renderer, renderer_phong = self.generate_renderer_for_eval(batch_inmats)
        rendered_res = cur_renderer(meshes)
        rendered_res /= 255.0
        
        mask_c3b = (rendered_res[:, :, :, 3:]).detach().expand(-1, -1, -1, 3) > 0.0001
        rendered_img = rendered_res[:, :, :, :3]
        rendered_img = torch.clamp(rendered_img, min=0.0, max=1.0) 
        lm_3d_posi = self.extract_lm3d_func(cam_vps)
        proj_lm2d = self.calc_ProjUV(lm_3d_posi, batch_inmats)

        color_phong = torch.ones_like(cam_vps)
        color_phong = TexturesVertex(color_phong)
        meshes_phong = Meshes(cam_vps, self.tris.unsqueeze(0).expand(batch_size, -1, -1), color_phong)

        rendered_phong = renderer_phong(meshes_phong)
        phong_mask_c3b = (rendered_phong[:, :, :, 3:]).detach().expand(-1, -1, -1, 3) > 0.0001
        rendered_phong = rendered_phong[:, :, :, :3]

        return rendered_img, mask_c3b, proj_lm2d, sh_vcs, rendered_phong, phong_mask_c3b


    def forward(self, 
                    iden_codes, text_codes, expr_codes, cur_sh, 
                    batch_Rmats, batch_Tvecs, batch_inmats, eval = False, **kwargs
                ):
        
        batch_vps = self.decoder_nl3dmm_new(iden_codes, expr_codes, scale = 0.01)
        batch_vcs = self.decoder_3dmm(text_codes)
        
        if eval:
            return self.render_img_for_eval(batch_vps, batch_vcs, cur_sh,
                                            batch_Rmats, batch_Tvecs, batch_inmats)
        else:
            c2l_Scales, c2l_Rmats, c2l_Tvecs = kwargs["c2l_Scales"], kwargs["c2l_Rmats"], kwargs["c2l_Tvecs"]
            return self.render_img(batch_vps, batch_vcs, cur_sh,
                                   c2l_Scales, c2l_Rmats, c2l_Tvecs, 
                                   batch_Rmats, batch_Tvecs, batch_inmats)
        

import torch
import torch.nn as nn
import os
import pickle as pkl

from pytorch3d.renderer import PointLights, Materials
from pytorch3d.renderer.blending import BlendParams, softmax_rgb_blend


class SoftSimpleShader(nn.Module):
    """
    Per pixel lighting - the lighting model is applied using the interpolated
    coordinates and normals for each pixel. The blending function returns the
    soft aggregated color using all the faces per pixel.

    To use the default values, simply initialize the shader with the desired
    device e.g.

    """

    def __init__(
        self, device="cpu", cameras=None, lights=None, materials=None, blend_params=None
    ):
        super().__init__()
        self.lights = lights if lights is not None else PointLights(device=device)
        self.materials = (
            materials if materials is not None else Materials(device=device)
        )
        self.cameras = cameras
        self.blend_params = blend_params if blend_params is not None else BlendParams()

    def to(self, device):
        # Manually move to device modules which are not subclasses of nn.Module
        self.cameras = self.cameras.to(device)
        self.materials = self.materials.to(device)
        self.lights = self.lights.to(device)
        return self

    def forward(self, fragments, meshes, **kwargs) -> torch.Tensor:
        
        texels = meshes.sample_textures(fragments)
        blend_params = kwargs.get("blend_params", self.blend_params)
        
        cameras = kwargs.get("cameras", self.cameras)
        if cameras is None:
            msg = "Cameras must be specified either at initialization \
                or in the forward pass of SoftPhongShader"
            raise ValueError(msg)
        znear = kwargs.get("znear", getattr(cameras, "znear", 1.0))
        zfar = kwargs.get("zfar", getattr(cameras, "zfar", 100.0))
        images = softmax_rgb_blend(
            texels, fragments, blend_params, znear=znear, zfar=zfar
        )
        
        return images



class ExtractLandMarkPosition(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.build_info()


    def build_info(self):
        lm_contour_info = "ConfigFiles/LandMarkInfo.pkl"
        assert os.path.exists(lm_contour_info)
        with open(lm_contour_info, "rb") as f:
            data_dict = pkl.load(f)
        
        contour_idx = data_dict["contour_idx"].T #[22, 17]
        inface_idx = data_dict["inface_idx"]   #[51, ]

        self.register_buffer("contour_idx", torch.from_numpy(contour_idx))
        self.register_buffer("inface_idx", torch.from_numpy(inface_idx))
        
        
    def forward(self, batch_cam_vps):
        
        batch_size = batch_cam_vps.size(0)
        
        contour_vps = batch_cam_vps[:, self.contour_idx, :] #[B, 22, 17, 3]
        contour_vps_x = contour_vps[:, :, :, 0]
        contour_vps_y = contour_vps[:, :, :, 1]
        
        left_contour_idx = torch.argmin(contour_vps_x[:, :, :8], dim=-2) #[B, 8]
        left_contour_idx = torch.gather(self.contour_idx[:, :8], 0, left_contour_idx)
        
        cent_contour_idx = torch.argmax(contour_vps_y[:, :, 8:9], dim=-2)    
        cent_contour_idx = torch.gather(self.contour_idx[:, 8:9], 0, cent_contour_idx)
        
        righ_contour_idx = torch.argmax(contour_vps_x[:, :, 9:17], dim=-2)    
        righ_contour_idx = torch.gather(self.contour_idx[:, 9:17], 0, righ_contour_idx)
        
        in_face_idx = self.inface_idx.view(1, 51).expand(batch_size, -1)
        # print()        
        lm_idx = torch.cat([
            left_contour_idx, 
            cent_contour_idx, 
            righ_contour_idx, 
            in_face_idx
            ], dim=-1)
        # print(lm_idx[2])
        # exit(0)
        
        lm_posi = torch.gather(batch_cam_vps, 1, lm_idx.unsqueeze(-1).expand(-1, -1, 3))
        
        return lm_posi
    

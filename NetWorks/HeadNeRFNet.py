import torch
import torch.nn as nn
from .utils import Embedder, CalcRayColor, GenSamplePoints, FineSample
from .models import MLPforNeRF
from NetWorks.neural_renderer import NeuralRenderer
import torch.nn.functional as F
from HeadNeRFOptions import BaseOptions


class HeadNeRFNet(nn.Module):
    def __init__(self, opt: BaseOptions, include_vd, hier_sampling) -> None:
        super().__init__()

        self.hier_sampling = hier_sampling
        self.include_vd = include_vd
        self._build_info(opt)
        self._build_tool_funcs()
        

    def _build_info(self, opt: BaseOptions):
        
        self.num_sample_coarse = opt.num_sample_coarse
        self.num_sample_fine = opt.num_sample_fine

        self.vp_n_freqs = 10
        self.include_input_for_vp_embeder = True

        self.vd_n_freqs = 4
        self.include_input_for_vd_embeder = True

        self.mlp_h_channel = opt.mlp_hidden_nchannels

        self.auxi_shape_code_dims = opt.auxi_shape_code_dims
        self.auxi_appea_code_dims = opt.auxi_appea_code_dims
        
        self.base_shape_code_dims = opt.iden_code_dims + opt.expr_code_dims
        self.base_appea_code_dims = opt.text_code_dims + opt.illu_code_dims
        
        self.featmap_size = opt.featmap_size
        self.featmap_nc = opt.featmap_nc        # num_channel
        self.pred_img_size = opt.pred_img_size
        self.opt = opt
        

    def _build_tool_funcs(self):

        vp_channels = self.base_shape_code_dims
        vp_channels += self.vp_n_freqs * 6 + 3 if self.include_input_for_vp_embeder else self.vp_n_freqs * 6
        self.vp_encoder = Embedder(N_freqs=self.vp_n_freqs, include_input=self.include_input_for_vp_embeder)
        
        vd_channels = self.base_appea_code_dims
        if self.include_vd:
            tv = self.vd_n_freqs * 6 + 3 if self.include_input_for_vd_embeder else self.vd_n_freqs * 6
            vd_channels += tv
            self.vd_encoder = Embedder(N_freqs=self.vd_n_freqs, include_input=self.include_input_for_vd_embeder)
        
        self.sample_func = GenSamplePoints(self.opt)
        
        if self.hier_sampling:
            self.fine_samp_func = FineSample(self.opt)
        
        self.fg_CD_predictor = MLPforNeRF(vp_channels=vp_channels, vd_channels=vd_channels, 
                                                    h_channel=self.mlp_h_channel, res_nfeat=self.featmap_nc)
        if self.hier_sampling:
            self.fine_fg_CD_predictor = MLPforNeRF(vp_channels=vp_channels, vd_channels=vd_channels, 
                                                    h_channel=self.mlp_h_channel, res_nfeat=self.featmap_nc)

        self.calc_color_func = CalcRayColor()
        self.neural_render = NeuralRenderer(bg_type=self.opt.bg_type, feat_nc=self.featmap_nc,  out_dim=3, final_actvn=True, 
                                                min_feat=32, featmap_size=self.featmap_size, img_size=self.pred_img_size)
        
        
    def calc_color_with_code(self, fg_vps, shape_code, appea_code, FGvp_embedder, 
                             FGvd_embedder, FG_zdists, FG_zvals, fine_level):
        
        ori_FGvp_embedder = torch.cat([FGvp_embedder, shape_code], dim=1)
        
        if self.include_vd:
            ori_FGvd_embedder = torch.cat([FGvd_embedder, appea_code], dim=1)
        else:
            ori_FGvd_embedder = appea_code
            
        if fine_level:
            FGmlp_FGvp_rgb, FGmlp_FGvp_density = self.fine_fg_CD_predictor(ori_FGvp_embedder, ori_FGvd_embedder)
        else:
            FGmlp_FGvp_rgb, FGmlp_FGvp_density = self.fg_CD_predictor(ori_FGvp_embedder, ori_FGvd_embedder)
            
        fg_feat, bg_alpha, batch_ray_depth, ori_batch_weight = self.calc_color_func(fg_vps, FGmlp_FGvp_rgb,
                                                                                    FGmlp_FGvp_density,
                                                                                    FG_zdists,
                                                                                    FG_zvals)
        
        batch_size = fg_feat.size(0)
        fg_feat = fg_feat.view(batch_size, self.featmap_nc, self.featmap_size, self.featmap_size)

        bg_alpha = bg_alpha.view(batch_size, 1, self.featmap_size, self.featmap_size)

        bg_featmap = self.neural_render.get_bg_featmap()
        bg_img = self.neural_render(bg_featmap)

        merge_featmap = fg_feat + bg_alpha * bg_featmap
        merge_img = self.neural_render(merge_featmap)

        res = {
            "merge_img": merge_img, 
            "bg_img": bg_img
        }
        
        return res, ori_batch_weight


    def _forward(
            self, 
            for_train, 
            batch_xy, batch_uv, 
            bg_code, shape_code, appea_code, 
            batch_Rmats, batch_Tvecs, batch_inv_inmats, dist_expr
        ):
        
        # cam - to - world
        batch_size, tv, n_r = batch_xy.size()
        assert tv == 2
        assert bg_code is None
        
        fg_sample_dict = self.sample_func(batch_xy, batch_Rmats, batch_Tvecs, batch_inv_inmats, for_train)
        fg_vps = fg_sample_dict["pts"]
        fg_dirs = fg_sample_dict["dirs"]

        FGvp_embedder = self.vp_encoder(fg_vps)
        
        if self.include_vd:
            FGvd_embedder = self.vd_encoder(fg_dirs)
        else:
            FGvd_embedder = None

        FG_zvals = fg_sample_dict["zvals"]
        FG_zdists = fg_sample_dict["z_dists"]
        
        cur_shape_code = shape_code.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, n_r, self.num_sample_coarse)
        cur_appea_code = appea_code.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, n_r, self.num_sample_coarse)

        c_ori_res, batch_weight = self.calc_color_with_code(
            fg_vps, cur_shape_code, cur_appea_code, FGvp_embedder, FGvd_embedder, FG_zdists, FG_zvals, fine_level = False
        )
        
        res_dict = {
            "coarse_dict":c_ori_res,
        }

        if self.hier_sampling:
            
            fine_sample_dict = self.fine_samp_func(batch_weight, fg_sample_dict, for_train)
            fine_fg_vps = fine_sample_dict["pts"]
            fine_fg_dirs = fine_sample_dict["dirs"]

            fine_FGvp_embedder = self.vp_encoder(fine_fg_vps)
            if self.include_vd:
                fine_FGvd_embedder = self.vd_encoder(fine_fg_dirs)
            else:
                fine_FGvd_embedder = None

            fine_FG_zvals = fine_sample_dict["zvals"]
            fine_FG_zdists = fine_sample_dict["z_dists"]
            
            num_sample = self.num_sample_coarse + self.num_sample_fine
            
            cur_shape_code = shape_code.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, n_r, num_sample)
            cur_appea_code = appea_code.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, n_r, num_sample)
            
            f_ori_res, _= self.calc_color_with_code(
               cur_shape_code, cur_appea_code, fine_FGvp_embedder, fine_FGvd_embedder, fine_FG_zdists, fine_FG_zvals, 
               fine_level=True
            )
            
            res_dict["fine_dict"] = f_ori_res

        return res_dict
    

    def forward(
                self,
                mode, 
                batch_xy, batch_uv,
                bg_code, shape_code, appea_code, 
                batch_Rmats, batch_Tvecs, batch_inv_inmats, dist_expr = False, **kwargs
        ):
        assert mode in ["train", "test"]
        return self._forward(
            mode == "train",
            batch_xy, batch_uv,
            bg_code, shape_code, appea_code, 
            batch_Rmats, batch_Tvecs, batch_inv_inmats, dist_expr
        )

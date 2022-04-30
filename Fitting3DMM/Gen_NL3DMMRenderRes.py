import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pickle as pkl
import torch

from NL3DMMRenderer import NL3DMMRenderer
from tool_funcs import soft_load_model

import cv2
import numpy as np
from os.path import join
from glob import glob
from HeadNeRFOptions import BaseOptions


class GenRenderRes(object):
    def __init__(self, gpu_id) -> None:
        super().__init__()
        self.img_size = 512
        self.opt = BaseOptions()
        self.device = torch.device("cuda:%d" % gpu_id)
        self.build_info()
        self.build_tool_funcs()


    def build_info(self):
        self.iden_code_dims = self.opt.iden_code_dims
        self.expr_code_dims = self.opt.expr_code_dims
        self.text_code_dims = self.opt.text_code_dims
        self.illu_code_dims = self.opt.illu_code_dims


    def build_tool_funcs(self):

        self.nl3dmm_render = NL3DMMRenderer(self.img_size, self.opt).to(self.device)
        self.nl3dmm_render = soft_load_model(self.nl3dmm_render, "ConfigFiles/nl3dmm_net_dict.pth")
        
        
    def render_3dmm(self, pkl_path, lm_path = None, img_path = None):
        
        with open(pkl_path, "rb") as f:
            temp_dict = pkl.load(f)

        code_info = temp_dict["code"]
        w2c_Rmat = temp_dict["w2c_Rmat"]
        w2c_Tvec = temp_dict["w2c_Tvec"]
        inmat = temp_dict["inmat"]
        
        iden_code = code_info[:self.iden_code_dims]
        expr_code = code_info[self.iden_code_dims:self.iden_code_dims + self.expr_code_dims]
        text_code = code_info[self.iden_code_dims + self.expr_code_dims : self.iden_code_dims + self.expr_code_dims + self.text_code_dims]
        illu_code = code_info[self.iden_code_dims + self.expr_code_dims + self.text_code_dims:]

        iden_code = iden_code.unsqueeze(0).to(self.device)
        expr_code = expr_code.unsqueeze(0).to(self.device)
        text_code = text_code.unsqueeze(0).to(self.device)
        illu_code = illu_code.unsqueeze(0).to(self.device)
        w2c_Rmat = w2c_Rmat.unsqueeze(0).to(self.device)
        w2c_Tvec = w2c_Tvec.unsqueeze(0).to(self.device)
        inmat = inmat.unsqueeze(0).to(self.device)
        
        with torch.set_grad_enabled(False):
            render_img, mask_c3b, pred_lm2ds, sh_vcs, phong_img, phong_maskc3b = self.nl3dmm_render(
                        iden_code, text_code, expr_code, illu_code, w2c_Rmat, w2c_Tvec, inmat, eval = True
                    )

        render_img = torch.clamp(render_img, 0.0, 1.0)
        phong_img = torch.clamp(phong_img, 0.0, 1.0)            
        
        render_img = (render_img[0].cpu().numpy() * 255).astype(np.uint8)
        phong_img = (phong_img[0].cpu().numpy() * 255).astype(np.uint8)
        
        # cv2.imwrite("./temp_res/render_res.png", render_img[:, :, ::-1])
        # cv2.imwrite("./temp_res/phong_img.png", phong_img)

        if img_path is not None:
            ori_img = cv2.imread(img_path)
        else:
            ori_img = np.ones((self.img_size, self.img_size, 3), dtype=np.uint8) * 255
            
        ori_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)

        mask_img = mask_c3b[0].cpu().numpy()
        img_1 = ori_img.copy()
        img_1[mask_img] = render_img[mask_img]

        mask_img = phong_maskc3b[0].cpu().numpy()
        img_2 = ori_img.copy()
        img_2[mask_img] = phong_img[mask_img]
        
        res_img = np.concatenate([ori_img, img_1, img_2], axis=1)
        res_img = cv2.resize(res_img, (0, 0), fx=0.5, fy=0.5)
        cv2.imwrite("./temp_res/res_img_new.png", res_img[:, :, ::-1])
        return res_img
        
    
if __name__ == "__main__":
    tt = GenRenderRes(gpu_id=0)
    
    base_name = "000006"
    pkl_path = "test_data/single_images/img_%s.pkl" % base_name
    img_path = "test_data/single_images/img_%s.png" % base_name
    
    tt.render_3dmm(pkl_path=pkl_path, img_path=img_path)
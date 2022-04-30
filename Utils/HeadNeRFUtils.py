import torch
from NetWorks.HeadNeRFNet import HeadNeRFNet

import json
import os
import pickle as pkl
import numpy as np

from tool_funcs import eulurangle2Rmat
from HeadNeRFOptions import BaseOptions


class HeadNeRFUtils(object):
    def __init__(self, model_path) -> None:
        super().__init__()

        self.model_path = model_path
        self.device = torch.device("cuda:0")
        
        self.source_img = None
        self.target_img = None
        
        self.build_net()
        self.build_cam()
        self.gen_uv_xy_info()


    def build_net(self):
        
        check_dict = torch.load(self.model_path, map_location=torch.device("cpu"))
        self.opt = BaseOptions(check_dict["para"])
        
        net = HeadNeRFNet(self.opt, include_vd=False, hier_sampling=False)        
        net.load_state_dict(check_dict["net"])
        
        self.net = net.to(self.device)
        self.net.eval()
        

    def build_cam(self):
        
        with open("ConfigFiles/cam_inmat_info_32x32.json", "r") as f:
            temp_dict = json.load(f)
        # temp_inmat = torch.as_tensor(temp_dict["inmat"])
        temp_inv_inmat = torch.as_tensor(temp_dict["inv_inmat"])
        scale = self.opt.featmap_size / 32.0
        temp_inv_inmat[:2, :2] /= scale
        
        self.inv_inmat = temp_inv_inmat.view(1, 3, 3).to(self.device)
        
        tv_z = 0.5 + 11.5        
        base_rmat = torch.eye(3).float().view(1, 3, 3).to(self.device)
        base_rmat[0, 1:, :] *= -1
        base_tvec = torch.zeros(3).float().view(1, 3, 1).float().to(self.device)
        base_tvec[0, 2, 0] = tv_z
        
        self.base_c2w_Rmats = base_rmat
        self.base_c2w_Tvecs = base_tvec
    

    def gen_uv_xy_info(self):
        mini_h = self.opt.featmap_size
        mini_w = self.opt.featmap_size
        
        indexs = torch.arange(mini_h * mini_w)
        x_coor = (indexs % mini_w).view(-1)
        y_coor = torch.div(indexs, mini_w, rounding_mode="floor").view(-1)
                
        xy = torch.stack([x_coor, y_coor], dim=0).float()
        uv = torch.stack([x_coor.float() / float(mini_w), y_coor.float() / float(mini_h)], dim=-1)
        
        self.xy = xy.unsqueeze(0).to(self.device)
        self.uv = uv.unsqueeze(0).to(self.device)


    def exact_code(self, code_pkl_path):
        assert os.path.exists(code_pkl_path)
        temp_dict = torch.load(code_pkl_path, map_location="cpu")
        code_info = temp_dict["code"]
        
        for k, v in code_info.items():
            if v is not None:
                code_info[k] = v.to(self.device)

        shape_code = code_info["shape_code"]
        iden_code = shape_code[:, :100]
        expr_code = shape_code[:, 100:]
        
        appea_code = code_info["appea_code"]
        text_code = appea_code[:, :100]
        illu_code = appea_code[:, 100:]
        
        return iden_code, expr_code, text_code, illu_code
    

    def build_code(self, config_path):
        assert os.path.exists(config_path)
        with open(config_path) as f:
            temp_dict = json.load(f)

        iden_code_1, expr_code_1, text_code_1, illu_code_1 = self.exact_code(temp_dict["code_path_1"])
        self.iden_code_1 = iden_code_1
        self.expr_code_1 = expr_code_1
        self.text_code_1 = text_code_1
        self.illu_code_1 = illu_code_1

        iden_code_2, expr_code_2, text_code_2, illu_code_2 = self.exact_code(temp_dict["code_path_2"])
        self.iden_code_2 = iden_code_2
        self.expr_code_2 = expr_code_2
        self.text_code_2 = text_code_2
        self.illu_code_2 = illu_code_2
        
        
    def update_code_1(self, file_path):
        assert os.path.exists(file_path)
        iden_code_1, expr_code_1, text_code_1, illu_code_1 = self.exact_code(file_path)
        self.iden_code_1 = iden_code_1
        self.expr_code_1 = expr_code_1
        self.text_code_1 = text_code_1
        self.illu_code_1 = illu_code_1

        shape_code = torch.cat([self.iden_code_1, self.expr_code_1], dim=1)
        appea_code = torch.cat([self.text_code_1, self.illu_code_1], dim=1)
        
        code_info = {
            "bg_code": None, 
            "shape_code":shape_code, 
            "appea_code":appea_code, 
        }
        cam_info = self.gen_cam(0.0, 0.0, 0.0)
        
        pred_dict = self.net("test", self.xy, self.uv,  **code_info, **cam_info)
        img = pred_dict["coarse_dict"]["merge_img"]
        self.source_img = (img[0].detach().cpu().permute(1, 2, 0).numpy()* 255).astype(np.uint8)


    def update_code_2(self, file_path):
        assert os.path.exists(file_path)
        iden_code_2, expr_code_2, text_code_2, illu_code_2 = self.exact_code(file_path)
        self.iden_code_2 = iden_code_2
        self.expr_code_2 = expr_code_2
        self.text_code_2 = text_code_2
        self.illu_code_2 = illu_code_2

        shape_code = torch.cat([self.iden_code_2, self.expr_code_2], dim=1)
        appea_code = torch.cat([self.text_code_2, self.illu_code_2], dim=1)
        
        code_info = {
            "bg_code": None, 
            "shape_code":shape_code, 
            "appea_code":appea_code, 
        }
        cam_info = self.gen_cam(0.0, 0.0, 0.0)
        
        pred_dict = self.net("test", self.xy, self.uv,  **code_info, **cam_info)
        img = pred_dict["coarse_dict"]["merge_img"]
        self.target_img = (img[0].detach().cpu().permute(1, 2, 0).numpy()* 255).astype(np.uint8)


    def gen_code(self, iden_t, expr_t, text_t, illu_t):
        iden_code = self.iden_code_1 * (1 - iden_t) + self.iden_code_2 * iden_t 
        text_code = self.text_code_1 * (1 - text_t) + self.text_code_2 * text_t
        illu_code = self.illu_code_1 * (1 - illu_t) + self.illu_code_2 * illu_t
        expr_code = self.expr_code_1 * (1 - expr_t) + self.expr_code_2 * expr_t

        # iden_code = iden_code_1
        # text_code = text_code_1
        # expr_code = expr_code_1
        # illu_code = illu_code_1
        
        shape_code = torch.cat([iden_code, expr_code], dim=1)
        appea_code = torch.cat([text_code, illu_code], dim=1)
        
        code_info = {
            "bg_code": None, 
            "shape_code":shape_code, 
            "appea_code":appea_code, 
        }
        
        return code_info


    def gen_cam(self, pitch, yaw, roll):

        angle = np.array([-pitch, -yaw, -roll])
        delta_rmat = eulurangle2Rmat(angle)
        delta_rmat = torch.from_numpy(delta_rmat).unsqueeze(0).to(self.device)
        
        new_rmat = torch.bmm(delta_rmat, self.base_c2w_Rmats)
        new_tvec = torch.bmm(delta_rmat, self.base_c2w_Tvecs)
        cam_info = {
            "batch_Rmats": new_rmat,
            "batch_Tvecs": new_tvec,
            "batch_inv_inmats": self.inv_inmat
        }
        
        return cam_info

    
    def gen_image(self, iden_t, expr_t, text_t, illu_t, pitch, yaw, roll):
        code_info = self.gen_code(iden_t, expr_t, text_t, illu_t)
        cam_info = self.gen_cam(pitch, yaw, roll)
        # cam_info = {
        #     "batch_Rmats": self.base_c2w_Rmats,
        #     "batch_Tvecs": self.base_c2w_Tvecs,
        #     "batch_inv_inmats": self.inv_inmat
        # }
        
        pred_dict = self.net("test", self.xy, self.uv,  **code_info, **cam_info)
        img = pred_dict["coarse_dict"]["merge_img"]
        img = (img[0].detach().cpu().permute(1, 2, 0).numpy()* 255).astype(np.uint8)
        # cv2.imwrite("./temp_res/ts_img.png", img)
        # print("save_img")
        return img
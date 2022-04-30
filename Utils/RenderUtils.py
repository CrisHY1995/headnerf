import torch
from tqdm import tqdm
from HeadNeRFOptions import BaseOptions
import json
import numpy as np
import math


class RenderUtils(object):
    def __init__(self, view_num, device, opt: BaseOptions) -> None:
        super().__init__()
        self.view_num = view_num
        self.device = device
        self.opt = opt
        self.build_base_info()
        self.build_cam_info()


    def build_base_info(self):
        mini_h = self.opt.featmap_size
        mini_w = self.opt.featmap_size

        indexs = torch.arange(mini_h * mini_w)
        x_coor = (indexs % mini_w).view(-1)
        y_coor = torch.div(indexs, mini_w, rounding_mode="floor").view(-1)
        
        xy = torch.stack([x_coor, y_coor], dim=0).float()
        uv = torch.stack([x_coor.float() / float(mini_w), y_coor.float() / float(mini_h)], dim=-1)
        
        self.ray_xy = xy.unsqueeze(0).to(self.device)
        self.ray_uv = uv.unsqueeze(0).to(self.device)
        
        with open("ConfigFiles/cam_inmat_info_32x32.json", "r") as f:
            temp_dict = json.load(f)
        # temp_inmat = torch.as_tensor(temp_dict["inmat"])
        temp_inv_inmat = torch.as_tensor(temp_dict["inv_inmat"])
        temp_inv_inmat[:2, :2] /= (self.opt.featmap_size / 32.0)
        self.inv_inmat = temp_inv_inmat.view(1, 3, 3).to(self.device)
        

    def build_cam_info(self):
        tv_z = 0.5 + 11.5
        tv_x = 5.3 

        center_ = np.array([0, 0.0, 0.0]).reshape(3)
        temp_center = np.array([0.0, 0.0, tv_z]).reshape(3)
        temp_cam_center = np.array([[tv_x, 0.0, tv_z]]).reshape(3)

        radius_ = math.sqrt(np.sum((temp_cam_center - center_)**2) - np.sum((temp_center - center_)**2))
        temp_d2 = np.array([[0.0, -1.0, 0.0]]).reshape(3)
        
        cam_info_list = []
        
        angles = np.linspace(0, 360.0, self.view_num)
        for angle in angles:
            theta_ = angle / 180.0 * 3.1415926535
            x_ = math.cos(theta_) * radius_
            y_ = math.sin(theta_) * radius_
            
            temp_vp = np.array([x_, y_, tv_z]).reshape(3)
            d_1 = (center_ - temp_vp).reshape(3)

            d_2 = np.cross(temp_d2, d_1)
            d_3 = np.cross(d_1, d_2)

            d_1 = d_1 / np.linalg.norm(d_1)
            d_2 = d_2 / np.linalg.norm(d_2)
            d_3 = d_3 / np.linalg.norm(d_3)

            rmat = np.zeros((3,3), dtype=np.float32)
            rmat[:, 0] = d_2
            rmat[:, 1] = d_3
            rmat[:, 2] = d_1
            rmat = torch.from_numpy(rmat).view(1, 3, 3).to(self.device)
            tvec = torch.from_numpy(temp_vp).view(1, 3, 1).float().to(self.device)
            
            cam_info = {
                "batch_Rmats": rmat,
                "batch_Tvecs": tvec,
                "batch_inv_inmats": self.inv_inmat,
            }
            cam_info_list.append(cam_info)
        
        base_rmat = torch.eye(3).float().view(1, 3, 3).to(self.device)
        base_rmat[0, 1:, :] *= -1
        base_tvec = torch.zeros(3).float().view(1, 3, 1).float().to(self.device)
        base_tvec[0, 2, 0] = tv_z
        
        self.base_cam_info = {
            "batch_Rmats": base_rmat,
            "batch_Tvecs": base_tvec,
            "batch_inv_inmats": self.inv_inmat,
        }
        
        self.cam_info_list = cam_info_list
        
        
    def render_novel_views(self, net, code_info):
        res_img_list = []
        
        batch_xy = self.ray_xy
        batch_uv = self.ray_uv
        loop_bar = tqdm(range(self.view_num), leave=True)
        for i in loop_bar:
            loop_bar.set_description("Generate Novel Views ")
            cam_info = self.cam_info_list[i]
            with torch.set_grad_enabled(False):
                pred_dict = net("test",batch_xy, batch_uv, **code_info,**cam_info)
            coarse_fg_rgb = pred_dict["coarse_dict"]["merge_img"]
            coarse_fg_rgb = (coarse_fg_rgb[0].detach().cpu().permute(1, 2, 0).numpy()* 255).astype(np.uint8)
            res_img_list.append(coarse_fg_rgb)

        return res_img_list
    
    
    def render_morphing_res(self, net, code_info_1, code_info_2, nums):
        
        batch_xy = self.ray_xy
        batch_uv = self.ray_uv
        res_img_list = []
        
        loop_bar = tqdm(range(nums), leave=True)
        for i in loop_bar:
            loop_bar.set_description("Generate Morphing Res")
            tv = 1.0 - (i / (nums - 1))
            shape_code = code_info_1["shape_code"] * tv + code_info_2["shape_code"] * (1 - tv)
            appea_code = code_info_1["appea_code"] * tv + code_info_2["appea_code"] * (1 - tv)
            
            code_info = {
                "bg_code":None,
                "shape_code":shape_code, 
                "appea_code":appea_code
            }
            
            with torch.set_grad_enabled(False):
                pred_dict = net("test",batch_xy, batch_uv, **code_info,**self.base_cam_info)
            coarse_fg_rgb = pred_dict["coarse_dict"]["merge_img"]
            coarse_fg_rgb = (coarse_fg_rgb[0].detach().cpu().permute(1, 2, 0).numpy()* 255).astype(np.uint8)
            res_img_list.append(coarse_fg_rgb)
            
        return res_img_list
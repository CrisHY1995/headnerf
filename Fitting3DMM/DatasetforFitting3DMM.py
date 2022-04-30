from ntpath import join
import torch
import cv2
from torch.utils.data import Dataset

import numpy as np
import math
import os
from glob import glob


class DatasetforFitting3DMM(Dataset):
    def __init__(self, img_dir, img_size, intermediate_size) -> None:
        super().__init__()

        if intermediate_size == img_size:
            self.cam_h = img_size
            self.cam_w = img_size
            self.img_size = (img_size, img_size)
            self.reize_img = False
            self.lm_scale = 1.0
        else:
            self.cam_h = intermediate_size
            self.cam_w = intermediate_size
            self.img_size = (intermediate_size, intermediate_size)
            self.reize_img = True
            self.lm_scale = intermediate_size / float(img_size)

        self.img_dir = img_dir
        self.build_info()
        
    
    def build_info(self):
        
        img_path_list = [x for x in glob("%s/*.png" % self.img_dir) if "mask" not in x]
        img_path_list.sort()
        
        self.img_path_list = img_path_list
        self.total_sample_num = len(self.img_path_list)
        
        if self.total_sample_num == 0:
            print("Dir: %s does not include ang *.png files")
            exit(0)
            
        inmat = torch.eye(3, dtype=torch.float32)
        inmat[0, 0] = 812.0 * self.cam_h / 400.0
        inmat[1, 1] = 812.0 * self.cam_h / 400.0
        inmat[0, 2] = self.cam_w // 2
        inmat[1, 2] = self.cam_h // 2
        self.base_inmat = inmat

        self.base_w2c_Rmat = torch.as_tensor([[ 0.9988,  0.0097, -0.0486],
                                              [ 0.0105, -0.9998,  0.0149],
                                              [-0.0485, -0.0154, -0.9987]])
        self.base_w2c_Tvec = torch.as_tensor([[0.0255], 
                                              [0.1288], 
                                              [6.2085]])
        

    def load_images(self, img_path):
        assert os.path.exists(img_path)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32)/255.0

        if self.reize_img:
            img = cv2.resize(img, dsize=self.img_size, fx=0, fy=0, interpolation=cv2.INTER_LINEAR)
        
        return img


    def load_lm_info(self, lm_path):
        assert os.path.exists(lm_path)
        lm_info = np.loadtxt(lm_path, dtype = np.float32).reshape(68, 2)
        
        if self.reize_img:
            lm_info *= self.lm_scale
        return torch.from_numpy(lm_info).unsqueeze(0)


    def load_one_sample(self, local_t_idx):
        img_path = self.img_path_list[local_t_idx]
        base_name = os.path.basename(img_path)[:-4]
        
        lm_path = join(self.img_dir, "%s_lm2d.txt" % base_name)

        img = self.load_images(img_path)
        img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
        lm_info = self.load_lm_info(lm_path)

        inmats = self.base_inmat.unsqueeze(0)
        w2c_Rmats = self.base_w2c_Rmat.unsqueeze(0)
        w2c_Tvecs = self.base_w2c_Tvec.unsqueeze(0)

        res_dict = {
            "img":img,
            "inmats":inmats,
            "lm":lm_info,
            "w2c_Rmats":w2c_Rmats,
            "w2c_Tvecs":w2c_Tvecs,
            "base_name":[base_name],
            }
        
        return res_dict
    

    def load_batch_sample(self, t_idx_list):

        img_list = []
        base_name_list = []
        lm_info_list = []
        
        for t_idx in t_idx_list:
            img_path = self.img_path_list[t_idx]
            
            base_name = os.path.basename(img_path)[:-4]
            lm_path = "%s/%s_lm2d.txt" % (self.img_dir, base_name)
                       
            img = self.load_images(img_path)
            img = torch.from_numpy(img).unsqueeze(0)
            lm_info = self.load_lm_info(lm_path)

            img_list.append(img)
            base_name_list.append(base_name)
            lm_info_list.append(lm_info)
        
        inmats = self.base_inmat.unsqueeze(0).repeat(len(t_idx_list), 1, 1)
        w2c_Rmats = self.base_w2c_Rmat.unsqueeze(0).repeat(len(t_idx_list), 1, 1)
        w2c_Tvecs = self.base_w2c_Tvec.unsqueeze(0).repeat(len(t_idx_list), 1, 1)

        res_dict = {
            "imgs":torch.cat(img_list, dim=0),
            "inmats":inmats,
            "w2c_Rmats":w2c_Rmats,
            "w2c_Tvecs":w2c_Tvecs,
            "base_names":base_name_list,
            "lms":torch.cat(lm_info_list, dim=0)
            }
        
        return res_dict


    def __len__(self):
        return self.total_sample_num


    def __getitem__(self, index):
        return self.load_one_sample(index)
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tqdm import tqdm

import torch
from Utils.FittingNL3DMM_LossUtils import FittingNL3DMMLossUtils
from NL3DMMRenderer import NL3DMMRenderer

import cv2
from tool_funcs import soft_load_model, convert_loss_dict_2_str, draw_res_img
import pickle as pkl

from os.path import join

from  DatasetforFitting3DMM import DatasetforFitting3DMM
from HeadNeRFOptions import BaseOptions
import argparse



class FittingNL3DMM(object):
    def __init__(self, img_size, intermediate_size, gpu_id, batch_size, img_dir) -> None:
        
        self.img_size = img_size
        self.intermediate_size = intermediate_size
        
        self.batch_size = batch_size
        self.img_dir = img_dir
        
        self.device = torch.device("cuda:%d" % gpu_id)
        self.opt = BaseOptions()

        self.iden_code_dims = self.opt.iden_code_dims
        self.expr_code_dims = self.opt.expr_code_dims
        self.text_code_dims = self.opt.text_code_dims
        self.illu_code_dims = self.opt.illu_code_dims

        self.build_dataset()
        self.build_tool_funcs()
        

    def build_tool_funcs(self):
        
        self.nl3dmm_render = NL3DMMRenderer(self.intermediate_size, self.opt).to(self.device)
        self.nl3dmm_render = soft_load_model(self.nl3dmm_render, "ConfigModels/nl3dmm_net_dict.pth")
        self.loss_utils = FittingNL3DMMLossUtils()


    def build_dataset(self):
        self.data_utils = DatasetforFitting3DMM(self.img_dir, self.img_size, self.intermediate_size)


    @staticmethod
    def compute_rotation(angles):
        """
        Return:
            rot              -- torch.tensor, size (B, 3, 3) pts @ trans_mat

        Parameters:
            angles           -- torch.tensor, size (B, 3), radian
        """
        cur_device = angles.device
        batch_size = angles.shape[0]
        ones = torch.ones([batch_size, 1]).to(cur_device)
        zeros = torch.zeros([batch_size, 1]).to(cur_device)
        x, y, z = angles[:, :1], angles[:, 1:2], angles[:, 2:],
        
        rot_x = torch.cat([
            ones, zeros, zeros,
            zeros, torch.cos(x), -torch.sin(x), 
            zeros, torch.sin(x), torch.cos(x)
        ], dim=1).reshape([batch_size, 3, 3])
        
        rot_y = torch.cat([
            torch.cos(y), zeros, torch.sin(y),
            zeros, ones, zeros,
            -torch.sin(y), zeros, torch.cos(y)
        ], dim=1).reshape([batch_size, 3, 3])

        rot_z = torch.cat([
            torch.cos(z), -torch.sin(z), zeros,
            torch.sin(z), torch.cos(z), zeros,
            zeros, zeros, ones
        ], dim=1).reshape([batch_size, 3, 3])

        rot = rot_z @ rot_y @ rot_x
        return rot.permute(0, 2, 1)
    
    
    def opt_batch_data(self, w2c_Rmats, w2c_Tvecs, inmats, imgs, lms, base_names):
        batch_size = imgs.size(0)

        batch_imgs, gt_lm2ds = imgs.to(self.device), lms.to(self.device)
        w2c_Rmats, w2c_Tvecs, w2c_inmats = w2c_Rmats.to(self.device), w2c_Tvecs.to(self.device), inmats.to(self.device)

        iden_codes = torch.zeros((batch_size, self.iden_code_dims), dtype=torch.float32, requires_grad=True, device=self.device)
        expr_codes = torch.zeros((batch_size, self.expr_code_dims), dtype=torch.float32, requires_grad=True, device=self.device)
        text_codes = torch.zeros((batch_size, self.text_code_dims), dtype=torch.float32, requires_grad=True, device=self.device)
        illu_codes = torch.zeros((batch_size, self.illu_code_dims), dtype=torch.float32, requires_grad=False, device=self.device)
        illu_codes = illu_codes.view(batch_size, 9, 3)
        illu_codes[:, 0, :] += 0.8
        illu_codes = illu_codes.view(batch_size, 27)
        illu_codes.requires_grad = True

        c2l_eulur = torch.zeros((batch_size, 3), dtype=torch.float32, requires_grad=True, device=self.device)
        c2l_Tvecs = torch.zeros((batch_size, 3), dtype=torch.float32, requires_grad=True, device=self.device)
        c2l_Scales = 1.0

        init_lr_1 = 0.01
        params_group = [
            {'params': [c2l_eulur, c2l_Tvecs], 'lr': init_lr_1},
        ]
        
        optimizer = torch.optim.Adam(params_group, betas=(0.9, 0.999))
        iter_num_1 = 50
        for iter_ in range(iter_num_1):
            with torch.set_grad_enabled(True):
                c2l_Rmats = self.compute_rotation(c2l_eulur)

                rendered_img, mask_c3b, proj_lm2d, sh_vcs = self.nl3dmm_render(
                            iden_codes, text_codes, expr_codes, illu_codes,
                            w2c_Rmats, w2c_Tvecs, w2c_inmats, eval = False, 
                            c2l_Scales = c2l_Scales, c2l_Rmats = c2l_Rmats, c2l_Tvecs = c2l_Tvecs
                        )
                # mask_c3b_backup = mask_c3b.clone()

                # masks = masks.expand(-1, -1, -1, 3)
                # mask_c3b[masks != 1] = False

                pred_and_gt_data_dict = {
                    "batch_vcs":sh_vcs,
                    "rendered_imgs":rendered_img, 
                    "gt_imgs":batch_imgs,
                    "mask_c3d":mask_c3b,
                    "proj_lm2ds":proj_lm2d,
                    "gt_lm2ds":gt_lm2ds
                }
                
                norm_code_info = {
                    "iden_codes":iden_codes, 
                    "text_codes":text_codes, 
                    "expr_codes":expr_codes
                }

                batch_loss_dict = self.loss_utils.calc_total_loss(
                    cur_illus = illu_codes,
                    **pred_and_gt_data_dict,
                    **norm_code_info,
                    lm_w=100.0
                )
            
            optimizer.zero_grad()
            batch_loss_dict["total_loss"].backward()
            optimizer.step()

            # print("Step 1, Iter: %04d |"%iter_, convert_loss_dict_2_str(batch_loss_dict))
            # res_img = draw_res_img(rendered_img, batch_imgs, mask_c3b_backup, proj_lm2ds=proj_lm2d, gt_lm2ds=gt_lm2ds, num_per_row=3)
            # res_img  = res_img[:, :, ::-1] 
            # cv2.imwrite("./temp_res/opt_imgs_2/opt_image_res%03d.png" % iter_, cv2.resize(res_img, (0,0), fx=0.5, fy=0.5))

        init_lr_2 = 0.01
        params_group = [
            {'params': [c2l_eulur, c2l_Tvecs], 'lr': init_lr_2 * 1.0},
            {'params': [iden_codes, text_codes, expr_codes], 'lr': init_lr_2 * 0.5},
            {'params': [illu_codes], 'lr': init_lr_2 * 0.5},
        ]

        optimizer = torch.optim.Adam(params_group, betas=(0.9, 0.999))
        iter_num_2 = iter_num_1 + 200
        for iter_ in range(iter_num_1, iter_num_2):
            lm_w = 25.0
            with torch.set_grad_enabled(True):
                c2l_Rmats = self.compute_rotation(c2l_eulur)

                rendered_img, mask_c3b, proj_lm2d, sh_vcs = self.nl3dmm_render(
                            iden_codes, text_codes, expr_codes, illu_codes,
                            w2c_Rmats, w2c_Tvecs, w2c_inmats, eval = False, 
                            c2l_Scales = c2l_Scales, c2l_Rmats = c2l_Rmats, c2l_Tvecs = c2l_Tvecs
                        )
                # mask_c3b_backup = mask_c3b.clone()
                # masks = masks.expand(-1, -1, -1, 3)
                # mask_c3b[masks != 1] = False
                pred_and_gt_data_dict = {
                    "batch_vcs":sh_vcs,
                    "rendered_imgs":rendered_img, 
                    "gt_imgs":batch_imgs,
                    "mask_c3d":mask_c3b,
                    "proj_lm2ds":proj_lm2d,
                    "gt_lm2ds":gt_lm2ds
                }
                
                norm_code_info = {
                    "iden_codes":iden_codes, 
                    "text_codes":text_codes, 
                    "expr_codes":expr_codes
                }

                batch_loss_dict = self.loss_utils.calc_total_loss(
                    cur_illus = illu_codes,
                    **pred_and_gt_data_dict,
                    **norm_code_info,
                    lm_w=lm_w
                )
            
            optimizer.zero_grad()
            batch_loss_dict["total_loss"].backward()
            optimizer.step()

            # print("Step 1, Iter: %04d |"%iter_, convert_loss_dict_2_str(batch_loss_dict))
            # res_img = draw_res_img(rendered_img, batch_imgs, mask_c3b_backup, proj_lm2ds=proj_lm2d, gt_lm2ds=gt_lm2ds, num_per_row=3)
            # res_img  = res_img[:, :, ::-1] 

            # cv2.imwrite("./temp_res/opt_imgs_2/opt_image_res%03d.png" % iter_, cv2.resize(res_img, (0,0), fx=0.5, fy=0.5))

        w2c_Tvecs = torch.bmm(w2c_Rmats, c2l_Tvecs.view(-1, 3, 1)).view(-1, 3) + w2c_Tvecs.view(-1, 3)
        w2c_Rmats = torch.bmm(w2c_Rmats, c2l_Rmats)
        # w2c_Tvecs = bmm_self_define_dim3(w2c_Rmats, c2l_Tvecs.view(-1, 3, 1)).view(-1, 3) + w2c_Tvecs.view(-1, 3)
        # w2c_Rmats = bmm_self_define_dim3(w2c_Rmats, c2l_Rmats)

        self.save_res(iden_codes, expr_codes, text_codes, illu_codes, w2c_Rmats, w2c_Tvecs, inmats, base_names)


    def save_res(self, iden_code, expr_code, text_code, illu_code, w2c_Rmats, w2c_Tvecs, inmats, base_names):
        iden_expr_text_illu_code = torch.cat([iden_code, expr_code, text_code, illu_code], dim=-1).detach().cpu()
        w2c_Rmats = w2c_Rmats.detach().cpu()
        w2c_Tvecs = w2c_Tvecs.detach().cpu()
        ori_inmats = inmats.detach().cpu()
        
        for cnt, str_name in enumerate(base_names):
            cur_code = iden_expr_text_illu_code[cnt]
            cur_w2c_Rmat = w2c_Rmats[cnt]
            cur_w2c_Tvec = w2c_Tvecs[cnt]
            inmat = ori_inmats[cnt]
            inmat[:2] /= self.data_utils.lm_scale
            cur_c2w_Rmat = cur_w2c_Rmat.t()
            cur_c2w_Tvec = -(cur_c2w_Rmat.mm(cur_w2c_Tvec.view(3, 1)))
            cur_c2w_Tvec = cur_c2w_Tvec.view(3)

            inv_inmat = torch.eye(3, dtype=torch.float32)
            inv_inmat[0, 0] = 1.0 / inmat[0, 0]
            inv_inmat[1, 1] = 1.0 / inmat[1, 1]
            inv_inmat[0, 2] = - (inv_inmat[0, 0] * inmat[0, 2])
            inv_inmat[1, 2] = - (inv_inmat[1, 1] * inmat[1, 2])

            # print(inmat)
            res = {
                "code": cur_code,
                "w2c_Rmat":cur_w2c_Rmat, 
                "w2c_Tvec":cur_w2c_Tvec, 
                "inmat":inmat, 
                "c2w_Rmat":cur_c2w_Rmat, 
                "c2w_Tvec":cur_c2w_Tvec, 
                "inv_inmat":inv_inmat, 
            }
            
            save_path = join(self.img_dir, str_name + "_nl3dmm.pkl")            
            with open(save_path, "wb") as f:
                pkl.dump(res, f)


    def main_process(self):
        total_sample_num = self.data_utils.total_sample_num
        
        loop_bar = tqdm(range(0, total_sample_num, self.batch_size), desc="Fitting 3DMM")
        for id_s in loop_bar:
            if id_s + self.batch_size < total_sample_num:
                idx_list = [x for x in range(id_s, id_s + self.batch_size)]
            else:
                idx_list = [x for x in range(id_s, total_sample_num)]
            
            temp_data = self.data_utils.load_batch_sample(idx_list)
            self.opt_batch_data(**temp_data)
            
            # exit(0)
            
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='The code for generating 3DMM parameters and camera parameters.')
    # parser.add_argument("--gpu_id", type=int, default=0)
    
    parser.add_argument("--img_size", type=int, required=True, help="the size of the input image.")
    parser.add_argument("--intermediate_size", type=int, required=True, help="Before fitting, the input image is resized as [intermediate_size, intermediate_size]")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--img_dir", type=str, required=True)
    args = parser.parse_args()


    tt = FittingNL3DMM(img_size=args.img_size, intermediate_size=args.intermediate_size, 
                        gpu_id=0, batch_size=args.batch_size, img_dir=args.img_dir)
    tt.main_process()



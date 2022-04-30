import torch
import torch.nn.functional as F
import numpy as np


class FittingNL3DMMLossUtils(object):
    def __init__(self) -> None:
        self.build_info()
    

    @staticmethod
    def photo_loss(pred_img, gt_img, mask_c3b):
        loss = F.mse_loss(pred_img[mask_c3b], gt_img[mask_c3b])
        return loss
    
    
    def build_info(self):
        skinmask = np.load("ConfigFiles/skinmask.npy")
        self.skinmask = torch.from_numpy(skinmask).view(1, -1, 1) #[1, n_v, 1]

        lm_weight = torch.ones(68, dtype=torch.float32)
        lm_weight[28:31] = 20
        lm_weight[-8:] = 20
        
        lm_weight = lm_weight / (lm_weight.sum(dim=0, keepdim=True))
        self.lm_weight = lm_weight.view(1, 68)
    
    def skin_loss(self, vert_colors):
        
        skinmask = self.skinmask.to(vert_colors.device)
        valid_v_num = torch.sum(skinmask)
        batch_size = vert_colors.size(0)
        colors = vert_colors / 255.0
        
        color_mean = torch.sum(colors * skinmask, dim=1, keepdim=True) / valid_v_num
        loss = torch.sum(torch.square(colors - color_mean) * skinmask) / (batch_size * valid_v_num)
        
        return loss
    
    
    def lm2d_loss(self, pred_lms_o, gt_lms_o, img_h = 400.0, img_w = 300.0):
        
        # pred_lms = pred_lms_o.clone()
        # gt_lms = gt_lms_o.clone()
        
        weight = self.lm_weight.to(pred_lms_o.device) #[1, 68]
        loss = torch.sum((pred_lms_o - gt_lms_o)**2, dim=-1) * weight
        loss = torch.sum(loss) / (pred_lms_o.shape[0] * pred_lms_o.shape[1])

        return loss
    

    @staticmethod
    def gamma_loss(gamma):

        gamma = gamma.reshape(-1, 9, 3)
        gamma_mean = torch.mean(gamma, dim=2, keepdims=True)
        gamma_loss = torch.mean(torch.square(gamma - gamma_mean))

        return gamma_loss


    @staticmethod
    def regu_cam_offset_loss(delta_eulur, delta_tvec):
        
        loss = torch.mean(delta_eulur * delta_eulur) + torch.mean(delta_tvec * delta_tvec)
        return loss
    
    
    @staticmethod
    def regu_illu_offset_loss(delta_illus):
        return torch.mean(delta_illus * delta_illus)
    
    
    @staticmethod
    def regu_code_loss(iden_codes, expr_codes, text_codes, **kwargs):
        loss = torch.mean(iden_codes * iden_codes) * 2.5 + torch.mean(expr_codes * expr_codes) * 2.5  +  torch.mean(text_codes * text_codes) * 2.5
        return loss
        
        
    def calc_total_loss(self, 
                        batch_vcs, 
                        cur_illus, 
                        rendered_imgs, gt_imgs, mask_c3d, 
                        proj_lm2ds, gt_lm2ds, 
                        iden_codes, text_codes, expr_codes,
                        lm_w
                        ):
    
        img_loss = self.photo_loss(rendered_imgs, gt_imgs, mask_c3d)
        lm_loss = self.lm2d_loss(proj_lm2ds, gt_lm2ds)
        
        illu_loss_regu_mean = self.gamma_loss(cur_illus)
        # cam_regu_loss = self.regu_cam_offset_loss(delta_eulur, delta_tvec)
        code_regu_loss = self.regu_code_loss(iden_codes, expr_codes, text_codes)
        skin_loss = self.skin_loss(batch_vcs)
        
        total_loss = img_loss * 10.0 + \
                     lm_loss * lm_w + \
                     illu_loss_regu_mean * 0.01 + \
                     code_regu_loss * 0.001 + \
                     skin_loss * 0.1
        
        loss_dict = {
            "img": img_loss,
            "lm": lm_loss,
            "illu_mean_loss": illu_loss_regu_mean,
            "code_regu": code_regu_loss, 
            "skin": skin_loss, 
            "total_loss":total_loss
        }
        
        return loss_dict
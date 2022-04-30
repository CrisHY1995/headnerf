import torch
from collections import OrderedDict
import numpy as np
import cv2
import math


def soft_load_model(net, pre_state_dict):
    
    if isinstance(pre_state_dict, str):
        pre_state_dict = torch.load(pre_state_dict,map_location="cpu")
    
    state_dict = net.state_dict()
    increment_state_dict = OrderedDict()
    
    for k, v in pre_state_dict.items():
        if k in state_dict:
            if v.size() == state_dict[k].size():
                increment_state_dict[k] = v

    state_dict.update(increment_state_dict)
    net.load_state_dict(state_dict)
    return net


def draw_res_img(rendered_imgs_0, ori_imgs, batch_mask, proj_lm2ds = None, gt_lm2ds = None, num_per_row = 1):
    num = rendered_imgs_0.size(0)
    res_list = []
    
    rendered_imgs = rendered_imgs_0.clone()
    rendered_imgs *= 255.0
    observed_imgs = ori_imgs.clone()
    observed_imgs *= 255.0
    
    if isinstance(rendered_imgs, torch.Tensor):
        rendered_imgs = rendered_imgs.detach().cpu().numpy() if rendered_imgs.is_cuda else rendered_imgs.numpy()
    
    if isinstance(observed_imgs, torch.Tensor):
        observed_imgs = observed_imgs.detach().cpu().numpy() if observed_imgs.is_cuda else observed_imgs.numpy()

    if isinstance(batch_mask, torch.Tensor):
        batch_mask = batch_mask.detach().cpu().numpy() if batch_mask.is_cuda else batch_mask.numpy()

    if gt_lm2ds is not None:
        if isinstance(gt_lm2ds, torch.Tensor):
            gt_lm2ds = gt_lm2ds.detach().cpu().numpy() if gt_lm2ds.is_cuda else gt_lm2ds

    if proj_lm2ds is not None:
        if isinstance(proj_lm2ds, torch.Tensor):
            proj_lm2ds = proj_lm2ds.detach().cpu().numpy() if proj_lm2ds.is_cuda else proj_lm2ds

    for cnt in range(num):
        re_img = rendered_imgs[cnt]
        mask = batch_mask[cnt]
        ori_img = observed_imgs[cnt]
        
        temp_img_1 = ori_img.copy()
        temp_img_2 = ori_img.copy()
        ori_img[mask] = re_img[mask]
        
        if gt_lm2ds is not None:
            lm2ds = gt_lm2ds[cnt]
            for lm2d in lm2ds:
                temp_img_2 = cv2.circle(temp_img_2, center=(int(lm2d[0]), int(lm2d[1])), radius=2, color=(255, 0, 0), thickness=1)
        
        if proj_lm2ds is not None:
            lm2ds = proj_lm2ds[cnt]
            for lm2d in lm2ds:
                temp_img_2 = cv2.circle(temp_img_2, center=(int(lm2d[0]), int(lm2d[1])), radius=2, color=(0, 0, 255), thickness=1)
                # ori_img = cv2.circle(ori_img, center=(int(lm2d[0]), int(lm2d[1])), radius=2, color=(0, 0, 255), thickness=1)
        
        img = np.concatenate([temp_img_1, temp_img_2, ori_img], axis=1)
        res_list.append(img)
    
    if num == 1:
        res = np.concatenate(res_list, axis=0)
    elif num_per_row != 1:
        n_rows = num // num_per_row

        last_res_imgs = []
        for cnt in range(n_rows):
            temp_res = np.concatenate(res_list[cnt * num_per_row:cnt * num_per_row + num_per_row], axis=1)
            last_res_imgs.append(temp_res)
        
        if num % num_per_row > 0:
            temp_img = np.ones_like(last_res_imgs[-1])
            temp_res = np.concatenate(res_list[n_rows * num_per_row:], axis=1)
            _, w, _ = temp_res.shape
            temp_img[:, :w, :] = temp_res
            last_res_imgs.append(temp_img)
        
        res = np.concatenate(last_res_imgs, axis=0)
    else:
        res = np.concatenate(res_list, axis=0)
    return res


def convert_loss_dict_2_str(loss_dict):
    res = ""
    for k,v in loss_dict.items():
        res = res + "{}:{:.04f}, ".format(k,v)
    res = res[:-2]
    return res


def put_text_alignmentcenter(img, img_size, text_str, color, offset_x):

    font = cv2.FONT_HERSHEY_COMPLEX
    textsize = cv2.getTextSize(text_str, font, 1, 2)[0]

    textX = (img_size - textsize[0]) // 2 + offset_x
    textY = (img_size - textsize[1])

    img = cv2.putText(img, text_str, (textX, textY ), font, 1, color, 2)
    
    return img




def eulurangle2Rmat(angles):
    """
    angles: (3, 1) or (1, 3)
    """
    angles = angles.reshape(-1)
    sinx = np.sin(angles[0])
    siny = np.sin(angles[1])
    sinz = np.sin(angles[2])
    cosx = np.cos(angles[0])
    cosy = np.cos(angles[1])
    cosz = np.cos(angles[2])
    
    mat_x = np.eye(3, dtype=np.float32)
    mat_y = np.eye(3, dtype=np.float32)
    mat_z = np.eye(3, dtype=np.float32)

    mat_x[1, 1] = cosx
    mat_x[1, 2] = -sinx
    mat_x[2, 1] = sinx
    mat_x[2, 2] = cosx

    mat_y[0, 0] = cosy
    mat_y[0, 2] = siny
    mat_y[2, 0] = -siny
    mat_y[2, 2] = cosy

    mat_z[0, 0] = cosz
    mat_z[0, 1] = -sinz
    mat_z[1, 0] = sinz
    mat_z[1, 1] = cosz

    res = mat_z.dot(mat_y.dot(mat_x))
    
    return res


def Rmat2EulurAng(Rmat):
    sy = math.sqrt(Rmat[0, 0] * Rmat[0, 0] + Rmat[1, 0] * Rmat[1, 0])

    if sy > 1e-6:
        x = math.atan2( Rmat[2, 1], Rmat[2, 2])
        y = math.atan2(-Rmat[2, 0], sy)
        z = math.atan2( Rmat[1, 0], Rmat[0,0])
    else:
        x = math.atan2(-Rmat[1, 2], Rmat[1, 1])
        y = math.atan2(-Rmat[2, 0], sy)
        z = 0
        
    return np.array([x, y, z])


if __name__ == "__main__":
    angle = np.array([0.00872665, 0.337, 0.113])
    rmat = eulurangle2Rmat(angle)
    res = Rmat2EulurAng(rmat)
    
    print(angle)
    print(res)
    
    

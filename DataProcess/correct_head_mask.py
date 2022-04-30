import cv2
import numpy as np


def erosion_hair_region(img_c1u, num_iter):
    
    fil = np.array([[     0, 0.25,    0], 
                    [  0.25, -1.0, 0.25],
                    [     0, 0.25,    0]])

    ddepth = cv2.CV_32FC1
    temp_img = img_c1u.copy()

    temp_img[temp_img == 1] = 3
    temp_img[temp_img == 2] = 1
    temp_img[temp_img == 3] = 2
    # cv2.imwrite("./temp_res/trans.png", temp_img)
    # exit(0)
    
    img_f = temp_img.astype(np.float32)

    for _ in range(num_iter):
        
        img_res = cv2.filter2D(img_f, ddepth, fil, borderType=cv2.BORDER_CONSTANT) 
        mask_reion = (img_c1u == 2) * (img_res < -0.01)
        
        img_f[mask_reion] = 0.0
        # cv2.imwrite("./temp_res/temp.pfm", img_f)
        # cv2.imwrite("./temp_res/img_res.pfm", img_res)
        # exit(0)
        # img_c1u[mask_reion] = 0
        # temp_img[mask_reion] = 0
    
    res = img_f.astype(np.uint8)
    res[res == 1] = 3
    res[res == 2] = 1
    res[res == 3] = 2
    
    return res


def extract_max_region(label_img, tar_value):
    mask_img = np.zeros_like(label_img)
    mask_img[label_img == tar_value] = 1
    num_labels, label_img = cv2.connectedComponents(mask_img, connectivity=8)
    
    max_label = -1
    max_area = -1.0
    
    for i in range(1, num_labels):
        cur_area = np.sum(label_img == i)
        if cur_area > max_area:
            max_label = i
            max_area = cur_area
    
    label_img[label_img != max_label] = 0
    label_img[label_img == max_label] = 255
    return label_img


def remover_free_block(img_c1u):
    temp_img = img_c1u.copy()
    
    temp_img[temp_img > 0.5] = 1
    label_img = extract_max_region(temp_img, 1)
    
    img_c1u[label_img != 255] = 0

    return img_c1u


def correct_hair_mask(mask_img):
    
    mask_img = remover_free_block(mask_img)
    mask_img = erosion_hair_region(mask_img, 7)
    mask_img = remover_free_block(mask_img)
    
    return mask_img
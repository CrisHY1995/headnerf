import os
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import torch
from BiSeNet import BiSeNet
from torchvision.transforms import transforms
import cv2
from tqdm import tqdm
import numpy as np
from glob import glob
from correct_head_mask import correct_hair_mask
import argparse


class GenHeadMask(object):
    def __init__(self, gpu_id) -> None:
        super().__init__()
        
        self.device = torch.device("cuda:%s" % gpu_id)
        self.model_path = "ConfigModels/faceparsing_model.pth"
        
        self.init_model()
        self.lut = np.zeros((256, ), dtype=np.uint8)
        self.lut[1:14] = 1
        self.lut[17] = 2
        
        #  ['skin', 'l_brow', 'r_brow', 'l_eye', 'r_eye', 'eye_g', 'l_ear', 'r_ear', 'ear_r',
        #     'nose', 'mouth', 'u_lip', 'l_lip', 'neck', 'neck_l', 'cloth', 'hair', 'hat']

    def init_model(self):
        n_classes = 19
        net = BiSeNet(n_classes=n_classes).to(self.device)
        net.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.net = net
        self.net.eval()

        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        
        
    def main_process(self, img_dir):
        img_path_list = [x for x in glob("%s/*.png" % img_dir) if "mask" not in x]  
        if len(img_path_list) == 0:
            print("Dir: %s does include any .png images." % img_dir)
            exit(0)
        img_path_list.sort()
        loop_bar = tqdm(img_path_list)
        loop_bar.set_description("Generate head masks")
        for img_path in loop_bar:
            save_path = img_path[:-4] + "_mask.png"
            
            bgr_img = cv2.imread(img_path)
            img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
            img = self.to_tensor(img)
            img = img.unsqueeze(0)
            img = img.to(self.device)
            with torch.set_grad_enabled(False):
                pred_res = self.net(img)
                out = pred_res[0]

            res = out.squeeze(0).cpu().numpy().argmax(0)
            res = res.astype(np.uint8)
            cv2.LUT(res, self.lut, res)
            
            res = correct_hair_mask(res)
            res[res != 0] = 255
            # temp_img = bgr_img.copy()
            # temp_img[res == 0] = 255
            
            # res = np.concatenate([bgr_img, temp_img], axis=1)
            cv2.imwrite(save_path, res)
            
            
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='The code for generating head mask images.')
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--img_dir", type=str, required=True)
    args = parser.parse_args()

    gpu_id = args.gpu_id
    img_dir = args.img_dir

    # assert len(sys.argv) == 2
    # img_dir = sys.argv[1]

    tt = GenHeadMask(gpu_id=gpu_id)
    tt.main_process(img_dir=img_dir)
    
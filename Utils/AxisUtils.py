import json
import cv2
import numpy as np
from tool_funcs import eulurangle2Rmat


class AxisUtils(object):
    
    def __init__(self, img_size) -> None:
        
        self.set_img_size(img_size)
        self.build_info()
        self.build_cam()
        
    
    def set_img_size(self, img_size):
        self.img_size = img_size   
        
        with open("ConfigFiles/cam_inmat_info_32x32.json", "r") as f:
            temp_dict = json.load(f)
        inmat = temp_dict["inmat"]
        
        scale = self.img_size / 32.0
        self.inmat = np.array(inmat)
        self.inmat[:2, :] *= scale
        
        self.fx = self.inmat[0, 0]
        self.fy = self.inmat[1, 1]
        self.cx = self.inmat[0, 2]
        self.cy = self.inmat[1, 2]


    def build_info(self):
        
        length = 0.75
        self.origin = np.array([0.0, 0.0, 0.0], dtype=np.float32).reshape(3, 1)
        self.axis_x = np.array([length, 0.0, 0.0], dtype=np.float32).reshape(3, 1)
        self.axis_y = np.array([0.0, length, 0.0], dtype=np.float32).reshape(3, 1)
        self.axis_z = np.array([0.0, 0.0, length], dtype=np.float32).reshape(3, 1)
    
    
    def build_cam(self):
        
        base_rmat = np.eye(3).astype(np.float32)
        base_rmat[1:, :] *= -1
        
        tv_z = 0.5 + 11.5
        base_tvec = np.zeros((3, 1), dtype=np.float32)
        base_tvec[2, 0] = tv_z
        
        self.base_w2c_Rmats = base_rmat
        self.base_w2c_Tvecs = base_tvec
        
        self.cur_Rmat = base_rmat.copy()
        self.cur_Tvec = base_tvec.copy()
        
    
    def calc_proj_pts(self, vp):
        cam_vp = self.cur_Rmat.dot(vp) + self.cur_Tvec
        u = self.fx * (cam_vp[0, 0] / cam_vp[2, 0]) + self.cx
        v = self.fy * (cam_vp[1, 0] / cam_vp[2, 0]) + self.cy
        
        return (int(u), int(v))
        

    def update_CurCam(self, pitch, yaw, roll):
        angles = np.zeros(3, dtype=np.float32)
                
        angles[0] = -pitch
        angles[1] = -yaw
        angles[2] = -roll
        
        delta_rmat = eulurangle2Rmat(angles)
        
        # c2w
        self.cur_Rmat = delta_rmat.dot(self.base_w2c_Rmats)
        self.cur_Tvec = delta_rmat.dot(self.base_w2c_Tvecs) 

        # w2c
        self.cur_Tvec = - (self.cur_Rmat.T.dot(self.cur_Tvec))
        self.cur_Rmat = self.cur_Rmat.T
        
    
    def generate_img(self, pitch, yaw, roll, img = None):
        self.update_CurCam(pitch, yaw, roll)
        
        if img is None:
            img = np.ones((self.img_size, self.img_size, 3), dtype=np.uint8) * 255
        
        pixel_o = self.calc_proj_pts(self.origin)
        pixel_x = self.calc_proj_pts(self.axis_x)
        pixel_y = self.calc_proj_pts(self.axis_y)
        pixel_z = self.calc_proj_pts(self.axis_z)
        
        img = img.copy()
        
        img = cv2.arrowedLine(img, pixel_o, pixel_x, color=(255, 0, 0), thickness=1)
        img = cv2.arrowedLine(img, pixel_o, pixel_y, color=(0, 255, 0), thickness=1)
        img = cv2.arrowedLine(img, pixel_o, pixel_z, color=(0, 0, 255), thickness=1)
        
        return img
    
        
if __name__ == "__main__":
    tt = AxisUtils()
    tt.generate_img()
    
    # a = np.random.rand(5, 5)
    # b = [[1, 2], [2, 3]]
    # c = a[b]
    # print(a)
    # print(c)
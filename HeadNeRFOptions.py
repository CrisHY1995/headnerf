import json
import os


class BaseOptions(object):
    def __init__(self, para_dict = None) -> None:
        super().__init__()
        
        self.bg_type = "white" # white: white bg, black: black bg.
        
        self.iden_code_dims = 100
        self.expr_code_dims = 79
        self.text_code_dims = 100
        self.illu_code_dims = 27

        self.auxi_shape_code_dims = 179
        self.auxi_appea_code_dims = 127
        
        # self.num_ray_per_img = 972 #972, 1200, 1452, 1728, 2028, 2352
        self.num_sample_coarse = 64
        self.num_sample_fine = 128

        self.world_z1 = 2.5
        self.world_z2 = -3.5
        self.mlp_hidden_nchannels = 384

        if para_dict is None:
            self.featmap_size = 64
            self.featmap_nc = 256       # nc: num_of_channel
            self.pred_img_size = 512
        else:
            self.featmap_size = para_dict["featmap_size"]
            self.featmap_nc = para_dict["featmap_nc"]
            self.pred_img_size = para_dict["pred_img_size"]

        
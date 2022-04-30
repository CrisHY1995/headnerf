import torch
import numpy as np
import torch.nn as nn
from HeadNeRFOptions import BaseOptions
import pickle as pkl


def parse_3dmm_file():

    with open("ConfigModels/nl3dmm_dict.pkl", "rb") as f:
        temp_dict = pkl.load(f)

    mu = temp_dict["mu"]
    b = temp_dict["b"]
    sig_id = temp_dict["sig_id"]
    sig_exp = temp_dict["sig_exp"]
    mu_tex = temp_dict["mu_tex"]
    b_tex = temp_dict["b_tex"]
    sig_tex = temp_dict["sig_tex"]
        
    return mu, b, sig_id, sig_exp, mu_tex, b_tex, sig_tex
        
        
class NonLinear_3DMM(nn.Module):
    def __init__(self, opt: BaseOptions):
        super(NonLinear_3DMM, self).__init__()
        self.opt = opt
        
        self.build_info()


    def build_info(self):
        self.point_num = 34650
        
        mu, b, sig_id, sig_exp, mu_tex, b_tex, sig_tex = parse_3dmm_file()
        
        self.geo_fc2 = nn.Linear(self.opt.iden_code_dims + self.opt.expr_code_dims, 1024)
        self.geo_fc3 = nn.Linear(1024, 3*self.point_num)
        self.geo_fc3.bias.data = torch.as_tensor(mu)
        self.geo_fc3.weight.data[:, 0:(self.opt.iden_code_dims + self.opt.expr_code_dims)] = torch.as_tensor(b).permute(1, 0)
        self.geo_fc3.weight.data[:, (self.opt.iden_code_dims + self.opt.expr_code_dims):500] = \
                            self.geo_fc3.weight.data[:, (self.opt.iden_code_dims + self.opt.expr_code_dims):500].fill_(0.001)
        self.activate_opt = nn.ReLU()


    def get_geo(self, pca_para):
        feature = self.activate_opt(self.geo_fc2(pca_para))
        return self.geo_fc3(feature)


    # def get_tex(self, tex_para):
    #     feature = self.activate_opt(self.tex_fc1(tex_para))
    #     feature = self.activate_opt(self.tex_fc2(feature))
    #     return self.tex_fc3(feature)


    def forward(self, id_para, exp_para, scale = 1.0):
        pca_para = torch.cat((id_para, exp_para), 1)
        geometry = self.get_geo(pca_para).reshape(-1, self.point_num, 3)
        return geometry * scale


    # def forawrd(self, norm_id_para, norm_exp_para, norm_tex_para):
    #     geometry = self.forward_geo(norm_id_para, norm_exp_para)
    #     texture = self.forward_tex(norm_tex_para)
    #     return geometry, texture


    # def forward_tex(self, tex_para):
    #     texture = self.get_tex(tex_para).reshape(-1, self.point_num, 3)
    #     return texture



class Linear_3DMM(nn.Module):
    def __init__(self, opt: BaseOptions):
        
        super(Linear_3DMM, self).__init__()
        self.opt = opt
        self.build_info()
        
    
    def build_info(self):
        
        self.point_num = 34650
        
        mu, b, sig_id, sig_exp, mu_tex, b_tex, sig_tex = parse_3dmm_file()
        
        self.register_buffer("mu",      torch.as_tensor(mu))
        self.register_buffer("b",       torch.as_tensor(b))
        self.register_buffer("sig_id",  torch.as_tensor(sig_id))
        self.register_buffer("sig_exp", torch.as_tensor(sig_exp))
        self.register_buffer("mu_tex",  torch.as_tensor(mu_tex))
        self.register_buffer("b_tex",   torch.as_tensor(b_tex))
        self.register_buffer("sig_tex", torch.as_tensor(sig_tex))

    
    # def get_geo(self, pca_para):
    #     return torch.mm(pca_para, self.b) + self.mu


    def get_tex(self, tex_para):
        return torch.mm(tex_para, self.b_tex) + self.mu_tex


    # def forward_geo(self, norm_id_para, norm_exp_para):
    #     id_para = norm_id_para*self.sig_id
    #     exp_para = norm_exp_para*self.sig_exp
    #     pca_para = torch.cat((id_para, exp_para), 1)
    #     geometry = self.get_geo(pca_para).reshape(-1, self.point_num, 3)
    #     return geometry


    # def forawrd(self, norm_id_para, norm_exp_para, norm_tex_para):
    #     geometry = self.forward_geo(norm_id_para, norm_exp_para)
    #     texture = self.forward_tex(norm_tex_para)
    #     return geometry, texture


    def forward(self, norm_tex_para):
        tex_para = norm_tex_para*self.sig_tex
        texture = self.get_tex(tex_para).reshape(-1, self.point_num, 3)
        return texture

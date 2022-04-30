import torch
import torch.nn as nn
import torch.nn.functional as F


class Embedder(nn.Module):
    def __init__(self, N_freqs, include_input, input_dims = 3) -> None:
        super().__init__()

        self.log_sampling = True
        self.periodic_fns = [torch.sin, torch.cos]
        self.max_freq = N_freqs - 1
        self.N_freqs = N_freqs
        self.include_input = include_input
        self.input_dims = input_dims

        self._Pre_process()


    def _Pre_process(self):
        
        embed_fns = []
        # out_dim = 0

        if self.include_input:
            embed_fns.append(lambda x: x)
            # out_dim += self.input_dims

        if self.log_sampling:
            freq_bands = 2.**torch.linspace(0., self.max_freq, steps=self.N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**self.max_freq, steps=self.N_freqs)
        
        #(sin(2^0 * \pi *p), cos(2^0 * \pi *p))
        for freq in freq_bands:
            for p_fn in self.periodic_fns:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                # out_dim += self.input_dims
        self.embed_fns = embed_fns
        # self.out_dim = out_dim
        

    def forward(self, x):
        """
        x: [B, 3, N_1, N_2]
        """

        res = [fn(x) for fn in self.embed_fns]
        res = torch.cat(res, dim=1)

        return res
    
    

class GenSamplePoints(nn.Module):
    def __init__(self, opt):
        super().__init__()

        self.world_z1 = opt.world_z1
        self.world_z2 = opt.world_z2
        self.n_sample_fg = opt.num_sample_coarse


    @staticmethod
    def _calc_sample_points_by_zvals(zvals, batch_ray_o, batch_ray_d, batch_ray_l, disturb):
        """
        zvals      :[B, N_r, N_p + 1]
        batch_ray_o:[B, 3,   N_r    ,   1]
        batch_ray_d:[B, 3,   N_r    ,   1]
        batch_ray_l:[B, 1,   N_r    ,   1]
        """

        if disturb:
            mids = 0.5 * (zvals[:, :, 1:] + zvals[:, :, :-1])
            upper = torch.cat([mids, zvals[:, :, -1:]], dim=-1)
            lower = torch.cat([zvals[:, :, :1], mids], dim=-1)
            t_rand = torch.rand_like(zvals)
            zvals = lower + (upper - lower) * t_rand

        z_dists = zvals[:, :, 1:] - zvals[:, :, :-1] #[B, N_r, N_p]
        z_dists = (z_dists.unsqueeze(1)) * batch_ray_l #[B, 1, N_r, N_p]

        zvals = zvals[:, :, :-1]   #[B, N_r, N_p]
        zvals = zvals.unsqueeze(1) #[B, 1, N_r, N_p]

        sample_pts = batch_ray_o + batch_ray_d * batch_ray_l * zvals #[B, 3, N_r, N_p]
        
        n_sample = zvals.size(-1)
        sample_dirs = batch_ray_d.expand(-1, -1, -1, n_sample)
        
        # # import time
        # # # import shutil
        # # # import os
        # save_dir = "./temp_res/pts"
        # # if os.path.exists(save_dir):
        # #     shutil.rmtree(save_dir)
        # # os.mkdir(save_dir)
        
        # # t = time.time()
        # pts_info = sample_pts[0].detach().view(3, -1).t()
        # with open("%s/%s.obj"%(save_dir, "coarse"), "w") as f:
        #     ll = pts_info.size(0)
        #     for i in range(0, ll, 30):
        #         f.write("v %f %f %f\n"%(pts_info[i, 0], pts_info[i, 1], pts_info[i, 2]))
        
        res = {
            "pts":sample_pts,        #[B, 3, N_r, N_p]
            "dirs":sample_dirs,      #[B, 3, N_r, N_p]
            "zvals":zvals,           #[B, 1, N_r, N_p]
            "z_dists":z_dists,          #[B, 1, N_r, N_p]
            "batch_ray_o":batch_ray_o,  #[B, 3, N_r, 1]
            "batch_ray_d":batch_ray_d,  #[B, 3, N_r, 1]
            "batch_ray_l":batch_ray_l,  #[B, 1, N_r, 1]
        }

        return res

    def _calc_sample_points(self, batch_ray_o, batch_ray_d, batch_ray_l, disturb):
        """
        batch_ray_o:[B, 3, N_r]
        batch_ray_d:[B, 3, N_r]
        batch_ray_l:[B, 1, N_r]
        """

        rela_z1 = batch_ray_o[:, -1, :] - self.world_z1 #[B, N_r]
        rela_z2 = batch_ray_o[:, -1, :] - self.world_z2 #[B, N_r]
        
        # rela_z1 = self.world_z1 * torch.ones_like(batch_ray_o[:, -1, :]) #[B, N_r]
        # rela_z2 = self.world_z2 * torch.ones_like(batch_ray_o[:, -1, :]) #[B, N_r]
        rela_z1 = rela_z1.unsqueeze(-1) #[B, N_r, 1]
        rela_z2 = rela_z2.unsqueeze(-1) #[B, N_r, 1]

        data_type = batch_ray_o.dtype
        data_device = batch_ray_o.device
        
        batch_ray_o = batch_ray_o.unsqueeze(-1)
        batch_ray_d = batch_ray_d.unsqueeze(-1)
        batch_ray_l = batch_ray_l.unsqueeze(-1)

        t_vals_fg = torch.linspace(0.0, 1.0, steps=self.n_sample_fg + 1, dtype=data_type, 
                                                        device=data_device).view(1, 1, self.n_sample_fg + 1)
        sample_zvals_fg = rela_z1 * (1.0 - t_vals_fg) + rela_z2 * t_vals_fg  #[B, N_r, N_p + 1]
        sample_dict_fg = self._calc_sample_points_by_zvals(sample_zvals_fg, batch_ray_o, batch_ray_d, batch_ray_l, disturb)
        
        return sample_dict_fg

    def forward(self, batch_xy, batch_Rmat, batch_Tvec, batch_inv_inmat, disturb):

        temp_xyz = F.pad(batch_xy, [0, 0, 0, 1, 0, 0], mode="constant", value=1.0)
        
        ray_d = batch_Rmat.bmm(batch_inv_inmat.bmm(temp_xyz))
        # ray_d = bmm_self_define_dim3(batch_Rmat, bmm_self_define_dim3(batch_inv_inmat, temp_xyz))
        ray_l = torch.norm(ray_d, dim=1, keepdim=True)
        ray_d = ray_d/ray_l
        ray_l = -1.0 / ray_d[:, -1:, :]

        batch_size, _, num_ray = batch_xy.size()
        ray_o = batch_Tvec.expand(batch_size, 3, num_ray)
        
        fg_sample_dict = self._calc_sample_points(ray_o, ray_d, ray_l, disturb)
        return fg_sample_dict


class FineSample(nn.Module):
    def __init__(self, opt) -> None:
        super().__init__()
        self.n_sample = opt.num_sample_fine + 1
        # self.world_z2 = opt.world_z2
    
    @staticmethod
    def _calc_sample_points_by_zvals(zvals, batch_ray_o, batch_ray_d, batch_ray_l):
        """
        zvals      :[B, N_r, N_p + 1]
        batch_ray_o:[B, 3,   N_r    ,   1]
        batch_ray_d:[B, 3,   N_r    ,   1]
        batch_ray_l:[B, 1,   N_r    ,   1]
        """
        
        z_dists = zvals[:, :, 1:] - zvals[:, :, :-1] #[B, N_r, N_p]
        z_dists = (z_dists.unsqueeze(1)) * batch_ray_l #[B, 1, N_r, N_p]

        zvals = zvals[:, :, :-1]   #[B, N_r, N_p]
        zvals = zvals.unsqueeze(1) #[B, 1, N_r, N_p]

        sample_pts = batch_ray_o + batch_ray_d * batch_ray_l * zvals #[B, 3, N_r, N_p]
        
        n_sample = zvals.size(-1)
        sample_dirs = batch_ray_d.expand(-1, -1, -1, n_sample)

        # # import time
        # # # import shutil
        # # # import os
        # save_dir = "./temp_res/pts"
        # # if os.path.exists(save_dir):
        # #     shutil.rmtree(save_dir)
        # # os.mkdir(save_dir)
        
        # # t = time.time()
        # pts_info = sample_pts[0].detach().view(3, -1).t()
        # with open("%s/%s.obj"%(save_dir, "fine"), "w") as f:
        #     ll = pts_info.size(0)
        #     for i in range(0, ll, 30):
        #         f.write("v %f %f %f\n"%(pts_info[i, 0], pts_info[i, 1], pts_info[i, 2]))
        # exit(0)
        
        res = {
            "pts":sample_pts,        #[B, 3, N_r, N_p]
            "dirs":sample_dirs,      #[B, 3, N_r, N_p]
            "zvals":zvals,           #[B, 1, N_r, N_p]
            "z_dists":z_dists,  #[B, 1, N_r, N_p]
        }

        return res

    def forward(self, batch_weight, coarse_sample_dict, disturb):
        
        NFsample = self.n_sample
        coarse_zvals = coarse_sample_dict["zvals"] #[B, 1, N_r, N_c]

        # coarse_zvals = F.pad(coarse_zvals, pad=[0, 1, 0, 0, 0, 0, 0, 0] , mode="constant", value=self.world_z2)

        temp_weight = batch_weight[:, :, :, 1:-1].detach()
        batch_size, _, num_ray, temp_NCsample = temp_weight.size() # temp_Csample = N_c - 2
        temp_weight = temp_weight.view(-1, temp_NCsample) # [N_t, N_c - 2]

        x = temp_weight + 1e-5
        pdf = temp_weight/torch.sum(x, dim=-1, keepdim=True)
        cdf = torch.cumsum(pdf, dim=-1)
        cdf = F.pad(cdf, pad=[1, 0, 0, 0], mode="constant", value=0.0) #[N_t, N_c - 1]
        cdf = cdf.contiguous()
        
        num_temp = cdf.size(0) #B*N_r
        if disturb:
            uniform_sample = torch.rand(num_temp, NFsample, device=batch_weight.device, dtype=batch_weight.dtype) #[N_t, N_f]
        else:
            uniform_sample = torch.linspace(0.0, 1.0, steps=NFsample, device=batch_weight.device, dtype=batch_weight.dtype).view(1, NFsample).expand(num_temp, NFsample)
        uniform_sample = uniform_sample.contiguous() #[N_t, N_f]

        inds = torch.searchsorted(cdf, uniform_sample, right=True) #[N_t, N_f]
        below = torch.max(torch.zeros_like(inds), inds - 1)
        above = torch.min(temp_NCsample * torch.ones_like(inds), inds)
        inds_g = torch.stack([below, above], dim=-1) #[N_t, N_f, 2]

        temp_coarse_vpz = coarse_zvals.view(num_temp, temp_NCsample + 2)
        bins = 0.5 * (temp_coarse_vpz[:, 1:] + temp_coarse_vpz[:, :-1]) #[N_t, N_c - 1]
        
        cdf_g = torch.gather(cdf.unsqueeze(1).expand(num_temp, NFsample, temp_NCsample + 1), 2, inds_g) #[N_t, n_f, 2]
        bins_g = torch.gather(bins.unsqueeze(1).expand(num_temp, NFsample, temp_NCsample + 1), 2, inds_g) #[N_t, n_f, 2]

        denom = cdf_g[:, :, 1] - cdf_g[:, :, 0] #[N_t, N_f]
        denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
        t = (uniform_sample - cdf_g[:, :, 0]) / denom
        fine_sample_vz = bins_g[:, :, 0] + t * (bins_g[:, :, 1] - bins_g[:, :, 0]) #[N_t, N_f]

        fine_sample_vz, _ = torch.sort(torch.cat([temp_coarse_vpz, fine_sample_vz], dim=-1), dim=-1) #[N_t, N_f + N_c]
        fine_sample_vz = fine_sample_vz.view(batch_size, num_ray, NFsample + temp_NCsample + 2)

        res = self._calc_sample_points_by_zvals(
            fine_sample_vz,
            coarse_sample_dict["batch_ray_o"],
            coarse_sample_dict["batch_ray_d"],
            coarse_sample_dict["batch_ray_l"],
        )

        return res


class CalcRayColor(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def _calc_alpha(batch_density, batch_dists):

        res = 1.0 - torch.exp(- batch_density * batch_dists)
        return res

    @staticmethod
    def _calc_weight(batch_alpha):
        """
        batch_alpha:[B, 1, N_r, N_p]
        """
        x = 1.0 - batch_alpha + 1e-10
        x = F.pad(x, [1, 0, 0, 0, 0, 0, 0, 0], mode="constant", value=1.0)
        x = torch.cumprod(x, dim=-1)

        res = batch_alpha * x[:, :, :, :-1]
        
        return res

    def forward(self, fg_vps, batch_rgb, batch_density, batch_dists, batch_z_vals):

        """
        batch_rgb: [B, 3, N_r, N_p]
        batch_density: [B, 1, N_r, N_p]
        batch_dists: [B, 1, N_r, N_p]
        batch_z_vals:[B, N_r, N_p]
        """

        batch_alpha = self._calc_alpha(batch_density, batch_dists)
        batch_weight = self._calc_weight(batch_alpha)

        rgb_res = torch.sum(batch_weight * batch_rgb, dim=-1) #[B, 3, N_r]
        depth_res = torch.sum(batch_weight * batch_z_vals, dim=-1) #[B, 1, N_r]

        acc_weight = torch.sum(batch_weight, dim= -1) #[B, 1, N_r]
        bg_alpha = 1.0 - acc_weight
        
        return rgb_res, bg_alpha, depth_res, batch_weight
import torch.nn as nn
import torch
from math import log, log2

if __name__ == "__main__":
    from PixelShuffleUpsample import PixelShuffleUpsample, Blur
else:
    from NetWorks.PixelShuffleUpsample import PixelShuffleUpsample, Blur
    

class NeuralRenderer(nn.Module):

    def __init__(
            self, bg_type = "white", feat_nc=256, out_dim=3, final_actvn=True, min_feat=32, featmap_size=32, img_size=256, 
            **kwargs):
        super().__init__()
        # assert n_feat == input_dim
        
        self.bg_type = bg_type
        self.featmap_size = featmap_size
        self.final_actvn = final_actvn
        # self.input_dim = input_dim
        self.n_feat = feat_nc
        self.out_dim = out_dim
        self.n_blocks = int(log2(img_size) - log2(featmap_size))
        self.min_feat = min_feat
        self._make_layer()
        self._build_bg_featmap()
        

    def _build_bg_featmap(self):
        
        if self.bg_type == "white":
            bg_featmap = torch.ones((1, self.n_feat, self.featmap_size, self.featmap_size), dtype=torch.float32)
        elif self.bg_type == "black":
            bg_featmap = torch.zeros((1, self.n_feat, self.featmap_size, self.featmap_size), dtype=torch.float32)
        else:
            bg_featmap = None
            print("Error bg_type")
            exit(0)
        
        self.register_parameter("bg_featmap", torch.nn.Parameter(bg_featmap))


    def get_bg_featmap(self):
        return self.bg_featmap
    

    def _make_layer(self):
        self.feat_upsample_list = nn.ModuleList(
            [PixelShuffleUpsample(max(self.n_feat // (2 ** (i)), self.min_feat)) for i in range(self.n_blocks)]
        )
        
        self.rgb_upsample = nn.Sequential(nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=False), Blur())

        self.feat_2_rgb_list = nn.ModuleList(
                [nn.Conv2d(self.n_feat, self.out_dim, 1, 1, padding=0)] +
                [nn.Conv2d(max(self.n_feat // (2 ** (i + 1)), self.min_feat),
                           self.out_dim, 1, 1, padding=0) for i in range(0, self.n_blocks)]
            )

        self.feat_layers = nn.ModuleList(
            [nn.Conv2d(max(self.n_feat // (2 ** (i)), self.min_feat),
                       max(self.n_feat // (2 ** (i + 1)), self.min_feat), 1, 1,  padding=0)
                for i in range(0, self.n_blocks)]
        )
        
        self.actvn = nn.LeakyReLU(0.2, inplace=True)
        
        
    def forward(self, x):
        
        # res = []
        rgb = self.rgb_upsample(self.feat_2_rgb_list[0](x))
        # res.append(rgb)
        net = x
        for idx in range(self.n_blocks):
            hid = self.feat_layers[idx](self.feat_upsample_list[idx](net))
            net = self.actvn(hid)
            
            rgb = rgb + self.feat_2_rgb_list[idx + 1](net)
            if idx < self.n_blocks - 1:
                rgb = self.rgb_upsample(rgb)
                # res.append(rgb)
        
        if self.final_actvn:
            rgb = torch.sigmoid(rgb)
        # res.append(rgb)
        
        return rgb
    
    
if __name__ == "__main__":
    tt = NeuralRenderer(img_size=512, featmap_size=64)
    a = torch.rand(2, 256, 64, 64)
    b = tt(a)
    
    # for s in b:
    #     print(s.size())
    print(b.size())
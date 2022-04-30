
import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia.filters import filter2d


class Blur(nn.Module):
    def __init__(self):
        super().__init__()
        f = torch.Tensor([1, 2, 1])
        self.register_buffer('f', f)


    def forward(self, x):
        f = self.f
        f = f[None, None, :] * f[None, :, None]
        return filter2d(x, f, normalized=True)


class PixelShuffleUpsample(nn.Module):
    def __init__(self, in_feature):
        super().__init__()
        self.in_feature = in_feature
        # self.out_feature = out_feature
        self._make_layer()
        

    def _make_layer(self):
        self.layer_1 = nn.Conv2d(self.in_feature, self.in_feature * 2, 1, 1, padding=0)
        self.layer_2 = nn.Conv2d(self.in_feature * 2, self.in_feature * 4, 1, 1, padding=0)
        self.blur_layer = Blur()
        self.actvn = nn.LeakyReLU(0.2, inplace=True)


    def forward(self, x:torch.Tensor):
        y = x.repeat(1, 4, 1, 1)
        out = self.actvn(self.layer_1(x))
        out = self.actvn(self.layer_2(out))
        
        out = out + y
        out = F.pixel_shuffle(out, 2)
        out = self.blur_layer(out)
        
        return out
    
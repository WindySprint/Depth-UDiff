import functools
import math
import torch
import torch.nn as nn
import models.blocks as bs
from models import davm
import torch.nn.functional as F

from utils import data_util

# Non-uniform Lightness Enhancement Network
class NLENet(nn.Module):
    def __init__(self, n_feats=16):
        super(NLENet, self).__init__()
        self.n_feats = n_feats
        self.l_ini = nn.Sequential(nn.Conv2d(1, self.n_feats, 3, 1, 1),
                                   nn.LeakyReLU(0.2, True),
                                   nn.Conv2d(self.n_feats, self.n_feats, 3, 1, 1, bias=True))
        self.rb1 = bs.RB(self.n_feats)
        self.l_down1 = nn.Sequential(nn.Conv2d(self.n_feats, self.n_feats * 2, 3, 2, 1),
                                      nn.LeakyReLU(0.2, True))
        self.l_down2 = nn.Sequential(nn.Conv2d(self.n_feats * 2, self.n_feats * 4, 3, 2, 1),
                                      nn.LeakyReLU(0.2, True))

        basic_block1 = functools.partial(bs.RB, self.n_feats * 2)
        basic_block2 = functools.partial(bs.RB, self.n_feats * 4)
        basic_block3 = functools.partial(bs.RB, self.n_feats * 2)
        self.mrb1 = bs.make_layer(basic_block1, 2)
        self.mrb2 = bs.make_layer(basic_block2, 4)
        self.mrb3 = bs.make_layer(basic_block3, 2)

        self.dafm1 = davm.DAFM(dim=n_feats,
                               d_state=math.ceil(n_feats / 2),
                               attn_drop=0., drop_path=0.,
                               norm_layer=nn.LayerNorm)
        self.dafm2 = davm.DAFM(dim=n_feats * 2,
                         d_state=math.ceil(n_feats * 2 / 4),
                         attn_drop=0., drop_path=0.,
                         norm_layer=nn.LayerNorm)

        self.l_up1 = nn.Sequential(nn.PixelShuffle(2),
                                   nn.LeakyReLU(0.2, True),
                                   nn.Conv2d(self.n_feats, self.n_feats * 2, 3, 1, 1))
        self.sosb1 = bs.SOSB(self.n_feats * 2)

        self.l_up2 = nn.Sequential(nn.PixelShuffle(2),
                                   nn.LeakyReLU(0.2, True),
                                   nn.Conv2d(self.n_feats // 2, self.n_feats, 3, 1, 1))
        self.sosb2 = bs.SOSB(self.n_feats)

        self.rb2 = bs.RB(self.n_feats)
        self.l_last = nn.Sequential(nn.Conv2d(self.n_feats, self.n_feats, 3, 1, 1, bias=True),
                                    nn.LeakyReLU(0.2, True),
                                    nn.Conv2d(self.n_feats, 1, 3, 1, 1, bias=True))

    def forward(self, x_lab, d):
        l = x_lab[:, :1, :, :]
        le1 = self.l_ini(l)
        le1 = self.rb1(le1)

        le2 = self.l_down1(le1)
        le2 = self.mrb1(le2)

        le3 = self.l_down2(le2)

        l3 = self.mrb2(le3)

        le1 = self.dafm1(le1, d)
        le2 = self.dafm2(le2, d)

        ld2 = self.l_up1(l3)
        l_sos2 = self.sosb1(ld2, le2)
        ld2 = self.mrb3(l_sos2)

        ld1 = self.l_up2(ld2)
        l_sos1 = self.sosb2(ld1, le1)

        ld1 = self.rb2(l_sos1)
        l = self.l_last(ld1)
        return l


##### Color Channel Correction Network
class C3Net(nn.Module):
    def __init__(self, n_feats=16):
        super(C3Net, self).__init__()
        self.n_feats = n_feats
        self.ab_cat = nn.Sequential(nn.Conv2d(4, self.n_feats, 3, 1, 1, bias=True),
                                    nn.LeakyReLU(0.2, True),
                                    nn.Conv2d(self.n_feats, self.n_feats, 3, 1, 1, bias=True))
        self.ab_down1 = nn.Sequential(nn.Conv2d(self.n_feats, self.n_feats * 2, 3, 2, 1),
                                      nn.LeakyReLU(0.2, True))
        self.ab_down2 = nn.Sequential(nn.Conv2d(self.n_feats * 2, self.n_feats * 4, 3, 2, 1),
                                      nn.LeakyReLU(0.2, True))

        self.a_ini = nn.Sequential(nn.Conv2d(1, self.n_feats * 4, 3, 1, 1),
                                   nn.LeakyReLU(0.2, True))
        self.a_cat = nn.Sequential(nn.Conv2d(self.n_feats * 8, self.n_feats * 4, 3, 1, 1, bias=True),
                                   nn.LeakyReLU(0.2, True),
                                   nn.Conv2d(self.n_feats * 4, self.n_feats * 4, 3, 1, 1, bias=True))
        self.a_up1 = nn.Sequential(nn.PixelShuffle(2),
                                   nn.LeakyReLU(0.2, True),
                                   nn.Conv2d(self.n_feats, self.n_feats * 2, 3, 1, 1))
        self.a_up2 = nn.Sequential(nn.PixelShuffle(2),
                                   nn.LeakyReLU(0.2, True),
                                   nn.Conv2d(self.n_feats // 2, self.n_feats, 3, 1, 1))
        self.a_last = nn.Sequential(nn.Conv2d(self.n_feats, self.n_feats, 3, 1, 1, bias=True),
                                    nn.LeakyReLU(0.2, True),
                                    nn.Conv2d(self.n_feats, 2, 3, 1, 1, bias=True))

        self.dgcfb1 = bs.DGCFB(self.n_feats * 2)
        self.dgcfb2 = bs.DGCFB(self.n_feats)

        self.b_ini = nn.Sequential(nn.Conv2d(1, self.n_feats * 4, 3, 1, 1),
                                   nn.LeakyReLU(0.2, True))
        self.b_cat = nn.Sequential(nn.Conv2d(self.n_feats * 8, self.n_feats * 4, 3, 1, 1, bias=True),
                                   nn.LeakyReLU(0.2, True),
                                   nn.Conv2d(self.n_feats * 4, self.n_feats * 4, 3, 1, 1, bias=True))
        self.b_up1 = nn.Sequential(nn.PixelShuffle(2),
                                   nn.LeakyReLU(0.2, True),
                                   nn.Conv2d(self.n_feats, self.n_feats * 2, 3, 1, 1))
        self.b_up2 = nn.Sequential(nn.PixelShuffle(2),
                                   nn.LeakyReLU(0.2, True),
                                   nn.Conv2d(self.n_feats // 2, self.n_feats, 3, 1, 1))
        self.b_last = nn.Sequential(nn.Conv2d(self.n_feats, self.n_feats, 3, 1, 1, bias=True),
                                    nn.LeakyReLU(0.2, True),
                                    nn.Conv2d(self.n_feats, 2, 3, 1, 1, bias=True))

        # Squeeze-Extract Global Gating Block
        self.segg = nn.Sequential(nn.Conv2d(3, self.n_feats, 3, 1, 1),
                                  nn.AdaptiveAvgPool2d(1),
                                  nn.Conv2d(self.n_feats, self.n_feats // 2, 1),
                                  nn.LeakyReLU(0.2, True),
                                  nn.Conv2d(self.n_feats // 2, 1, 1),
                                  nn.Sigmoid()
                                  )

    def forward(self, x, x_lab, d):
        ################# AB ##################
        ab = x_lab[:, 1:, :, :]
        ab1 = self.ab_cat(torch.cat((ab, ab * d), dim=1))
        ab2 = self.ab_down1(ab1)
        ab3 = self.ab_down2(ab2)

        a = x_lab[:, 1, :, :].unsqueeze(1)
        a = F.interpolate(a, size=[ab3.size(2), ab3.size(3)], mode='nearest')
        a = self.a_ini(a)
        a_ab = self.a_cat(torch.cat((ab3, a), dim=1))

        b = x_lab[:, 2, :, :].unsqueeze(1)
        b = F.interpolate(b, size=[ab3.size(2), ab3.size(3)], mode='nearest')
        b = self.b_ini(b)
        b_ab = self.b_cat(torch.cat((ab3, b), dim=1))

        a_ab = self.a_up1(a_ab) + ab2
        b_ab = self.b_up1(b_ab) + ab2
        a_ab, b_ab = self.dgcfb1(a_ab, b_ab, d)

        a_ab = self.a_up2(a_ab) + ab1
        b_ab = self.b_up2(b_ab) + ab1
        a_ab, b_ab = self.dgcfb2(a_ab, b_ab, d)

        a_ab = self.a_last(a_ab)
        b_ab = self.b_last(b_ab)

        w = self.segg(x)
        ab = (1 - w) * a_ab + w * b_ab
        return ab

##### Depth-wise Degradation Correction Network
class DDCNet(nn.Module):
    def __init__(self, n_feats=16):
        super(DDCNet, self).__init__()
        self.n_feats = n_feats

        self.nl_enhancer = NLENet(self.n_feats)
        self.cc_corrector = C3Net(self.n_feats)

    def forward(self, x, x_lab, d):
        l = self.nl_enhancer(x_lab, d)
        ab = self.cc_corrector(x, x_lab, d)

        lab = torch.cat([l, ab], dim=1)
        out = data_util.lab_to_rgb(lab)
        return out
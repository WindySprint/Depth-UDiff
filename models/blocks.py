import torch
import torch.nn as nn
import torch.nn.functional as F

##############--- Multi-blocks ---################
def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)

##############--- Residual Block (RB) ---################
class RB(nn.Module):
    def __init__(self, n_feats=16):
        super(RB, self).__init__()
        self.CLC1 = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(n_feats, n_feats, 3, 1, 1)
        )
        self.CLC2 = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(n_feats, n_feats, 3, 1, 1)
        )

    def forward(self, x):
        x1 = self.CLC2(self.CLC1(x))
        x = x + x1
        return x

##############--- Channel Attention Block (CAB) ---################
class CAB(nn.Module):

    def __init__(self, n_feats=16, squeeze_rate=8):
        super(CAB, self).__init__()
        self.AvgP = nn.AdaptiveAvgPool2d(1)
        self.MaxP = nn.AdaptiveMaxPool2d(1)
        self.CLC = nn.Sequential(
            nn.Conv2d(n_feats, n_feats // squeeze_rate, 1, bias=False),
            nn.LeakyReLU(),
            nn.Conv2d(n_feats // squeeze_rate, n_feats, 1, bias=False)
        )
        self.Sig = nn.Sigmoid()

    def forward(self, x):
        a = self.CLC(self.AvgP(x))
        m = self.CLC(self.MaxP(x))
        ca = self.Sig(a + m)
        out = x * ca
        return out

##############--- Dynamic Dual-Fusion Block (DDFB) ---################
class DDFB(nn.Module):
    def __init__(self, n_feats=16, squeeze_rate=4):
        super(DDFB, self).__init__()
        self.CBL = nn.Sequential(
            nn.Conv2d(2 * n_feats, n_feats, 3, 1, 1),
            nn.BatchNorm2d(n_feats),
            nn.LeakyReLU(0.2)
        )
        self.CSL1 = nn.Sequential(
            nn.Conv2d(n_feats, 1, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Sigmoid()
        )
        self.CAB = CAB(n_feats, squeeze_rate)
        self.CSL2 = nn.Sequential(
            nn.Conv2d(2 * n_feats, n_feats, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Sigmoid()
        )
        self.w = torch.nn.Parameter(torch.FloatTensor([1.0]), requires_grad=True)

    def forward(self, ie, me):
        ime = self.CBL(torch.cat((ie, me), dim=1))
        im = self.CSL1(ime) # [B, 1, H, W]
        imc = self.CAB(ime * im) # [B, C, 1, 1]
        imc = self.CSL2(torch.cat((ie * imc, me * imc), 1)) # [B, C, H, W]
        out = ie + self.w * imc  # [B, C, H, W]
        return out

##############--- Dynamic Triple-Fusion Block (DDFB) ---################
class DTFB(nn.Module):
    def __init__(self, n_feats=64, squeeze_rate=16):
        super(DTFB, self).__init__()
        self.CBLC = nn.Sequential(
            nn.Conv2d(256, n_feats, 3, 1, 1),
            nn.BatchNorm2d(n_feats),
            nn.LeakyReLU(0.2),
            nn.Conv2d(n_feats, n_feats, 3, 1, 1),
        )
        self.CL1 = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 3, 1, 1),
            nn.LeakyReLU(0.2)
        )
        self.CL2 = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 3, 1, 1),
            nn.LeakyReLU(0.2)
        )
        self.CAB = CAB(n_feats, squeeze_rate)
        self.RB1 = RB(n_feats)
        self.RB2 = RB(n_feats)
        self.refine_CLS = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 1, bias=False),
            nn.LeakyReLU(0.2),
            nn.Sigmoid()
        )

    def forward(self, ie, me, sam_f):
        sam_f = self.CBLC(sam_f)
        ie = self.CL1(ie)
        me = self.CL2(me)
        si = ie + ie * sam_f
        se = me + me * sam_f
        si = sam_f + self.CAB(si)
        se = sam_f + self.CAB(se)
        si = self.RB1(si)
        se = self.RB2(se)
        out = self.refine_CLS(si + se)
        return out

##############--- Depth-guided Channel Fusion Block (DGCFB) ---################
class DGCFB(nn.Module):
    def __init__(self, n_feats=64):
        super(DGCFB, self).__init__()
        self.d_ini = nn.Sequential(
            nn.Conv2d(1, n_feats, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(n_feats, n_feats, 3, 1, 1),
            nn.Sigmoid()
        )
        self.ab_ini = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(n_feats, n_feats, 3, 1, 1),
        )
        self.ab_gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Sigmoid()
        )
        self.channel_conv = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 1, bias=False),
            nn.LeakyReLU(0.2)
        )
        self.a_out = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(n_feats, n_feats, 3, 1, 1)
        )
        self.b_out = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(n_feats, n_feats, 3, 1, 1)
        )

    def forward(self, a_ab, b_ab, d):
        d = F.interpolate(d, size=[a_ab.size(2), a_ab.size(3)], mode='nearest')
        d = self.d_ini(d)
        ab = self.ab_ini(a_ab + b_ab)
        ab_gap = self.ab_gap(ab + ab * d)
        ab_c = self.channel_conv(ab_gap)
        a_ab = a_ab + ab * ab_c
        b_ab = b_ab + ab * ab_c
        a_ab = self.a_out(a_ab)
        b_ab = self.b_out(b_ab)
        return a_ab, b_ab

##############---  Strength Operate Subtractor Block (SOSB) ---################
class SOSB(nn.Module):
    def __init__(self, n_feats=16):
        super(SOSB, self).__init__()
        self.sos_unit = nn.Sequential(nn.Conv2d(n_feats, n_feats, 3, 1, 1),
                                       nn.LeakyReLU(0.2, True),
                                       nn.Conv2d(n_feats, n_feats, 3, 1, 1),
                                       nn.Tanh())
        self.sos_scale = nn.Parameter(torch.zeros(1))

    def forward(self, ld, le):
        l_de = ld + le
        sos_res = self.sos_unit(l_de) - l_de
        l_sos = l_de + self.sos_scale * sos_res
        return l_sos


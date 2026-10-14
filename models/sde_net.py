import torch.nn as nn
import torch.nn.functional as F
import models.blocks as bs

##### SAM-guided Depth Estimation Network
class SDENet(nn.Module):
    def __init__(self, n_feats=16):
        super(SDENet, self).__init__()
        # Encoder_ori
        self.o_ini = nn.Sequential(nn.Conv2d(3, n_feats, 3, 1, 1, bias=True),
                                       nn.LeakyReLU(0.2, True),
                                       nn.Conv2d(n_feats, n_feats, 3, 1, 1, bias=True))

        self.o_down1 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(n_feats, n_feats * 2, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, True)
        )
        self.o_down2 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(n_feats * 2, n_feats * 4, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, True)
        )
        # Mask Encoder
        self.m_ini = nn.Sequential(nn.Conv2d(1, n_feats, 3, 1, 1, bias=True),
                                      nn.LeakyReLU(0.2, True),
                                      nn.Conv2d(n_feats, n_feats, 3, 1, 1, bias=True))
        self.m_down1 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(n_feats, n_feats * 2, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, True))
        self.m_down2 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(n_feats * 2, n_feats * 4, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, True)
        )

        self.ddfb1 = bs.DDFB(n_feats, 4)
        self.ddfb2 = bs.DDFB(n_feats * 2, 8)
        self.dtfb = bs.DTFB(n_feats * 4, 16)

        # Decoder
        self.up2 = nn.Sequential(
            nn.Conv2d(n_feats * 4, n_feats * 2, 3, 1, 1),
            nn.LeakyReLU(0.2, True)
        )

        self.up1 = nn.Sequential(
            nn.Conv2d(n_feats * 2, n_feats, 3, 1, 1),
            nn.LeakyReLU(0.2, True)
        )

        self.conv_last = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(n_feats, 1, 3, 1, 1, bias=True)
        )

    def forward(self, img_ori, mask, sam_f):
        oe1 = self.o_ini(img_ori)
        me1 = self.m_ini(mask)
        skip1 = self.ddfb1(oe1, me1)

        oe2 = self.o_down1(oe1)
        me2 = self.m_down1(me1)
        skip2 = self.ddfb2(oe2, me2)

        oe3 = self.o_down2(oe2)
        me3 = self.m_down2(me2)
        skip3 = self.dtfb(oe3, me3, sam_f)

        ####################################
        d2 = F.interpolate(skip3, scale_factor=2, mode='bilinear', align_corners=False)
        d2 = self.up2(d2) + skip2
        d1 = F.interpolate(d2, scale_factor=2, mode='bilinear', align_corners=False)
        d1 = self.up1(d1) + skip1

        img_depth = self.conv_last(d1)

        return img_depth
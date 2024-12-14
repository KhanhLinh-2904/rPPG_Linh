import torch
import torch.nn as nn
from nets.blocks.blocks import  ECA_Block, TSM_CSTM,  TSM_Block_Adv  # CSTM with TAM




#-- TS_CST Motion Stream --#
class MotionModel_TS_CSTM(nn.Module):
    def __init__(self, eca, in_planes, kernel_size, frame_depth, shift_factor, skip, group_on):
        super().__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=in_planes, kernel_size=kernel_size, padding='same'),
            nn.LayerNorm([in_planes, 36, 36]),
            nn.Tanh())

        #-- TMB1 --#
        self.block_1 = TSM_CSTM(in_planes, frame_depth, shift_factor, skip, group_on)    

        self.pooling1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=(2, 2)),
            nn.Dropout2d(p=0.25)
        )

        #-- TSM --#
        self.tsm = TSM_Block_Adv(in_planes, in_planes * 2, frame_depth, False, shift_factor, group_on, on=True)

        #-- TMB2 --#
        self.block_2 = TSM_CSTM(in_planes * 2, frame_depth, shift_factor, skip, group_on)

        if eca:
            self.eca = True
            self.meca = ECA_Block(in_planes * 2)
        else:
            self.eca = False

        self.pooling2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=(2, 2)),
            nn.Dropout2d(p=0.25)
        )

    def forward(self, inputs, mask1, mask2):
        B, T, C, H, W = inputs.shape
        inputs = inputs.view(B*T, C, H, W)

        F0 = self.conv_1(inputs)
        _, C, H, W = F0.shape
        F1 = self.block_1(F0.view(B, T, C, H, W))
        # F2 = self.tam_1(F1.view(B, T, C, H, W))

        _, C, H, W = F1.shape
        mask1 = torch.reshape(mask1, (B, 1, C, H, W))
        mask1 = torch.tile(mask1, (1, T, 1, 1, 1))
        mask1 = torch.reshape(mask1, (B*T, C, H, W))

        F3 = torch.mul(mask1, F1)
        F4 = self.pooling1(F3)

        _, C, H, W = F4.shape
        F5 = self.tsm(F4.view(B, T, C, H, W))
        F6 = self.block_2(F5)
        # _, C, H, W = F6.shape
        # F7 = self.tam_2(F6.view(B, T, C, H, W))

        _, C, H, W = F6.shape
        mask2 = torch.reshape(mask2, (B, 1, C, H, W))
        mask2 = torch.tile(mask2, (1, T, 1, 1, 1))
        mask2 = torch.reshape(mask2, (B*T, C, H, W))

        F8 = torch.mul(mask2, F6)

        if self.eca:
            F9 = self.meca(F8)
        else:
            F9 = F8

        out = self.pooling2(F9)
        _, C, H, W = out.shape
        return out.view(B, T, C, H, W)



import torch
import torch.nn as nn

from nets.blocks.attentionBlocks import AttentionBlock
from nets.blocks.blocks import ECA_Block


class AppearanceModel_2D(nn.Module):
    def __init__(self, eca, in_planes, kernel_size=(3, 3)):
        # Appearance model
        super().__init__()
        self.conv1 = nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=in_planes, kernel_size=kernel_size, padding='same'),
            nn.BatchNorm2d(in_planes),
            nn.ReLU(True))

        self.conv2 = nn.Sequential(
            torch.nn.Conv2d(in_channels=in_planes, out_channels=in_planes, kernel_size=kernel_size, padding='same'),
            nn.BatchNorm2d(in_planes),
            nn.ReLU(True))

        if eca:
            self.eca1 = ECA_Block(in_planes)
            self.eca2 = ECA_Block(in_planes * 2)
            self.eca = True
        else:
            self.eca = False

        # Attention mask1
        self.attention_mask1 = AttentionBlock(in_planes)

        self.pooling = nn.Sequential(
            torch.nn.AvgPool2d(kernel_size=(2, 2)), torch.nn.Dropout2d(p=0.25))

        self.conv3 = nn.Sequential(
            torch.nn.Conv2d(in_channels=in_planes, out_channels=in_planes * 2, kernel_size=kernel_size,
                            padding='same'),
            nn.BatchNorm2d(in_planes * 2),
            nn.ReLU(True))
        self.conv4 = nn.Sequential(
            torch.nn.Conv2d(in_channels=in_planes * 2, out_channels=in_planes * 2, kernel_size=kernel_size,
                            padding='same'),
            nn.BatchNorm2d(in_planes * 2),
            nn.ReLU(True))

        # Attention mask2
        self.attention_mask2 = AttentionBlock(in_planes * 2)

    def forward(self, inputs):
        # inputs has shape B, C, H, W
        if self.eca:
            A1 = self.eca1(self.conv1(inputs))
        else:
            A1 = self.conv1(inputs)
        A2 = self.conv2(A1)
        # Calculate Mask1
        M1 = self.attention_mask1(A2)
        # Pooling and Dropout
        A3 = self.conv3(self.pooling(A2))

        if self.eca:
            M2 = self.attention_mask2(self.conv4(self.eca1(A3)))
        else:
            M2 = self.attention_mask2(self.conv4(A3))

        return M1, M2






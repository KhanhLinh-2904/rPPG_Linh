import torch
import torch.nn as nn
import torch.nn.functional as F





class LinearModel_TS_CSTM(torch.nn.Module):
    def __init__(self, eca, frame_depth):
        super().__init__()
        self.linear_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(5184*frame_depth, 128*frame_depth),
            nn.Tanh(),
            nn.Dropout(p=0.5),
            nn.Linear(128*frame_depth, frame_depth)
        )

    def forward(self, x):
        # B, T, C, H, W = x.shape
        # x = x.view(B*T, C, H, W)
        out = self.linear_layer.forward(x)
        return out



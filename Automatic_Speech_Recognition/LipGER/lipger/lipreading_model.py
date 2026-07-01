#!/usr/bin/env python
# lipreading_model_fixed.py

import torch
import torch.nn as nn

from lipger.lipreading_models.resnet import ResNet, BasicBlock
from lipger.lipreading_models.shufflenetv2 import ShuffleNetV2
from lipger.lipreading_models.tcn import TemporalConvNet

# --------------------------------------------------
# Helpers
# --------------------------------------------------
def threeD_to_2D_tensor(x):
    B, C, T, H, W = x.size()
    x = x.transpose(1, 2)  # (B, T, C, H, W)
    return x.reshape(B * T, C, H, W)


def temporal_average(x, lengths):
    out = []
    for i, l in enumerate(lengths):
        out.append(torch.mean(x[i, :l], dim=0))
    return torch.stack(out, dim=0)


# --------------------------------------------------
# TCN
# --------------------------------------------------
class TCN(nn.Module):
    def __init__(self, num_inputs, num_channels, num_classes,
                 tcn_options, dropout=0.5, relu_type='prelu'):
        super().__init__()

        self.tcn_trunk = TemporalConvNet(
            num_inputs=num_inputs,
            num_channels=num_channels,
            dropout=dropout,
            tcn_options=tcn_options,
            relu_type=relu_type
        )

        self.classifier = nn.Linear(num_channels[-1], num_classes)

    def forward(self, x, lengths, extract_feats=False):
        x = x.transpose(1, 2)  # (B, C, T)
        x = self.tcn_trunk(x)
        x = x.transpose(1, 2)  # (B, T, C)

        if extract_feats:
            return x

        x = temporal_average(x, lengths)
        return self.classifier(x)


# --------------------------------------------------
# Lipreading Model (FIXED FOR CHECKPOINT)
# --------------------------------------------------
class Lipreading(nn.Module):
    def __init__(self,
                 hidden_dim=256,
                 backbone_type='resnet',
                 num_classes=500,
                 relu_type='prelu',
                 tcn_options=None,
                 extract_feats=False):

        super().__init__()
        self.extract_feats = extract_feats
        self.backbone_type = backbone_type

        # --------------------------------------------------
        # MATCHED TCN CONFIG (IMPORTANT)
        # --------------------------------------------------
        if tcn_options is None:
            tcn_options = {
                "kernel_size": [3],
                "num_layers": 4,
                "width_mult": 2,   # MUST BE 2 (checkpoint expects 512)
                "dwpw": False,
                "dropout": 0.5
            }

        # --------------------------------------------------
        # BACKBONE (ResNet ONLY like checkpoint)
        # --------------------------------------------------
        self.trunk = ResNet(BasicBlock, [2, 2, 2, 2], relu_type=relu_type)

        self.frontend_nout = 24   # 🔥 FIXED (was 64)
        self.backend_out = 512

        # --------------------------------------------------
        # 3D FRONTEND (MATCH CHECKPOINT)
        # --------------------------------------------------
        frontend_relu = nn.PReLU(self.frontend_nout) if relu_type == 'prelu' else nn.ReLU()

        self.frontend3D = nn.Sequential(
            nn.Conv3d(1, self.frontend_nout,
                      kernel_size=(5, 7, 7),
                      stride=(1, 2, 2),
                      padding=(2, 3, 3),
                      bias=False),
            nn.BatchNorm3d(self.frontend_nout),
            frontend_relu,
            nn.MaxPool3d(kernel_size=(1, 3, 3),
                         stride=(1, 2, 2),
                         padding=(0, 1, 1))
        )

        # --------------------------------------------------
        # TCN (512 hidden match checkpoint)
        # --------------------------------------------------
        tcn_hidden = hidden_dim * tcn_options['width_mult']  # 256 * 2 = 512
        num_channels = [tcn_hidden] * tcn_options['num_layers']

        self.tcn = TCN(
            num_inputs=self.backend_out,
            num_channels=num_channels,
            num_classes=num_classes,
            tcn_options=tcn_options,
            dropout=tcn_options['dropout'],
            relu_type=relu_type
        )

    # --------------------------------------------------
    # Forward
    # --------------------------------------------------
    def forward(self, x, lengths):
        B, C, T, H, W = x.size()

        if isinstance(lengths, int):
            lengths = [lengths] * B

        # 3D CNN
        x = self.frontend3D(x)
        Tnew = x.shape[2]

        # reshape for 2D CNN
        x = threeD_to_2D_tensor(x)
        x = self.trunk(x)

        x = x.view(B, Tnew, -1)

        # TCN
        return self.tcn(x, lengths, extract_feats=self.extract_feats)
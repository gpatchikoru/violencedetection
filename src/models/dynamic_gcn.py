import torch.nn as nn
class DynamicGCN(nn.Module):
    def __init__(self, in_ch=2, hid_ch=64, n_joints=17):
        super().__init__()

    def forward(self, x):
        return x.view(x.size(0), x.size(1), -1)
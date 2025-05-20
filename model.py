import torch.nn.functional as F
import torch.nn as nn
import torch

class PreprocessLSBLayer(nn.Module):
    def forward(self, x):
        x_uint8 = (x * 255).to(torch.uint8)
        lsb_r = x_uint8[:, 0, :, :] & 1
        lsb_g = x_uint8[:, 1, :, :] & 1
        lsb_b = x_uint8[:, 2, :, :] & 1
        lsb_sum = lsb_r + lsb_g + lsb_b
        return lsb_sum.unsqueeze(1).float() / 3.0

class StegoNet(nn.Module):
    def __init__(self):
        super(StegoNet, self).__init__()
        self.preprocess = PreprocessLSBLayer()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4,4))
        self.fc1 = nn.Linear(16 * 4 * 4, 64)
        self.fc2 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.preprocess(x)             # shape (B,1,32,32)
        x = self.pool(F.relu(self.conv1(x)))  # (B,8,16,16)
        x = self.pool(F.relu(self.conv2(x)))  # (B,16,8,8)
        x = self.adaptive_pool(x)
        x = x.flatten(start_dim=1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.sigmoid(self.fc2(x))      # (B,1)
        return x
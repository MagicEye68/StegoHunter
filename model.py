import torch.nn.functional as F
import torch.nn as nn
import torch

class PreprocessLSBLayer(nn.Module):
    def __init__(self, top_fraction=1.0):
        super().__init__()
        self.top_fraction = top_fraction

    def forward(self, x):
        x_uint8 = (x * 255).to(torch.uint8)
        #lsb_r = x_uint8[:, 0, :, :] & 1
        #lsb_g = x_uint8[:, 1, :, :] & 1
        #lsb_b = x_uint8[:, 2, :, :] & 1
        #lsb = lsb_r + lsb_g + lsb_b
        #return lsb.unsqueeze(1).float() / 3.0
        h = x.shape[2]
        top_h = int(h * self.top_fraction)
        x_top = x_uint8[:, :, :top_h, :]
        lsb_r = (x_top[:, 0, :, :] & 1).unsqueeze(1)  # (B,1,H,W)
        lsb_g = (x_top[:, 1, :, :] & 1).unsqueeze(1)  # (B,1,H,W)
        lsb_b = (x_top[:, 2, :, :] & 1).unsqueeze(1)  # (B,1,H,W)
        lsb = torch.cat([lsb_r, lsb_g, lsb_b], dim=1)   # (B,3,H,W)
        return lsb.float()


class StegoNet(nn.Module):
    def __init__(self, num_classes=9):
        super(StegoNet, self).__init__()
        self.preprocess = PreprocessLSBLayer()
        
        self.conv1 = nn.Conv2d(3, 36, kernel_size=(1,5), padding=(0,2))
        self.bn1 = nn.BatchNorm2d(36)

        self.conv1_v = nn.Conv2d(3, 12, kernel_size=(5,1), padding=(2,0))
        self.bn1_v = nn.BatchNorm2d(12)

        self.conv2 = nn.Conv2d(48, 48, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(48)
        
        self.conv3 = nn.Conv2d(48, 96, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(96)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.fc1 = nn.Linear(96, 128)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.preprocess(x)
        x_h = self.pool(F.relu(self.bn1(self.conv1(x))))
        x_v = self.pool(F.relu(self.bn1_v(self.conv1_v(x))))
        x = torch.cat([x_h, x_v], dim=1)
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.adaptive_pool(x)
        x = x.flatten(start_dim=1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x
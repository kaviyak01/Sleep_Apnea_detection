# model.py
import torch
import torch.nn as nn

class ResNetBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=7, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride=1, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        identity = self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        out = self.relu(out)
        return out

class SleepApneaNet(nn.Module):
    def __init__(self, input_channels=1, num_classes_bin=2, num_classes_sev=3):
        super().__init__()
        self.res1 = ResNetBlock1D(input_channels, 16)
        self.res2 = ResNetBlock1D(16, 32)
        self.res3 = ResNetBlock1D(32, 64)
        self.gru = nn.GRU(input_size=64, hidden_size=32, batch_first=True, bidirectional=True)
        self.fc_bin = nn.Linear(64, num_classes_bin)
        self.fc_sev = nn.Linear(64, num_classes_sev)

    def forward(self, x):
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = x.permute(0, 2, 1)  # [batch, time, features]
        out_gru, _ = self.gru(x)
        out_gru = out_gru[:, -1, :]
        bin_out = self.fc_bin(out_gru)
        sev_out = self.fc_sev(out_gru)
        return bin_out, sev_out

import torch
import torch.nn as nn


class ECGNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=7, stride=1, padding=3)
        self.relu  = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.pool  = nn.AdaptiveAvgPool1d(output_size=1)
        self.fc    = nn.Linear(in_features=64, out_features=5)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.squeeze(-1)
        x = self.fc(x)
        return x
    
class ECGNetRR(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=7, stride=1, padding=3)
        self.relu  = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.pool  = nn.AdaptiveAvgPool1d(output_size=1)
        self.fc_rr = nn.Linear(in_features=3, out_features=16)  # RR features branch
        self.dropout = nn.Dropout(0.5)
        self.fc_combined = nn.Linear(in_features=64 + 16, out_features=5)

    def forward(self, x, rr):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.squeeze(-1)

        rr_out = self.fc_rr(rr)
        rr_out = self.relu(rr_out)

        combined = torch.cat((x, rr_out), dim=1)
        combined = self.dropout(combined)
        output = self.fc_combined(combined)
        return output
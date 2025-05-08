import argparse, os
import decord
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, Dataset
class FlowDataset(Dataset):
    def __init__(self, raw_dir, transform=None):
        self.files = [os.path.join(raw_dir,f) for f in os.listdir(raw_dir) if f.endswith('.mp4')]
    def __len__(self): return len(self.files)
    def __getitem__(self, i):
        raise NotImplementedError

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2,16,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16,32,3,padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.fc = nn.Linear(32*56*56, 2)
    def forward(self, x):
        return self.fc(self.conv(x).view(x.size(0), -1))
# dataset.py
import numpy as np
import torch
from torch.utils.data import Dataset

class SleepApneaDataset(Dataset):
    def __init__(self, signals, labels_bin, labels_sev, segment_len=6000):
        self.signals = signals
        self.labels_bin = labels_bin
        self.labels_sev = labels_sev
        self.segment_len = segment_len

    def __len__(self):
        return len(self.labels_bin)

    def __getitem__(self, idx):
        start = idx * self.segment_len
        end = (idx + 1) * self.segment_len
        x = self.signals[start:end]
        if len(x) < self.segment_len:
            x = np.pad(x, (0, self.segment_len - len(x)), 'constant')
        x = (x - np.mean(x)) / np.std(x)
        x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
        y_bin = torch.tensor(self.labels_bin[idx], dtype=torch.long)
        y_sev = torch.tensor(self.labels_sev[idx], dtype=torch.long)
        return x, (y_bin, y_sev)

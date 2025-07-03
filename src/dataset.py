import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split

# Map your folder names to integer labels:
LABELS = {"Inside": 0, "Right": 1, "Flat": 2}

class JumpSequenceDataset(Dataset):
    def __init__(self, processed_dir):
        self.samples = []
        for label, idx in LABELS.items():
            d = os.path.join(processed_dir, label)
            for fn in os.listdir(d):
                if fn.endswith(".npy"):
                    self.samples.append((os.path.join(d,fn), idx))
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, i):
        path, label = self.samples[i]
        seq = np.load(path)                     # (60,66)
        return torch.from_numpy(seq).float(), label

def get_loaders(processed_dir, batch_size=16, val_frac=0.1, test_frac=0.1, seed=42):
    full = JumpSequenceDataset(processed_dir)
    n = len(full)
    n_test  = int(n * test_frac)
    n_val   = int(n * val_frac)
    n_train = n - n_val - n_test
    train_ds, val_ds, test_ds = random_split(
        full,
        [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(seed)
    )
    return (
      DataLoader(train_ds, batch_size=batch_size, shuffle=True),
      DataLoader(val_ds,   batch_size=batch_size, shuffle=False),
      DataLoader(test_ds,  batch_size=batch_size, shuffle=False),
    )

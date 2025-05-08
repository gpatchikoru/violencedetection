import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from utils.enhanced_features import compute_enhanced_features, augment_pose_sequence

class EnhancedViolenceDataset(Dataset):  
    def __init__(self, h5_path, seq_len=16, augment=False):
        self.h5_path = h5_path
        self.seq_len = seq_len
        self.augment = augment
        self._h5 = None
    
    def _ensure_open(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, 'r')
            self._keys = list(self._h5.keys())
    
    def __len__(self):
        self._ensure_open()
        return len(self._keys)
    
    def __getitem__(self, idx):
        self._ensure_open()
        grp = self._h5[self._keys[idx]]
        poses = grp['pose'][:]  
        label = grp.attrs['label']

        N = poses.shape[0]
        idxs = np.linspace(0, N - 1, self.seq_len, dtype=int)
        seq = poses[idxs]  
        if self.augment:
            if label == 0 and np.random.random() < 0.7: 
                seq = augment_pose_sequence(seq, augment_prob=0.6)
            elif label == 1 and np.random.random() < 0.5: 
                seq = augment_pose_sequence(seq, augment_prob=0.4)
        features = compute_enhanced_features(seq)
        
        return torch.from_numpy(features).float(), torch.tensor(label)
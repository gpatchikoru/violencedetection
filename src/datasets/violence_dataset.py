import os
import cv2
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from utils.video_utils import extract_pose

class ViolenceDataset(Dataset):
    def __init__(self, h5_path: str, seq_len: int = 16):
        self.h5_path = h5_path
        self.seq_len = seq_len
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
        seq  = poses[idxs]                        
        return torch.from_numpy(seq).float(), torch.tensor(label)



class OnTheFlyViolenceDataset(Dataset):
    def __init__(self, raw_dir: str, seq_len: int = 16, skip: int = 1):
        self.items = []         
        for root, _, files in os.walk(raw_dir):
            for f in files:
                if not f.lower().endswith('.mp4'):
                    continue
                label = 1 if os.path.basename(root).lower() == 'violence' else 0
                self.items.append((os.path.join(root, f), label))
        self.seq_len = seq_len
        self.skip    = skip

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        cap = cv2.VideoCapture(path)
        poses = []
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % self.skip == 0:
                kp = extract_pose(frame)          # (17,2)
                poses.append(kp)
            frame_idx += 1
        cap.release()

        if len(poses) == 0:
            poses = [np.zeros((17, 2), dtype=np.float32)]

        poses = np.stack(poses, axis=0)            # (N,17,2)
        # pad or subsample to seq_len
        N = poses.shape[0]
        if N >= self.seq_len:
            idxs = np.linspace(0, N - 1, self.seq_len, dtype=int)
            seq = poses[idxs]
        else:
            pad = np.zeros((self.seq_len - N, 17, 2), dtype=np.float32)
            seq = np.concatenate([poses, pad], axis=0)

        return torch.from_numpy(seq).float(), torch.tensor(label)

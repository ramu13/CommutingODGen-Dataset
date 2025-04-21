import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

class CommutingODDataset(Dataset):
    """
    1つの area (= フォルダ) を 1サンプルとみなし、
    x: (N, N, F)   y: (N, N)  を返す Dataset
    """
    def __init__(self, root, areas, shuffle_areas = True):
        self.root = root
        self.areas = areas.copy()
        if shuffle_areas:
            random.shuffle(self.areas)

    def __len__(self):
        return len(self.areas)

    def _load_area_arrays(self, area):
        prefix = os.path.join(self.root, area)
        demos = np.load(f"{prefix}/demos.npy")     # shape (N, D_d)
        pois  = np.load(f"{prefix}/pois.npy")      # shape (N, D_p)
        dis   = np.load(f"{prefix}/dis.npy")       # shape (N, N)
        od    = np.load(f"{prefix}/od.npy")        # shape (N, N)
        return demos, pois, dis, od

    def _make_feature_tensor(self, demos, pois, dis):
        feat = np.concatenate([demos, pois], axis=1)          # (N, F_d+F_p)
        N, F = feat.shape

        # ブロードキャスト展開（メモリ効率版）
        feat_o = feat[:, None, :]                              # (N, 1, F)
        feat_d = feat[None, :, :]                              # (1, N, F)
        dis    = dis[..., None]                                # (N, N, 1)

        x = np.concatenate([np.repeat(feat_o, N, axis=1),     # (N, N, F)
                            np.repeat(feat_d, N, axis=0),     # (N, N, F)
                            dis], axis=2)                      # (N, N, 2F+1)
        return torch.from_numpy(x).float()                    # (N, N, C)

    def __getitem__(self, idx):
        area = self.areas[idx]
        demos, pois, dis, od = self._load_area_arrays(area)
        x = self._make_feature_tensor(demos, pois, dis)        # (N, N, C)
        y = torch.from_numpy(od).float()                       # (N, N)
        return {"x": x, "y": y, "area": area}

def pad_collate(batch):
    """
    各 area の N が異なるため、最大 N にゼロパディングし
    (B, N_max, N_max, C) / (B, N_max, N_max) へ揃える。
    """
    xs  = [item["x"] for item in batch]
    ys  = [item["y"] for item in batch]
    areas = [item["area"] for item in batch]

    # 4-D Tensor へパディング
    C = xs[0].shape[-1]
    N_max = max(x.shape[0] for x in xs)
    x_pad = torch.zeros(len(xs), N_max, N_max, C)
    y_pad = torch.zeros(len(ys), N_max, N_max)

    for i, (x, y) in enumerate(zip(xs, ys)):
        n = x.shape[0]
        x_pad[i, :n, :n, :] = x
        y_pad[i, :n, :n]    = y

    mask = (y_pad.sum(-1) != 0).float()   # (B, N_max) ― 損失計算用マスクなどに利用
    return {"x": x_pad, "y": y_pad, "mask": mask, "areas": areas}
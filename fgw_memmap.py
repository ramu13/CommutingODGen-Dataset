import numpy as np
import torch
from torch.utils.data import DataLoader
import ot
from tqdm import tqdm
from models.yours.dataset import CommutingODDataset
import os

# ---------- 設定 ----------
ROOT_DIR = "data"
ALPHA    = 0.5
DIST_BIN = "fgw_dist.dat"            # 距離行列
IDS_BIN  = "fgw_area_ids.npy"        # 行列順の area_id を保存
DTYPE    = np.float32

# ---------- 1. area_id を確定（shuffle 無し） ----------
area_ids = sorted(os.listdir("data")) 
np.save(IDS_BIN, np.array(area_ids))          # ★必ず保存

N = len(area_ids)
loader = DataLoader(
    CommutingODDataset(ROOT_DIR, area_ids),
    batch_size=1, shuffle=False, num_workers=0
)

# ---------- 2. 距離行列ファイルを用意 ----------
D = np.memmap(DIST_BIN, mode="w+", dtype=DTYPE, shape=(N, N))
D[:] = 0.0                                     # 対角 0

# ---------- 3. FGW 距離関数 ----------
def split_feats_adj(x):
    if x.ndim == 3:
        feats, adj = x[0], x[1]                # (C,N,F) → (F) と隣接
    else:
        feats, adj = x, None
    return feats.numpy(), (adj.numpy() if adj is not None else None)

def fgw_distance(x1, x2, alpha=ALPHA):
    f1, adj1 = split_feats_adj(x1)
    f2, adj2 = split_feats_adj(x2)

    M  = np.linalg.norm(f1[:, None] - f2[None, :], axis=-1)
    C1 = 1.0 - adj1 if adj1 is not None and adj1.shape[0]==adj1.shape[1] else np.ones((f1.shape[0],)*2)
    C2 = 1.0 - adj2 if adj2 is not None and adj2.shape[0]==adj2.shape[1] else np.ones((f2.shape[0],)*2)

    p  = np.full(f1.shape[0], 1/f1.shape[0])
    q  = np.full(f2.shape[0], 1/f2.shape[0])

    return ot.gromov.fused_gromov_wasserstein2(
        M, C1, C2, p=p, q=q,
        loss_fun="square_loss", alpha=alpha, symmetric=True
    )

# ---------- 4. 距離行列を逐次計算 ----------
loader_list = list(loader)                     # インデックスアクセス
for i in tqdm(range(N), desc="FGW rows"):
    g_i = loader_list[i]["x"].squeeze(0).cpu()
    for j in range(i+1, N):
        g_j = loader_list[j]["x"].squeeze(0).cpu()
        D[i, j] = D[j, i] = fgw_distance(g_i, g_j)

D.flush()
print(f"distance matrix  : {DIST_BIN}")
print(f"area id sequence : {IDS_BIN}")
print("FGW distance matrix saved.")

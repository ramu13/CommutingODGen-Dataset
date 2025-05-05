"""
FGW で 10 グラフを相互比較して近いグラフを出力するスクリプト
--------------------------------------------------------------
- データ取得         : CommutingODDataset から最初の 10 件
- コスト行列作成     : ノード特徴 (L2) / 構造コスト (1‑adjacency)
- FGW 距離計算       : ot.gromov.fused_gromov_wasserstein2
"""

import numpy as np
import torch
from torch.utils.data import DataLoader
import ot
from models.yours.dataset import CommutingODDataset
from models.yours.utils.load_utils import load_all_areas, split_train_valid_test

# ---------------------------------------------------------------------
# 0. 設定
K_GRAPHS   = 10             # 比較対象のグラフ数
ALPHA      = 0.5            # 構造 : 特徴 の重み（0〜1）
ROOT_DIR   = "data"

# ---------------------------------------------------------------------
# 1. データ読み込み（最初の K_GRAPHS だけ）
areas_all = load_all_areas()
train_areas, _, _ = split_train_valid_test(areas_all)

loader = DataLoader(
    CommutingODDataset(ROOT_DIR, train_areas),
    batch_size=1, shuffle=True, num_workers=0
)

graphs, area_ids = [], []
for batch in loader:
    graphs.append(batch["x"].squeeze(0).cpu())   # (C,N,F) or (N,F)
    area_ids.append(batch["area"][0])
    if len(graphs) >= K_GRAPHS:                  # 10 件そろったら抜ける
        break

print("Loaded areas:", area_ids)

# ---------------------------------------------------------------------
# 2. FGW 用ユーティリティ
def split_feats_adj(x: torch.Tensor):
    """Dataset テンソル → (features, adjacency or None)"""
    if x.ndim == 3:               # (C,N,F) を想定
        feats, adj = x[0], x[1]
    else:                         # (N,F)
        feats, adj = x, None
    return feats.numpy(), (adj.numpy() if adj is not None else None)

def feature_cost(f1, f2):
    diff = f1[:, None, :] - f2[None, :, :]
    return np.linalg.norm(diff, axis=-1)         # (n1,n2)

def structure_cost(adj, N):
    if adj is not None and adj.ndim == 2 and adj.shape[0] == adj.shape[1]:
        return 1.0 - adj                         # (N,N)
    return np.ones((N, N))                       # 構造情報なし

def fgw_distance(x1, x2, alpha=ALPHA):
    f1, adj1 = split_feats_adj(x1)
    f2, adj2 = split_feats_adj(x2)

    M  = feature_cost(f1, f2)
    C1 = structure_cost(adj1, f1.shape[0])
    C2 = structure_cost(adj2, f2.shape[0])
    p  = np.full(f1.shape[0], 1.0 / f1.shape[0])
    q  = np.full(f2.shape[0], 1.0 / f2.shape[0])

    return ot.gromov.fused_gromov_wasserstein2(
        M, C1, C2, p=p, q=q,
        loss_fun="square_loss",
        alpha=alpha,
        symmetric=True
    )

# ---------------------------------------------------------------------
# 3. 距離行列の作成
n = len(graphs)
D = np.zeros((n, n))
for i in range(n):
    for j in range(i + 1, n):
        D[i, j] = D[j, i] = fgw_distance(graphs[i], graphs[j])

# ---------------------------------------------------------------------
# 4. 近傍グラフを表示
for i, aid in enumerate(area_ids):
    nearest = np.argsort(D[i])[1:4]              # 自分を除いた上位 3 件
    print(f"{aid} → {[area_ids[j] for j in nearest]}  dist={D[i, nearest]}")

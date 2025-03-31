import time

from models.GM_E.data_load import load_data
from models.GM_E.metrics import *
from models.GM_E.model import *

import torch
import torch.nn.functional as F
from pprint import pprint

area = '01001'

demos = np.load(f"data/{area}/demos.npy")[:, 0][:, np.newaxis]
pois = np.load(f"data/{area}/pois.npy")
feat = np.concatenate([demos, pois], axis=1)

feat_o, feat_d = feat, feat
feat_o = feat_o.reshape([feat_o.shape[0], 1, feat_o.shape[1]]).repeat(feat_o.shape[0], axis=1)
feat_d = feat_d.reshape([1, feat_d.shape[0], feat_d.shape[1]]).repeat(feat_d.shape[0], axis=0)

dis = np.load(f"data/{area}/dis.npy")
dis = dis.reshape([dis.shape[0], dis.shape[1], 1])

x = np.concatenate([feat_o, feat_d, dis], axis=2) # shape: (n, n, d)
y = np.load(f"data/{area}/od.npy") # shape: (n, n)



from models.OT_R.model import RIOT
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# feat_o, feat_d は元の特徴量 feat（shape: (n, 1, feat_dim) と (1, n, feat_dim) から作成済み）
# ここで feat_dim を取得
feat_dim = feat_o.shape[2]  # 出発地特徴の次元

# x の形状は (n, n, 2*feat_dim+1)
x = np.concatenate([feat_o, feat_d, dis], axis=2)  # 形状: (n, n, 2*feat_dim+1)
y = np.load(f"data/{area}/od.npy")  # 形状: (n, n)

# モデル初期化時は pdim = feat_dim, qdim = feat_dim
riot = RIOT(pdim=feat_dim, qdim=feat_dim, n_iter=100).to(device)

x_tensor = torch.tensor(x).float().to(device)  # shape: (n, n, 2*feat_dim+1)
y_tensor = torch.tensor(y).float().to(device)  # shape: (n, n)

# 周辺分布（mu, nu）: 各行・各列の和 → (n, 1)
mu = y_tensor.sum(dim=1, keepdim=True)
nu = y_tensor.sum(dim=0, keepdim=True).T

# 正しい U, V の抽出:
U = x_tensor[:, 0, :feat_dim]          # shape: (n, feat_dim) → 出発地特徴
V = x_tensor[0, :, feat_dim:2*feat_dim]  # shape: (n, feat_dim) → 到着地特徴
W = x_tensor[:, :, 2*feat_dim:].squeeze()  # shape: (n, n) → 距離行列

# 推論、loss 計算、最適化
y_hat = riot(U, V, W, mu, nu)
loss = F.mse_loss(y_hat, y_tensor)

optimizer = torch.optim.Adam(riot.parameters(), lr=1e-4)
optimizer.zero_grad()
loss.backward()
optimizer.step()

print(f"Loss: {loss.item()}")
print(f"y_hat: {y_hat}")
print(f"y: {y_tensor}")
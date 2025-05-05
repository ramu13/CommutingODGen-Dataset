import os, pathlib, numpy as np, torch, ot
from tqdm import tqdm
from models.yours.dataset import CommutingODDataset

# ---------- 設定 ----------
DATA_DIR = "data"                    # area フォルダが置かれている場所
ALPHA    = 0.5                       # FGW の α
IDS_BIN  = "fgw_area_ids.npy"        # 行列順の area_id を保存
DIST_BIN = "fgw_dist.dat"            # 距離行列 (memmap)
DTYPE    = np.float32

# ---------- 1. area_id を固定順で列挙 ----------
area_ids = sorted(os.listdir(DATA_DIR))          # .DS_Store 等は事前に除外しておく
np.save(IDS_BIN, np.array(area_ids))              # ★ 順序をファイル保存

# ---------- 2. Dataset 準備（シャッフルなし） ----------
dataset = CommutingODDataset(DATA_DIR, area_ids)  # データは都度ロード
N = len(dataset)

# ---------- 3. 距離行列ファイルを確保 ----------
D = np.memmap(DIST_BIN, mode="w+", dtype=DTYPE, shape=(N, N))
D[:] = 0.0                                        # 対角 0

# ---------- 4. dis.npy へのパス関数 ----------
def dis_path(aid: str):
    return os.path.join(DATA_DIR, aid, "dis.npy")  # (N,N) の距離行列

# ---------- 5. Node attribute 抽出 (距離チャネルを除く) ----------
def node_features(x_tensor: torch.Tensor):
    """
    x : (N, N, 2F+1)
    -> ノードごとの特徴行列 (N, F_attr)
       ・最後の距離チャネルは捨てる
       ・出発ノード側の特徴だけを抽出
    """
    twoFplus1 = x_tensor.shape[-1]
    F = (twoFplus1 - 1) // 2  #元の 1 ノードあたりの次元

    # 出発ノード i の特徴は 行方向に一定なので j=0 列を取ればよい
    return x_tensor[:, 0, :F].numpy()           # (N, F)

# ---------- 6. FGW 距離関数 ----------
def fgw_dist(sample_i, sample_j, alpha=ALPHA):
    x1, aid1 = sample_i["x"], sample_i["area"]
    x2, aid2 = sample_j["x"], sample_j["area"]

    f1 = node_features(x1)
    f2 = node_features(x2)

    C1 = np.load(dis_path(aid1)).astype(np.float32)
    C2 = np.load(dis_path(aid2)).astype(np.float32)
    C1 /= C1.max() or 1.0
    C2 /= C2.max() or 1.0

    M  = np.linalg.norm(f1[:, None] - f2[None, :], axis=-1)

    p = np.full(f1.shape[0], 1 / f1.shape[0])
    q = np.full(f2.shape[0], 1 / f2.shape[0])

    return ot.gromov.fused_gromov_wasserstein2(
        M, C1, C2, p=p, q=q,
        loss_fun="square_loss", alpha=alpha, symmetric=True
    )

# ---------- 7. 距離行列を逐次計算（常に 2 サンプルしかメモリに載らない） ----------
for i in tqdm(range(N), desc="FGW rows"):
    samp_i = dataset[i]
    for j in range(i + 1, N):
        samp_j = dataset[j]
        D[i, j] = D[j, i] = fgw_dist(samp_i, samp_j)

D.flush()
print(f"distance matrix  : {DIST_BIN}")
print(f"area id sequence : {IDS_BIN}")
print("FGW distance matrix saved.")

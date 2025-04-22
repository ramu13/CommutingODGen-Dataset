# utils/train_utils.py

import torch
import torch.nn.functional as F
from tqdm import tqdm


def flat_batch(batch, device):
    """
    pad_collate が返す 4‑D/2‑D テンソルを「有効セルだけの 2‑D 行列」に畳む。

    戻り値
    -------
    x_flat   : (M, C)   入力特徴（C = 2F+1）
    tgt_flat : (M,)     教師確率 P(j|i)
    ori_flat : (M,)     origin id （エリア間で衝突しないようオフセット付与）
    """
    x4d, y2d, mask2d = batch["x"], batch["y"], batch["mask"]   # (B,N,N,C), (B,N,N), (B,N)
    B, N_max, _, C = x4d.shape

    # 有効セルを示す 3‑D マスク (B,N,N)
    valid_mask = (mask2d[:, :, None] * mask2d[:, None, :]).bool()

    x_flat, tgt_flat, ori_flat = [], [], []
    offset = 0  # origin id 衝突を避けるための累積オフセット

    for b in range(B):
        n = int(mask2d[b].sum().item())          # 真のノード数 n_b
        if n == 0:
            continue

        m = valid_mask[b, :n, :n].reshape(-1)    # (n²,) 有効セル 1/0

        if m.sum() == 0:                         # このエリアにフローが無ければスキップ
            offset += n
            continue

        # ---------- 入力特徴 ----------
        # x4d[b, :n, :n, :]  -> (n, n, C) -> (n², C) → m でフィルタ
        xf = x4d[b, :n, :n, :].reshape(-1, C)[m]   # (M_b, C)
        x_flat.append(xf)

        # ---------- 教師確率 P(j|i) ----------
        flows = y2d[b, :n, :n]                   # (n, n)
        probs = flows / (flows.sum(dim=1, keepdim=True) + 1e-8)
        yf = probs.reshape(-1)[m]                # (M_b,)
        tgt_flat.append(yf)

        # ---------- origin ids ----------
        # 行インデックスを作成 → m で抜粋 → offset を加算
        ori = torch.arange(n, device=device).repeat_interleave(n)
        ori = ori[m] + offset                    # (M_b,)
        ori_flat.append(ori)

        offset += n                              # 次エリアに備えオフセット更新

    # ----------- 連結して (M, …) ----------
    return (
        torch.cat(x_flat).to(device),            # (M, C)
        torch.cat(tgt_flat).to(device),          # (M,)
        torch.cat(ori_flat).to(device, dtype=torch.long)  # (M,)
    )


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


def pad_collate_for_reg(batch):
    """
    回帰用: 各 area の x: (N,N,C), y: (N,N) を flatten し、(M,C), (M,) にまとめて返す
    """
    x_list, y_list = [], []

    for item in batch:
        x = item["x"]  # shape: (N, N, C)
        y = item["y"]  # shape: (N, N)

        N = x.shape[0]
        x = x.reshape(-1, x.shape[-1])  # (N*N, C)
        y = y.reshape(-1)               # (N*N,)

        # 0フローは除く（optional）
        mask = y > 0
        x_list.append(x[mask])
        y_list.append(y[mask])

    return {
        "x": torch.cat(x_list, dim=0),   # (M, C)
        "y": torch.cat(y_list, dim=0)    # (M,)
    }


def epoch_pass(model, loader, optimizer=None, device='cuda'):
    model.train() if optimizer else model.eval()
    total_loss, total_count = 0.0, 0

    loop = tqdm(loader, desc="Train" if optimizer else "Valid", leave=False)

    with torch.set_grad_enabled(optimizer is not None):
        for batch in loop:
            x, tgt, ori = flat_batch(batch, device)
            pred = model(x, ori)  # pred: (N,), tgt: (N,)

            if torch.isnan(tgt).any() or (tgt < 0).any():
                print("ERROR tgt contains NaN or negative values!")

            eps = 1e-8
            loss = F.kl_div((pred + eps).log(), tgt, reduction='batchmean')

            if optimizer:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            batch_loss = loss.item()
            total_loss += batch_loss * tgt.size(0)
            total_count += tgt.size(0)

            loop.set_postfix(loss=batch_loss)

    return total_loss / total_count

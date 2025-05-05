import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from models.yours.dataset import CommutingODPairDataset
from models.yours.model import GRAVITY_P
from models.yours.utils.load_utils import load_all_areas, split_train_valid_test
from tqdm import tqdm

# ---------- 1. Data ----------
root_dir = "data"
train_areas, valid_areas, _ = split_train_valid_test(load_all_areas())

train_loader = DataLoader(
    CommutingODPairDataset(root_dir, train_areas),
    batch_size=2, shuffle=True, num_workers=0
)

valid_loader = DataLoader(
    CommutingODPairDataset(root_dir, valid_areas, shuffle_areas=False),
    batch_size=1, shuffle=False, num_workers=0
)

# ---------- 2. Train Loop ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# モデル構築
input_dim = train_loader.dataset[0]["x"].shape[-1]
model = GRAVITY_P(input_dim=input_dim).to(device)

# オプティマイザ
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# エポック数
num_epochs = 10

# 学習ループ
for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    model.train()
    total_loss = 0.0
    total_count = 0

    # tqdm で進捗表示
    for batch in tqdm(train_loader, desc="Train"):
        x = batch["x"].to(device)           # shape: (N, input_dim)
        y_true = batch["y"].to(device)      # shape: (N,)

        # フォワード
        y_pred = model(x)                   # shape: (N,)

        # 損失計算（回帰）
        loss = F.mse_loss(y_pred, y_true)
        print(f"  Train Loss: {loss.item():.6f}")

        # バックプロパゲーション
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ロス積算
        total_loss += loss.item() * y_true.size(0)
        total_count += y_true.size(0)

    avg_loss = total_loss / total_count
    print(f"  Avg Train Loss: {avg_loss:.6f}")

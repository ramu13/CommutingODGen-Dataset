# colab, jupyter で実行するときのみ使う
# import sys
# sys.path.append('/content/CommutingODGen-Dataset')

# import torch, torch.nn as nn, torch.nn.functional as F
# from torch.utils.data import DataLoader
# from models.yours.data_load import load_all_areas, split_train_valid_test
# from models.yours.dataset import CommutingODDataset
# from models.yours.model import DeepGravity
# from models.yours.utils.train_utils import epoch_pass, pad_collate_for_reg


# # ---------- 1. Data ----------
# root_dir = "data"
# train_areas, valid_areas, _ = split_train_valid_test(load_all_areas())

# train_loader = DataLoader(
#     CommutingODDataset(root_dir, train_areas),
#     batch_size=2, shuffle=True, collate_fn=pad_collate_for_reg, num_workers=0
# )
# valid_loader = DataLoader(
#     CommutingODDataset(root_dir, valid_areas, shuffle_areas=False),
#     batch_size=1, shuffle=False, collate_fn=pad_collate_for_reg, num_workers=0
# )


# ---------- 2. Train Loop ----------
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# input_dim = train_loader.dataset[0]["x"].shape[-1]   # 2F+1
# model = DeepGravity(input_dim).to(device)
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# num_epochs = 10
# for epoch in range(num_epochs):
#     print(f"Epoch {epoch+1}/{num_epochs}")
#     train_loss = epoch_pass(model, train_loader, optimizer, device)
#     print(f"  Train Loss: {train_loss:.6f}")
#     valid_loss = epoch_pass(model, valid_loader, None, device)
#     print(f"  Valid Loss: {valid_loss:.6f}")




# colab, jupyter で実行するときのみ使う
# import sys
# sys.path.append('/content/CommutingODGen-Dataset')

import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from models.yours.data_load import load_all_areas, split_train_valid_test
from models.yours.dataset import CommutingODDataset
from models.yours.model import DeepGravityReg
from models.yours.utils.train_utils import pad_collate_for_reg
from tqdm import tqdm

# ---------- 1. Data ----------
root_dir = "data"
train_areas, valid_areas, _ = split_train_valid_test(load_all_areas())

train_loader = DataLoader(
    CommutingODDataset(root_dir, train_areas),
    batch_size=2, shuffle=True, collate_fn=pad_collate_for_reg, num_workers=0
)
valid_loader = DataLoader(
    CommutingODDataset(root_dir, valid_areas, shuffle_areas=False),
    batch_size=1, shuffle=False, collate_fn=pad_collate_for_reg, num_workers=0
)

# ---------- 2. Train Loop ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# モデル構築
input_dim = train_loader.dataset[0]["x"].shape[-1]
model = DeepGravityReg(input_dim=input_dim).to(device)

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

        # バックプロパゲーション
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ロス積算
        total_loss += loss.item() * y_true.size(0)
        total_count += y_true.size(0)

    avg_loss = total_loss / total_count
    print(f"  Avg Train Loss: {avg_loss:.6f}")

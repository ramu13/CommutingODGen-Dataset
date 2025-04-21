# colab, jupyter で実行するときのみ使う
# import sys
# sys.path.append('/content/CommutingODGen-Dataset')

import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from models.yours.data_load import load_all_areas, split_train_valid_test
from models.yours.dataset import CommutingODDataset
from models.yours.model import DeepGravity
from utils.train_utils import epoch_pass, pad_collate



# ---------- 1. Data ----------
root_dir = "data"
train_areas, valid_areas, _ = split_train_valid_test(load_all_areas())

train_loader = DataLoader(
    CommutingODDataset(root_dir, train_areas),
    batch_size=2, shuffle=True, collate_fn=pad_collate, num_workers=0
)
valid_loader = DataLoader(
    CommutingODDataset(root_dir, valid_areas, shuffle_areas=False),
    batch_size=1, shuffle=False, collate_fn=pad_collate, num_workers=0
)


# ---------- 2. Train Loop ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_dim = train_loader.dataset[0]["x"].shape[-1]   # 2F+1
model = DeepGravity(input_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


num_epochs = 10
for epoch in range(num_epochs):
    train_loss = epoch_pass(model, train_loader, optimizer, device)
    valid_loss = epoch_pass(model, valid_loader, None, device)
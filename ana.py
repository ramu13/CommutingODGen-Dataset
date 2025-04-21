import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from models.yours.data_load import load_all_areas, split_train_valid_test
from models.yours.dataset import CommutingODDataset, pad_collate
from models.yours.model import DeepGravity, flat_batch


def epoch_pass(loader, train=True):
    model.train() if train else model.eval()
    total_kl, total_obs = 0.0, 0
    with torch.set_grad_enabled(train):
        for batch in loader:
            x, tgt, ori = flat_batch(batch, device)
            pred = model(x, ori)                         # (M,)
            loss = F.kl_div(pred.log(), tgt, reduction='batchmean')

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_kl  += loss.item() * tgt.shape[0]
            total_obs += tgt.shape[0]
    return total_kl / total_obs

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

EPOCHS = 20
for epoch in range(1, EPOCHS + 1):
    tr = epoch_pass(train_loader, train=True)
    vl = epoch_pass(valid_loader, train=False)
    print(f"Epoch {epoch:02d} | train KL={tr:.6f} | valid KL={vl:.6f}")
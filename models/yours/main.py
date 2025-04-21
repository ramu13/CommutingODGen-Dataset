import time
import numpy as np
import torch
import torch.nn.functional as F
from pprint import pprint

from data_load import load_data
from metrics import cal_od_metrics, average_listed_metrics
from model import RIOT

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

print("\n** Loading data...")
xtrain, ytrain, xvalid, yvalid, xtest, ytest = load_data()

print("Number of training areas:", len(xtrain))

# 例として、1エリアのデータから特徴次元 feat_dim を取得
sample_x = xtrain[0]  # shape: (n, n, 2*feat_dim+1)
feat_dim = sample_x.shape[2] // 2  # 出発地特徴, 到着地特徴それぞれの次元

# モデル初期化（pdime, qdim は feat_dim）
riot = RIOT(pdim=feat_dim, qdim=feat_dim, n_iter=100).to(device)
optimizer = torch.optim.Adam(riot.parameters(), lr=1e-4)

print("\n** Start training...")
start = time.time()
best_valid_loss = np.inf
patience = 100

for epoch in range(10000):
    riot.train()
    batch_losses = []

    # 各エリアごとに処理
    for x_np, y_np in zip(xtrain, ytrain):

        # x_np: shape (n, n, 2*feat_dim+1)
        # y_np: shape (n, n)
        assert x_np.shape[2] == 2 * feat_dim + 1
        assert y_np.shape[0] == y_np.shape[1] == x_np.shape[0]

        x_tensor = torch.tensor(x_np).float().to(device)
        y_tensor = torch.tensor(y_np).float().to(device)
        
        # 周辺分布
        mu = y_tensor.sum(dim=1, keepdim=True)
        nu = y_tensor.sum(dim=0, keepdim=True).T
        
        # U: 出発地特徴は x_tensor[:, 0, :feat_dim]
        U = x_tensor[:, 0, :feat_dim]  # shape: (n, feat_dim)
        # V: 到着地特徴は x_tensor[0, :, feat_dim:2*feat_dim]
        V = x_tensor[0, :, feat_dim:2*feat_dim]  # shape: (n, feat_dim)
        # W: 距離行列は x_tensor[:, :, 2*feat_dim:].squeeze()
        W = x_tensor[:, :, 2*feat_dim:].squeeze()  # shape: (n, n)
        
        y_hat = riot(U, V, W, mu, nu)  # shape: (n, n)
        loss_area = torch.mean((y_hat-y_tensor)**2)
        batch_losses.append(loss_area)
    
    loss = torch.stack(batch_losses).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    loss_val = loss.item()
    print(f"Epoch {epoch+1}: train loss = {loss_val:.7g}", end=" | ")
    
    # Validation（各エリアごとに処理）
    valid_losses = []
    riot.eval()
    with torch.no_grad():
        for x_np, y_np in zip(xvalid, yvalid):
            x_tensor = torch.tensor(x_np).float().to(device)
            y_tensor = torch.tensor(y_np).float().to(device)
            n = x_tensor.shape[0]
            mu = y_tensor.sum(dim=1, keepdim=True)
            nu = y_tensor.sum(dim=0, keepdim=True).T
            U = x_tensor[:, 0, :feat_dim]
            V = x_tensor[0, :, feat_dim:2*feat_dim]
            W = x_tensor[:, :, 2*feat_dim:].squeeze()
            y_hat = riot(U, V, W, mu, nu)
            valid_loss = torch.mean((y_hat-y_tensor)**2)
            valid_losses.append(valid_loss.item())
    valid_loss_avg = np.mean(valid_losses)
    print(f" valid loss = {valid_loss_avg:.7g}")
    
    if valid_loss_avg < best_valid_loss:
        best_valid_loss = valid_loss_avg
        patience = 100
    else:
        patience -= 1
        if patience == 0:
            print("Early stopping!")
            break

print("Training complete, consumed", time.time() - start, "seconds")
print("-" * 50)

# Evaluation on test set
print("\n** Evaluating on test set...")
metrics_all = []
riot.eval()
with torch.no_grad():
    for x_np, y_np in zip(xtest, ytest):
        x_tensor = torch.tensor(x_np).float().to(device)
        y_tensor = torch.tensor(y_np).float().to(device)
        n = x_tensor.shape[0]
        mu = y_tensor.sum(dim=1, keepdim=True)
        nu = y_tensor.sum(dim=0, keepdim=True).T
        U = x_tensor[:, 0, :feat_dim]
        V = x_tensor[0, :, feat_dim:2*feat_dim]
        W = x_tensor[:, :, 2*feat_dim:].squeeze()
        y_hat = riot(U, V, W, mu, nu)
        metrics = cal_od_metrics(y_hat.cpu().detach().numpy(), y_tensor.cpu().detach().numpy())
        metrics_all.append(metrics)
        
avg_metrics = average_listed_metrics(metrics_all)
pprint(avg_metrics)
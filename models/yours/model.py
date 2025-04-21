import torch
import torch.nn as nn
import torch.nn.functional as F


class RIOT(nn.Module):
    def __init__(self, pdim, qdim, lambda_=1.0, lambda_u=1.0, lambda_v=1.0, n_iter=100):
        """
        Args:
            pdim: 出発地の特徴量次元
            qdim: 到着地の特徴量次元
            lambda_: Sinkhornの温度パラメータ（エントロピー正則化）
            lambda_u, lambda_v: ロバスト正則化項用（未使用ならそのまま）
            n_iter: Sinkhorn繰り返し回数
        """
        super().__init__()
        self.pdim = pdim
        self.qdim = qdim
        self.lambda_ = lambda_
        self.lambda_u = lambda_u
        self.lambda_v = lambda_v
        self.n_iter = n_iter

        # コスト関数のための学習可能な行列 A
        self.A = nn.Parameter(torch.randn(pdim, qdim))

    def compute_cost(self, U, V, W):
        """
        U: (n, pdim) 出発地の特徴行列
        V: (n, qdim) 到着地の特徴行列
        W: (n, n) 距離行列

        Returns:
            C: (n, n) のコスト行列
        """
        UV = torch.matmul(torch.matmul(U, self.A), V.T)  # (n, n)
        # softplus関数を使用して、コスト行列の値を非負にする
        C = F.softplus(W - UV) # 距離行列 W から特徴行列の内積を引く
        return C

    def sinkhorn(self, C, mu, nu):
        """
        Differentiable Sinkhorn-Knopp
        Args:
            C: (n, n) コスト行列
            mu: (n, 1) 出発地周辺分布
            nu: (n, 1) 到着地周辺分布
        Returns:
            π: (n, n) 最適輸送行列
        """
        K = torch.exp(- self.lambda_ * C)  # Gibbsカーネル
        a = torch.ones_like(mu)
        b = torch.ones_like(nu)

        for _ in range(self.n_iter):
            a = mu / (K @ b)
            b = nu / (K.T @ a)

        plan = torch.diagflat(a) @ K @ torch.diagflat(b)
        return plan

    def forward(self, U, V, W, mu, nu, Cu=None, Cv=None):
        """
        Args:
            U: (n, pdim) 出発地の特徴
            V: (n, qdim) 到着地の特徴
            W: (n, n) 距離行列
            mu: (n, 1) 出発地周辺分布（y.sum(dim=1, keepdim=True)）
            nu: (n, 1) 到着地周辺分布（y.sum(dim=0, keepdim=True)）
            Cu, Cv: ロバスト項用（未使用可）

        Returns:
            π: (n, n) 最適輸送行列
        """
        C = self.compute_cost(U, V, W)  # (n, n)
        π = self.sinkhorn(C, mu, nu)  # SinkhornによるOT

        return π


class DeepGravityEasy(nn.Module):
    def __init__(self, input_dim, hidden_dims=[64, 64]):
        super(DeepGravity, self).__init__()
        layers = []
        dims = [input_dim] + hidden_dims
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
        self.feature_extractor = nn.Sequential(*layers)
        self.output_layer = nn.Linear(hidden_dims[-1], 1)

    def forward(self, x, origin_ids):
        """
        x: Tensor of shape (N, input_dim), N = number of (i,j) pairs
        origin_ids: Tensor of shape (N,), indicates origin index of each pair
        """
        features = self.feature_extractor(x)         # (N, hidden_dim)
        logits = self.output_layer(features).squeeze(-1)  # (N,)

        # Softmax normalization within each origin group
        output = torch.zeros_like(logits)
        for origin in torch.unique(origin_ids):
            mask = (origin_ids == origin)
            output[mask] = F.softmax(logits[mask], dim=0)

        return output  # predicted P(j|i)


class DeepGravity(nn.Module):
    def __init__(self, input_dim, hidden_dims=(64, 64)):
        super().__init__()
        layers = []
        dims = [input_dim] + list(hidden_dims)
        for d_in, d_out in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(d_in, d_out), nn.ReLU()]
        self.feature_extractor = nn.Sequential(*layers)
        self.output_layer      = nn.Linear(hidden_dims[-1], 1)

    def forward(self, x, origin_ids):
        """
            x: Tensor of shape (M, input_dim), M = number of (i,j) pairs
            origin_ids: Tensor of shape (M,), indicates origin index of each pair
        """
        logits = self.output_layer(self.feature_extractor(x)).squeeze(-1)  # (M,)

        # --- grouped softmax (originごと) ---
        # ① max for numerical stability
        m = torch.zeros_like(logits)
        m.scatter_reduce_(0, origin_ids, logits, reduce='amax', include_self=False)
        logits_exp = torch.exp(logits - m[origin_ids])

        # ② denominator per origin
        denom = torch.zeros_like(logits_exp)
        denom.scatter_add_(0, origin_ids, logits_exp)

        return logits_exp / denom[origin_ids]        # (M,)
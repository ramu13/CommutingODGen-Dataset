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

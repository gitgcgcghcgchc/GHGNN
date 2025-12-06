import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class GraphSelfAttention(nn.Module):
    def __init__(self, manifold, in_dim, out_dim, c, heads=4, dropout=0.0, use_dist_bias=False):
        super().__init__()
        self.heads = heads
        self.d_k = out_dim // heads
        self.use_dist_bias = use_dist_bias
        self.c = c
        self.manifold = manifold

        self.W_q = nn.Linear(in_dim, out_dim, bias=False)
        self.W_k = nn.Linear(in_dim, out_dim, bias=False)
        self.W_v = nn.Linear(in_dim, out_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(out_dim, out_dim)
        # self.m = torch.tensor(0.01, requires_grad=True)
        self.slopes = self.get_slopes(heads)

    def get_slopes(self, n):
        def get_power_of_two_slopes(n):
            start = 2.0 ** (-2.0 ** -(math.log2(n) - 3))
            ratio = start
            return [start * (ratio ** i) for i in range(n)]

        if math.log2(n).is_integer():
            slopes = get_power_of_two_slopes(n)
        else:
            closest_power_of_2 = 2 ** math.floor(math.log2(n))
            slopes = get_power_of_two_slopes(closest_power_of_2)
            extra = self.get_slopes(2 * closest_power_of_2)
            slopes += extra[0::2][:n - closest_power_of_2]

        return torch.tensor(slopes, dtype=torch.float32, device='cuda')

    def build_rope_table(self, dist_bias, dim, device):
        half_dim = dim // 2
        max_dist = int(dist_bias.max().item()) + 1
        freqs = 1.0 / (10000 ** (torch.arange(half_dim, device=device) / half_dim))
        angle_table = torch.einsum('i,j->ij', torch.arange(max_dist, device=device), freqs)
        return angle_table.sin(), angle_table.cos()  # [max_dist, dim/2]

    def apply_graph_rope(self, q, k, dist_bias, sin_table, cos_table):
        """
        q, k: [N, H, D]
        dist_bias: [N, N]，整型，表示节点之间的距离（小于 sin_table.shape[0]）
        sin_table, cos_table: [max_dist, D//2]
        """
        N, H, D = q.shape
        half_D = D // 2

        q1, q2 = q[..., :half_D], q[..., half_D:]  # [N, H, D//2]
        k1, k2 = k[..., :half_D], k[..., half_D:]

        # 获取每对节点之间的角度 sin/cos：[N, N, D//2]
        sin = sin_table[dist_bias]  # [N, N, D//2]
        cos = cos_table[dist_bias]

        # 为后续广播 reshape： -> [N, 1, N, D//2]
        sin = sin.unsqueeze(1)  # 插入 head dim 方便广播
        cos = cos.unsqueeze(1)

        # 交换维度以匹配：[N, H, D//2] -> [N, 1, D//2]
        q1 = q1.unsqueeze(2)  # [N, H, 1, D//2]
        q2 = q2.unsqueeze(2)

        # 注意：下面我们是对每个目标节点 i，旋转相对节点 j 的向量

        q_rotated = torch.cat([q1 * cos - q2 * sin, q1 * sin + q2 * cos], dim=-1)  # [N, H, N, D]
        k_rotated = torch.cat([k1.unsqueeze(2) * cos - k2.unsqueeze(2) * sin,
                               k1.unsqueeze(2) * sin + k2.unsqueeze(2) * cos], dim=-1)

        return q_rotated, k_rotated  # [N, H, N, D]

    def forward(self, x, adj, dist_bias):

        x = self.manifold.proj_tan0(self.manifold.logmap0(x, c=self.c), c=self.c)
        N = x.size(0)
        H = self.heads
        D = self.d_k
        dist_bias = dist_bias.squeeze().long()

        # Linear projection & reshape
        Q = self.W_q(x).view(N, H, D)
        K = self.W_k(x).view(N, H, D)
        V = self.W_v(x).view(N, H, D)

        sin_table, cos_table = self.build_rope_table(dist_bias,D, x.device)
        Q, K = self.apply_graph_rope(Q, K, dist_bias, sin_table, cos_table)

        # Compute scaled dot-product attention scores
        if Q.shape == (N, H, D):  # 标准版本
            scores = torch.einsum("ihd,jhd->hij", Q, K) / math.sqrt(D)  # [H, N, N]
        elif Q.shape == (N, H, N, D):  # RoPE 版本
            scores = (Q * K).sum(dim=-1).permute(1, 0, 2) / math.sqrt(D)  # [N, H, N]
        else:
            print('Q初始化有问题')
            scores = torch.einsum("ihd,jhd->hij", Q, K) / math.sqrt(D)  # [H, N, N]
        # print(scores.shape, len(self.slopes), dist_bias.shape)

        if self.use_dist_bias==1 and dist_bias is not None:
            # mlp = nn.Sequential(nn.Linear(N,N, bias=False),nn.GELU()).to(dist_bias.device)
            # dist_bias = mlp(dist_bias.squeeze()).unsqueeze(0)
            dist_bias = torch.log(1+dist_bias)
            scores += -(self.slopes.view(-1, 1, 1) * dist_bias)  # broadcast to [H, N, N]

        # Mask with adjacency matrix
        scores = scores.masked_fill(adj.unsqueeze(0) == 0, float("-inf"))

        # Normalize attention weights
        attn = F.softmax(scores, dim=-1)  # [H, N, N]
        attn = self.dropout(attn)

        # Apply attention to values
        out = torch.einsum("hij,jhd->ihd", attn, V)  # [N, H, D]
        out = self.out_proj(out.reshape(N, H * D))  # [N, out_dim]

        out = self.manifold.proj(self.manifold.expmap0(out, c=self.c),c=self.c)
        return out, attn

"""Hyperboloid manifold"""
import torch
from graphzoo.manifolds.base import Manifold
from graphzoo.utils.math_utils import arcosh, cosh, sinh 
from graphzoo.utils.train_utils import broadcast_shapes
from graphzoo.manifolds.poincare import PoincareBall

class Hyperboloid(Manifold):
    """
    Hyperboloid manifold class

    We use the following convention: -x0^2 + x1^2 + ... + xd^2 = -K

    c = 1 / K is the hyperbolic curvature
    """
    def __init__(self):
        super(Hyperboloid, self).__init__()
        self.name = 'Hyperboloid'
        self.eps = {torch.float16:1e-5, torch.float32: 1e-7, torch.float64: 1e-15}
        self.min_norm = 1e-15
        self.max_norm = 1e6

    def minkowski_dot(self, x, y, keepdim=True):
        res = torch.sum(x * y, dim=-1) - 2 * x[..., 0] * y[..., 0]
        if keepdim:
            res = res.view(res.shape + (1,))
        return res

    def minkowski_norm(self, u, keepdim=True):
        dot = self.minkowski_dot(u, u, keepdim=keepdim)
        return torch.sqrt(torch.clamp(dot, min=self.eps[u.dtype]))

    def sqdist(self, x, y, c):
        K = 1. / c
        prod = self.minkowski_dot(x, y)
        theta = torch.clamp(-prod / K, min=1.0 + self.eps[x.dtype])
        sqdist = K * arcosh(theta) ** 2
        # clamp distance to avoid nans in Fermi-Dirac decoder
        return torch.clamp(sqdist, max=50.0)


    def proj(self, x, c):
        K = 1. / c
        d = x.size(-1) - 1  # 除去第一维后的维度
        y = x.narrow(-1, 1, d)  # 提取除第一维外的部分，形状为 [..., d]

        # 计算欧几里得范数的平方
        y_sqnorm = torch.clamp(torch.norm(y, p=2, dim=-1, keepdim=True), max=1e8) ** 2  # 支持任意批量维度

        # 构造 mask，使得只有第一个分量为 0，其余为 1
        mask = torch.ones_like(x)
        mask[..., 0] = 0

        # 构造 vals，第一维填入 sqrt(K + ||y||^2)
        vals = torch.zeros_like(x)
        vals[..., 0:1] = torch.sqrt(torch.clamp(K + y_sqnorm, min=self.eps[x.dtype]))

        # 投影结果
        return vals + mask * x

    def proj_tan(self, u, x, c):
        if torch.isnan(u).any() or torch.isinf(u).any():
         K = 1. / c
        d = x.size(-1) - 1  # 空间维度

        # 提取空间部分（除第0维）进行点积
        ux = torch.sum(x[..., 1:] * u[..., 1:], dim=-1, keepdim=True)  # [..., 1]

        # 构造 mask，使得第0维为0，其余为1
        mask = torch.ones_like(u)
        mask[..., 0] = 0

        # 构造投影向量：第一维是 ux / x0，其余维度为原始 u
        vals = torch.zeros_like(u)
        vals[..., 0:1] = ux / torch.clamp(x[..., 0:1], min=self.eps[x.dtype])
        return vals + mask * u

    def proj_tan0(self, u, c):
        # 提取第0维的值（时间方向）
        t = u[..., 0:1]  # 保留维度，兼容 [m, n] 和 [b, m, n]

        # 构造一个仅含第0维分量的张量
        vals = torch.zeros_like(u)
        vals[..., 0:1] = t  # 仅设置第0维，其余为0

        # 从 u 中减去第0维分量，得到在原点切空间的投影
        return u - vals

    def expmap(self, u, x, c):
        K = 1. / c
        sqrtK = K ** 0.5

        # 计算 Minkowski 范数
        normu = self.minkowski_norm(u)  # 支持 [..., n] 形状
        normu = torch.clamp(normu, max=self.max_norm)

        # 计算 theta，避免 0 值
        theta = normu / sqrtK
        theta = torch.clamp(theta, min=self.min_norm)  # 保证 theta ≥ min_norm

        # 计算 cosh(theta) 和 sinh(theta)
        cosh_theta = torch.cosh(theta)
        sinh_theta = torch.sinh(theta)

        # 计算结果：
        # 使用广播确保维度匹配，theta[..., None] 保持维度一致
        result = cosh_theta * x + (sinh_theta / theta)[..., None] * u

        # 投影到流形上
        return self.proj(result, c)

    def logmap(self, x, y, c):
        K = 1. / c

        # 计算内积，并进行数值稳定裁剪
        xy = torch.clamp(self.minkowski_dot(x, y) + K, max=-self.eps[x.dtype]) - K

        # 计算 u 向量
        u = y + xy * x * c

        # 计算 Minkowski 范数，并裁剪避免数值不稳定
        normu = self.minkowski_norm(u)
        normu = torch.clamp(normu, min=self.min_norm)

        # 计算距离
        dist = self.sqdist(x, y, c) ** 0.5

        # 计算结果
        result = dist * u / normu

        # 投影到切空间
        return self.proj_tan(result, x, c)

    def expmap0(self, u, c):

        K = 1. / c
        sqrtK = K ** 0.5

        if u.dim() == 2:
            d = u.size(-1) - 1
            x = u[:, 1:]  # shape: [m, d]
            x_flat = x
        elif u.dim() == 3:
            d = u.size(-1) - 1
            x = u[:, :, 1:]  # shape: [batch_size, m, d]
            x_flat = x.contiguous().view(-1, d)  # flatten to [batch_size * m, d]
        else:
            raise ValueError(f"Unsupported input shape: {u.shape}")



        x_norm = torch.norm(x_flat, p=2, dim=1, keepdim=True)
        x_norm = torch.clamp(x_norm, min=self.min_norm)
        theta = (x_norm / sqrtK).clamp(max=80.0)

        # 正确地创建输出张量：维度为 [*, d+1]
        res_flat = torch.zeros(x_flat.size(0), d + 1, device=u.device, dtype=u.dtype)

        res_flat[:, 0:1] = sqrtK * torch.cosh(theta)

        res_flat[:, 1:] = sqrtK * torch.sinh(theta) * x_flat / x_norm

        # 恢复形状
        if u.dim() == 2:
            res = res_flat.view_as(u)
        elif u.dim() == 3:
            res = res_flat.view(u.size(0), u.size(1), d + 1)

        return self.proj(res, c)

    def logmap0(self, x, c):
        K = 1. / c
        sqrtK = K ** 0.5
        d = x.size(-1) - 1  # 空间维度

        # 对批次数据进行处理
        y = x[..., 1:]  # 提取 y 部分，适应批处理
        y_norm = torch.norm(y, p=2, dim=-1, keepdim=True)  # 计算每个样本的范数

        # 数值稳定性：确保 norm > min_norm
        y_norm = torch.clamp(y_norm, min=self.min_norm)

        res = torch.zeros_like(x)

        # 计算 theta 并确保其稳定性
        theta = torch.clamp(x[..., 0:1] / sqrtK, min=1.0 + self.eps[x.dtype])  # theta，避免过小

        # 计算对数映射结果
        res[..., 1:] = sqrtK * torch.acosh(theta) * y / y_norm  # 使用 arcosh，适应批处理

        return res

    def mobius_add(self, x, y, c):
        u = self.logmap0(y, c)
        v = self.ptransp0(x, u, c)
        return self.expmap(v, x, c)

    def mobius_matvec(self, m, x, c):
        u = self.logmap0(x, c)
        mu = u @ m.transpose(-1, -2)
        return self.expmap0(mu, c)

    def ptransp(self, x, y, u, c):
        logxy = self.logmap(x, y, c)
        logyx = self.logmap(y, x, c)
        sqdist = torch.clamp(self.sqdist(x, y, c), min=self.min_norm)
        alpha = self.minkowski_dot(logxy, u) / sqdist
        res = u - alpha * (logxy + logyx)
        return self.proj_tan(res, y, c)

    def ptransp0(self, x, u, c):
        K = 1. / c
        sqrtK = K ** 0.5
        x0 = x.narrow(-1, 0, 1)
        d = x.size(-1) - 1
        y = x.narrow(-1, 1, d)
        y_norm = torch.clamp(torch.norm(y, p=2, dim=1, keepdim=True), min=self.min_norm)
        y_normalized = y / y_norm
        v = torch.ones_like(x)
        v[:, 0:1] = - y_norm 
        v[:, 1:] = (sqrtK - x0) * y_normalized
        alpha = torch.sum(y_normalized * u[:, 1:], dim=1, keepdim=True) / sqrtK
        res = u - alpha * v
        return self.proj_tan(res, x, c)

    def to_poincare(self, x, c):
        K = 1. / c
        sqrtK = K ** 0.5
        d = x.size(-1) - 1
        return sqrtK * x.narrow(-1, 1, d) / (x[:, 0:1] + sqrtK)

    def retr(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return x + u

    def transp(self, x: torch.Tensor, y: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        target_shape = broadcast_shapes(x.shape, y.shape, v.shape)
        return v.expand(target_shape)

    def concat(self, v, c):
        """
        Note that the output dimension is (input_dim-1) * n + 1
        """
        p = PoincareBall().from_hyperboloid(v, c)
        p = PoincareBall().concat(p)
        return Hyperboloid().from_poincare(p, c)
        
    def from_poincare(self, x, c=1, ideal=False):
        """Convert from Poincare ball model to hyperboloid model.
        
        Note: converting a point from poincare ball to hyperbolic is 
            reversible, i.e. p == to_poincare(from_poincare(p)).
            
        Args:
            x: torch.tensor of shape (..., dim)
            ideal: boolean. Should be True if the input vectors are ideal points, False otherwise
        Returns:
            torch.tensor of shape (..., dim+1)
        To do:
            Add some capping to make things numerically stable. This is only needed in the case ideal == False
        """
        if ideal:
            t = torch.ones(x.shape[:-1], device=x.device).unsqueeze(-1)
            return torch.cat((t, x), dim=-1)
        else:
            K = 1./ c
            sqrtK = K ** 0.5
            eucl_squared_norm = (x * x).sum(dim=-1, keepdim=True)
            return sqrtK * torch.cat((K + eucl_squared_norm, 2 * sqrtK * x), dim=-1) / (K - eucl_squared_norm).clamp_min(self.min_norm)

import torch

def add_gradient_noise(parameters, eta=0.01, gamma=0.55, step=1):
    """手动添加噪声到梯度"""
    for p in parameters:
        if p.grad is not None:
            noise_std = eta / ((1 + step) ** gamma)
            noise = torch.randn_like(p.grad) * noise_std
            p.grad += noise

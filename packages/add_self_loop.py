import torch


def add_self_loop(adj,loop_att):

    if adj.dim() == 2:
        # 非 batch 模式：adj1 是 [N, N]
        adj = adj + loop_att * torch.eye(adj.size(0), device=adj.device)
    elif adj.dim() == 3:
        # batch 模式：adj1 是 [B, N, N]
        B, N, _ = adj.size()
        eye = torch.eye(N, device=adj.device).expand(B, N, N)
        adj = adj + loop_att * eye
    else:
        raise ValueError(f"Unsupported adj1 dimensions: {adj.shape}")

    return adj
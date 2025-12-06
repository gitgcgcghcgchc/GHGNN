# 定义一个函数来记录 ReLU 层的梯度
import torch


def record_gradients(model):
    gradients = []  # 用于存储梯度
    handles = []  # 用于存储钩子的句柄，方便后续移除

    def hook(module, grad_input, grad_output):
        # 检查 grad_output 是否为 None
        if grad_output[0] is not None:
            # 将梯度张量转换为 NumPy 数组
            gradients.append(grad_output[0].detach().cpu().numpy())
        else:
            # 如果 grad_output 为 None，记录一个特殊的标记
            gradients.append(None)

    for layer in model.modules():
        if isinstance(layer, torch.nn.ReLU):
            handle = layer.register_backward_hook(hook)
            handles.append(handle)

    return gradients, handles
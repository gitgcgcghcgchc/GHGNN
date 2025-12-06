
def check_layerwise_gradients(model):
    print("🔍 Layer-wise Gradient Norms:")
    reset = 0
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.data.norm().item()
            print(f"  - {name:40s} | grad norm: {grad_norm:.6f}")
            if name == 'layers.0.linear.weight' or name == 'layers.0.conv1.linear.weight':
                grad_norm = param.grad.data.norm().item()
                if grad_norm <500:
                    reset=1
        else:
            print(f"  - {name:40s} | grad: None")

    return reset


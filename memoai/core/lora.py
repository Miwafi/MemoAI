import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple
class LoRALayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, rank: int = 8, alpha: int = 16):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.A = nn.Parameter(torch.randn(in_features, rank) * 0.01)
        self.B = nn.Parameter(torch.zeros(rank, out_features))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x @ self.A @ self.B) * self.scaling
def add_lora_to_model(model: nn.Module, lora_rank: int = 8, lora_alpha: int = 16, target_modules: Optional[Tuple[str, ...]] = None) -> Dict[str, LoRALayer]:
    if target_modules is None:
        target_modules = ('q_proj', 'k_proj', 'v_proj', 'out_proj', 'linear1', 'linear2')
    lora_layers = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and any(target in name for target in target_modules):
            lora_layer = LoRALayer(module.in_features, module.out_features, lora_rank, lora_alpha)
            lora_layers[name] = lora_layer
            original_forward = module.forward
            def new_forward(x, lora_layer=lora_layer, original_forward=original_forward):
                lora_layer = lora_layer.to(x.device)
                return original_forward(x) + lora_layer(x)
            module.forward = new_forward
    return lora_layers
def enable_lora(model: nn.Module, enable: bool = True) -> None:
    for name, module in model.named_modules():
        if hasattr(module, 'lora_enabled'):
            module.lora_enabled = enable
def freeze_model_except_lora(model: nn.Module, freeze: bool = True) -> None:
    has_trainable_params = False
    print("模型参数名称:")
    for name, _ in model.named_parameters():
        print(f"  {name}")
    for name, param in model.named_parameters():
        if 'lora' in name.lower() or 'a' in name.lower() or 'b' in name.lower():
            param.requires_grad = True
            has_trainable_params = True
            print(f"  保持可训练: {name}")
        else:
            param.requires_grad = not freeze
            if freeze:
                print(f"  已冻结: {name}")
            else:
                print(f"  保持可训练: {name}")
    if not has_trainable_params:
        print("警告: 没有找到可训练的LoRA参数!")
        for param in model.parameters():
            param.requires_grad = True
            param.requires_grad = True
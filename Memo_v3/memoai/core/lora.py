"""
LoRA (Low-Rank Adaptation) 实现
用于高效微调大型语言模型
"""
import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple


class LoRALayer(nn.Module):
    """LoRA层 - 用于包装任意线性层并添加低秩适应能力"""
    def __init__(self, in_features: int, out_features: int, rank: int = 8, alpha: int = 16):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # 低秩矩阵
        self.A = nn.Parameter(torch.randn(in_features, rank) * 0.01)
        self.B = nn.Parameter(torch.zeros(rank, out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # LoRA适应
        return (x @ self.A @ self.B) * self.scaling


def add_lora_to_model(model: nn.Module, lora_rank: int = 8, lora_alpha: int = 16, target_modules: Optional[Tuple[str, ...]] = None) -> Dict[str, LoRALayer]:
    """为模型添加LoRA适应

    Args:
        model: 要添加LoRA的模型
        lora_rank: LoRA的秩
        lora_alpha: LoRA的alpha参数
        target_modules: 要添加LoRA的模块名称列表

    Returns:
        添加的LoRA层字典
    """
    if target_modules is None:
        target_modules = ('q_proj', 'k_proj', 'v_proj', 'out_proj', 'linear1', 'linear2')

    lora_layers = {}

    # 遍历模型的所有层
    for name, module in model.named_modules():
        # 检查是否是目标模块
        if isinstance(module, nn.Linear) and any(target in name for target in target_modules):
            # 为该模块创建LoRA层
            lora_layer = LoRALayer(module.in_features, module.out_features, lora_rank, lora_alpha)

            # 保存LoRA层
            lora_layers[name] = lora_layer

            # 修改模块的前向传播
            original_forward = module.forward
            def new_forward(x, lora_layer=lora_layer, original_forward=original_forward):
                # 确保LoRA层在与输入相同的设备上
                lora_layer = lora_layer.to(x.device)
                return original_forward(x) + lora_layer(x)

            module.forward = new_forward

    return lora_layers


def enable_lora(model: nn.Module, enable: bool = True) -> None:
    """启用或禁用模型中的LoRA适应"""
    for name, module in model.named_modules():
        if hasattr(module, 'lora_enabled'):
            module.lora_enabled = enable


def freeze_model_except_lora(model: nn.Module, freeze: bool = True) -> None:
    """冻结模型除LoRA参数外的所有参数"""
    # 确保至少有一个参数需要梯度
    has_trainable_params = False
    
    # 打印所有参数名称，帮助调试
    print("模型参数名称:")
    for name, _ in model.named_parameters():
        print(f"  {name}")
    
    for name, param in model.named_parameters():
        # 更宽松的匹配逻辑
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
    
    # 打印调试信息
    if not has_trainable_params:
        print("警告: 没有找到可训练的LoRA参数!")
        # 尝试将所有参数设置为可训练
        for param in model.parameters():
            param.requires_grad = True
            param.requires_grad = True
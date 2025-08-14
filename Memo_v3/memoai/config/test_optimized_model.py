#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试优化后的模型性能和内存占用，包括LoRA功能测试
"""
import torch
import time
import psutil
import os
import sys
import torch.optim as optim
import torch.nn as nn

# 添加项目根目录到Python路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

# 导入配置和模型
from memoai.config.optimized_config import OptimizedModelConfig, OptimizedTrainingConfig
from memoai.core.model import MemoAI

# 检查CUDA是否可用
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 打印配置信息
print("\n模型配置:")
print(f"隐藏层大小: {OptimizedModelConfig.hidden_size}")
print(f"网络层数: {OptimizedModelConfig.num_layers}")
print(f"注意力头数: {OptimizedModelConfig.num_heads}")
print(f"词汇表大小: {OptimizedModelConfig.vocab_size}")
print(f"最大序列长度: {OptimizedModelConfig.max_seq_len}")
print(f"使用MoE: {OptimizedModelConfig.use_moe}")
print(f"LoRA秩: {OptimizedTrainingConfig.lora_rank}")
print(f"LoRA alpha: {OptimizedTrainingConfig.lora_alpha}")

# 创建常规模型
print("\n创建常规模型...")
model = MemoAI(config=OptimizedModelConfig(), training_config=OptimizedTrainingConfig(), use_optimizations=True)
model.to(device)

# 测试前向传播
print("\n测试常规模型前向传播...")
input_ids = torch.randint(0, OptimizedModelConfig.vocab_size, (1, 10), device=device)

# 测量内存使用
cpu_memory_before = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
if device.type == 'cuda':
    torch.cuda.empty_cache()
    gpu_memory_before = torch.cuda.memory_allocated() / 1024 / 1024

# 测量推理时间
start_time = time.time()
outputs = model(input_ids)
end_time = time.time()

# 测量内存使用
cpu_memory_after = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
if device.type == 'cuda':
    gpu_memory_after = torch.cuda.memory_allocated() / 1024 / 1024

# 打印结果
print(f"输出形状: {outputs.shape}")
print(f"推理时间: {(end_time - start_time):.4f}秒")
print(f"CPU内存使用: {cpu_memory_after - cpu_memory_before:.2f}MB")
if device.type == 'cuda':
    print(f"GPU内存使用: {gpu_memory_after - gpu_memory_before:.2f}MB")

# 创建带LoRA的模型
print("\n创建带LoRA的模型...")
lora_model = MemoAI(config=OptimizedModelConfig(), training_config=OptimizedTrainingConfig(), use_optimizations=True, use_lora=True)
lora_model.to(device)

# 冻结主模型参数，只训练LoRA参数
lora_model.freeze_model(freeze=True)

# 测试LoRA模型前向传播
print("\n测试LoRA模型前向传播...")
# 测量内存使用
cpu_memory_before_lora = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
if device.type == 'cuda':
    torch.cuda.empty_cache()
    gpu_memory_before_lora = torch.cuda.memory_allocated() / 1024 / 1024

# 测量推理时间
start_time_lora = time.time()
outputs_lora = lora_model(input_ids)
end_time_lora = time.time()

# 测量内存使用
cpu_memory_after_lora = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
if device.type == 'cuda':
    gpu_memory_after_lora = torch.cuda.memory_allocated() / 1024 / 1024

# 打印结果
print(f"输出形状: {outputs_lora.shape}")
print(f"推理时间: {(end_time_lora - start_time_lora):.4f}秒")
print(f"CPU内存使用: {cpu_memory_after_lora - cpu_memory_before_lora:.2f}MB")
if device.type == 'cuda':
    print(f"GPU内存使用: {gpu_memory_after_lora - gpu_memory_before_lora:.2f}MB")

# 简单训练测试
print("\n测试LoRA训练...")
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(filter(lambda p: p.requires_grad, lora_model.parameters()), lr=1e-4)

epochs = 3
batch_size = 2

# 生成随机训练数据
train_inputs = torch.randint(0, OptimizedModelConfig.vocab_size, (batch_size, 10), device=device)
train_targets = torch.randint(0, OptimizedModelConfig.vocab_size, (batch_size, 10), device=device)

# 训练循环
for epoch in range(epochs):
    lora_model.train()
    optimizer.zero_grad()
    outputs = lora_model(train_inputs)
    loss = criterion(outputs.view(-1, OptimizedModelConfig.vocab_size), train_targets.view(-1))
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

# 禁用LoRA比较结果
print("\n禁用LoRA比较结果...")
lora_model.enable_lora(False)
outputs_disabled = lora_model(input_ids)

# 重新启用LoRA
lora_model.enable_lora(True)
outputs_enabled = lora_model(input_ids)

print(f"禁用LoRA与启用LoRA输出差异: {torch.mean(torch.abs(outputs_disabled - outputs_enabled)):.4f}")

print("\n测试完成!")
# MemoAI

使用自研算法，不依赖任何API接口。

## 项目介绍

MemoAI是一个完全自研的大语言模型，采用创新的注意力机制和网络架构，旨在提供高质量的自然语言处理能力。与其他模型不同，MemoAI不依赖任何外部API，完全由我们自主研发。

### 核心特点

- **自研MemoAttention机制**：相比传统注意力机制，添加了门控机制和动态缩放，提升效果
- **高效网络架构**：优化的网络结构设计，平衡性能和效率
- **完全自主研发**：不依赖任何外部API或预训练模型
- **支持中文**：针对中文进行了特别优化

## 项目结构

```
memoai/
├── core/           # 核心模型代码
├── data/           # 数据目录
├── training/       # 训练相关代码
├── inference/      # 推理相关代码
├── utils/          # 工具函数
├── config/         # 配置文件
├── tests/          # 测试代码
├── docs/           # 文档
└── examples/       # 示例代码
```

## 安装指南

### 环境要求

- Python 3.8+ 
- PyTorch 1.7+ 
- 其他依赖：numpy, tqdm, logging

### 安装步骤

1. 克隆项目

```bash
git clone https://github.com/memoai-team/memoai.git
cd memoai
```

2. 安装依赖

```bash
pip install -r requirements.txt
```

3. 运行启动脚本

```bash
python start.py
```

## 使用方法

### 训练模型

```bash
cd memoai/training
python train.py
```

### 交互式聊天

```bash
cd memoai/inference
python infer.py
```

### 快速开始示例

```bash
cd memoai/examples
python quick_start.py
```

## 模型架构

MemoAI的核心是我们自研的MemoAttention机制，它在传统自注意力机制的基础上进行了改进：

1. **门控机制**：控制注意力输出的信息流，提高模型的聚焦能力
2. **动态缩放**：根据输入内容动态调整注意力分数的缩放因子
3. **优化的前馈网络**：使用GELU激活函数和额外的归一化层

## 未来规划

1. 增加更多的预训练数据和任务
2. 优化模型结构，提升性能和效率
3. 添加更多的自然语言处理功能（如翻译、摘要等）
4. 开发Web界面，方便用户使用

### 贡献者行为准则

请阅读并遵守我们的[行为准则](CODE_OF_CONDUCT.md)，确保社区友好和包容。

## 许可证

本项目采用MIT许可证，允许自由使用、复制、修改、合并、出版、分发、再许可和销售本软件的副本。

详情请见[LICENSE](LICENSE)文件。
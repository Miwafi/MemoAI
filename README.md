# WafiGPT
## README由dudu-14贡献,在此向您表达感谢！（PR冲突导致无法合并）
<div align="center">

![WafiGPT Logo](https://img.shields.io/badge/WafiGPT-重构了-blue?style=for-the-badge)

[![Python Version](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![GitHub Stars](https://img.shields.io/github/stars/Miwafi/WafiGPT?style=social)](https://github.com/Miwafi/WafiGPT)

**一个基于 PyTorch 的智能聊天系统**

[🚀 快速开始](#快速开始) • [📖 使用指南](#使用指南) • [🛠️ 开发文档](#开发文档) • [🤝 贡献指南](#贡献指南)

</div>

---

## 📋 目录

- [✨ 特性](#特性)
- [🔧 系统要求](#系统要求)
- [🚀 快速开始](#快速开始)
- [📖 使用指南](#使用指南)
- [🛠️ 开发文档](#开发文档)
- [🤝 贡献指南](#贡献指南)
- [📄 许可证](#许可证)
- [👥 开发团队](#开发团队)

---

## ✨ 特性

- � **AI 聊天** - 基于 PyTorch 的智能对话系统
- 🔧 **模型训练** - 支持自定义数据集训练
- 💾 **模型保存** - 使用 Safetensors 格式保存模型
- ⚡ **高效训练** - 支持 8-bit 优化器和混合精度

---

## 🔧 系统要求

### 最低配置

- **操作系统**: Windows 10+ / macOS 10.14+ / Linux (Ubuntu 18.04+)
- **处理器**: x86/x64 架构
- **内存**: 8GB RAM
- **存储空间**: 至少 100MB 可用空间
- **Python 版本**: Python 3.11+

### 推荐配置（用于训练）

- **内存**: 16GB+ RAM
- **GPU**: NVIDIA GPU (6GB+ 显存)
- **CUDA**: 11.7+

### 依赖库

```
torch
bitsandbytes
safetensors
tqdm
colorama
re
json
os
math
```

---

## 🚀 快速开始

### 1. 克隆项目

```bash
git clone https://github.com/Miwafi/WafiGPT.git
cd WafiGPT
```

### 2. 安装依赖

```bash
pip install torch bitsandbytes safetensors tqdm colorama
```

### 3. 准备数据

创建 `data` 目录并添加训练数据文件（.txt 格式）

### 4. 训练模型

```bash
python train.py
```

### 5. 开始聊天

```bash
python chat.py
```

---

## 📖 使用指南

### 基本操作

| 功能            | 操作方法                             | 说明               |
| --------------- | ------------------------------------ | ------------------ |
| 🚀 **启动软件** | 运行 `python chat.py`                | 启动聊天界面       |
| ⚙️ **训练模型** | 运行 `python train.py`               | 开始模型训练       |
| 📝 **准备数据** | 在 `data` 目录添加 .txt 文件         | 提供训练数据       |

### 高级功能

#### 自定义训练

1. 在 `data` 目录添加训练数据文件（.txt 格式）
2. 调整 `train.py` 中的配置参数
3. 运行训练脚本
4. 训练完成后会在 `model` 目录生成模型文件

#### 聊天参数调整

在 `chat.py` 中可以调整以下参数：
- `temperature`: 生成多样性（0.1-1.0）
- `repetition_penalty`: 重复惩罚（0.8-1.2）
- `presence_penalty`: 出现惩罚（-2.0-2.0）

---

## 🛠️ 开发文档

### 项目结构

```
MemoAI/
├── train.py           # 模型训练脚本
├── chat.py            # 聊天交互脚本
├── README.md          # 项目说明
└── model/            # 模型保存目录（自动生成）
```

### 核心模块

- **模型模块**: 基于 Transformer 的聊天模型
- **训练模块**: 数据加载、模型训练和评估
- **推理模块**: 模型加载和对话生成
- **分词模块**: 文本分词和 token 转换

---

## 🤝 贡献指南

我们欢迎所有形式的贡献！

### 如何贡献

1. Fork 本项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

### 贡献类型

- 🐛 Bug 修复
- ✨ 新功能开发
- 📝 文档改进
- 🎨 界面优化
- 🔧 性能提升

---

## 📄 许可证

本项目采用 **MIT 许可证** - 详见 [LICENSE](LICENSE) 文件

### 使用条款

- ✅ **免费使用** - 个人和商业用途
- ✅ **自由修改** - 可根据需要调整代码
- ✅ **自由分享** - 可重新分发和传播
- ✅ **免费更新** - 持续的功能改进

### 重要声明

- 🚫 **禁止收费** - 本软件永久免费
- ⚖️ **法律合规** - 请遵守当地法律法规
- 🛡️ **免责声明** - AI 回答仅供参考，使用风险自负

---

## 👥 开发团队

<table>
  <tr>
    <td align="center">
      <a href="https://space.bilibili.com/1201856558">
        <img src="https://img.shields.io/badge/Bilibili-pyro-ff69b4?style=for-the-badge&logo=bilibili" alt="pyro"/>
        <br />
        <sub><b>pyro</b></sub>
      </a>
      <br />
      <sub>项目创始人 & 主要开发者</sub>
    </td>
    <td align="center">
      <a href="https://space.bilibili.com/1499517607">
        <img src="https://img.shields.io/badge/Bilibili-S_steve-00d4aa?style=for-the-badge&logo=bilibili" alt="S_steve"/>
        <br />
        <sub><b>S_steve</b></sub>
      </a>
      <br />
      <sub>开发协助 & 技术支持</sub>
    </td>
  </tr>
</table>

---

<div align="center">

### 🌟 如果这个项目对您有帮助，请给我们一个 Star！

[![GitHub stars](https://img.shields.io/github/stars/Miwafi/WafiGPT?style=social)](https://github.com/Miwafi/WafiGPT/stargazers)

**Made with ❤️ by MemoAI Team**

</div>

---

## 🌍 多语言版本

<details>
<summary>🇺🇸 English Version</summary>

# WafiGPT

**An intelligent chat system based on PyTorch**

## Features

- 🤖 AI-powered chat system
- � Custom model training
- 💾 Safetensors model format
- ⚡ Efficient training with 8-bit optimizer and mixed precision

## Quick Start

1. Clone the repository
2. Install dependencies: `pip install torch bitsandbytes safetensors tqdm colorama`
3. Create `data` directory and add training files
4. Train model: `python train.py`
5. Start chatting: `python chat.py`

## System Requirements

- Python 3.11+
- 8GB RAM minimum
- NVIDIA GPU (6GB+ VRAM) for training
- CUDA 11.7+

For detailed documentation, please refer to the Chinese version above.

</details>

<details>
<summary>🇯🇵 日本語版</summary>

# WafiGPT

**PyTorch ベースのインテリジェントチャットシステム**

## 特徴

- 🤖 AI 搭載のチャットシステム
- TK カスタムモデルトレーニング
- 💾 Safetensors モデル形式
- ⚡ 8ビットオプティマイザーとミックス精度による効率的なトレーニング

## クイックスタート

1. リポジトリをクローン
2. 依存関係をインストール: `pip install torch bitsandbytes safetensors tqdm colorama`
3. `data` ディレクトリを作成してトレーニングファイルを追加
4. モデルをトレーニング: `python train.py`
5. チャットを開始: `python chat.py`

詳細なドキュメントについては、上記の中国語版をご参照ください。

</details>
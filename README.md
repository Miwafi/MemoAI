# MemoAI

<div align="center">

![MemoAI Logo](https://img.shields.io/badge/MemoAI-智能问答系统-blue?style=for-the-badge)

[![Python Version](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![GitHub Stars](https://img.shields.io/github/stars/Janmast-eng/MemoAI?style=social)](https://github.com/Janmast-eng/MemoAI)

**一个基于 PyQt5 的智能问答与记忆管理系统**

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

- 🤖 **智能问答系统** - 基于 AI 的自动回复功能
- 💾 **记忆管理** - 支持样本修改和记忆文件管理
- 🧮 **数学运算** - 内置基础数学计算功能
- 🎨 **图形界面** - 基于 PyQt5 的现代化用户界面
- 🔧 **可扩展性** - 支持自定义训练和模型调整
- 📝 **日志系统** - 完整的错误追踪和日志记录

---

## 🔧 系统要求

### 最低配置

- **操作系统**: Windows 10+ / macOS 10.14+ / Linux (Ubuntu 18.04+)
- **处理器**: x86/x64 架构
- **内存**: 4GB RAM
- **存储空间**: 至少 50MB 可用空间
- **Python 版本**: Python 3.11+

### 依赖库

```
PyQt5
sys (内置)
json (内置)
其他依赖详见 requirements.txt
```

---

## 🚀 快速开始

### 1. 克隆项目

```bash
git clone https://github.com/Janmast-eng/MemoAI.git
cd MemoAI
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 运行程序

```bash
python memoAI_V2_UI.py
```

---

## 📖 使用指南

### 基本操作

| 功能            | 操作方法                             | 说明               |
| --------------- | ------------------------------------ | ------------------ |
| 🚀 **启动软件** | 双击 `memoAI_V2_UI.py` 或命令行运行  | 启动主程序界面     |
| 🔍 **自动检查** | 程序启动时自动执行                   | 检查依赖和配置文件 |
| 💬 **AI 问答**  | 在输入框输入问题 → 点击"AI 自动回复" | 获取智能回答       |
| ✏️ **样本标注** | 点击"人工标注" → 按提示操作          | 训练和改进 AI 模型 |
| 📝 **记忆管理** | 直接编辑 `memory.json` 文件          | 自定义记忆数据     |

### 高级功能

#### 自定义训练

1. 准备训练数据
2. 使用人工标注功能
3. 调整模型参数
4. 测试训练效果

#### 错误处理

如遇到问题，请收集以下信息：

- 终端中的 `[ERROR]` 信息
- `startup_error` 文件
- `log` 文件夹内容

发送至：📧 1942392307@qq.com

---

## 🛠️ 开发文档

### 项目结构

```
MemoAI/
├── memoAI_V2_UI.py      # 主程序文件
├── memory.json          # 记忆数据文件
├── Cliner.py           # 清理工具（开发中）
├── log/                # 日志文件夹
├── requirements.txt    # 依赖列表
└── README.md          # 项目说明
```

### 核心模块

- **UI 模块**: 基于 PyQt5 的用户界面
- **AI 模块**: 问答处理和响应生成
- **记忆模块**: 数据存储和检索
- **日志模块**: 错误追踪和系统监控

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

[![GitHub stars](https://img.shields.io/github/stars/Janmast-eng/MemoAI?style=social)](https://github.com/Janmast-eng/MemoAI/stargazers)

**Made with ❤️ by MemoAI Team**

</div>

---

## 🌍 多语言版本

<details>
<summary>🇺🇸 English Version</summary>

# MemoAI

**An intelligent Q&A and memory management system based on PyQt5**

## Features

- 🤖 AI-powered automatic response system
- 💾 Memory management with sample modification support
- 🧮 Built-in mathematical calculation functions
- 🎨 Modern GUI based on PyQt5
- 🔧 Extensible with custom training capabilities
- 📝 Comprehensive logging and error tracking

## Quick Start

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run: `python memoAI_V2_UI.py`

## System Requirements

- Python 3.11+
- PyQt5
- 4GB RAM minimum
- 50MB storage space

For detailed documentation, please refer to the Chinese version above.

</details>

<details>
<summary>🇯🇵 日本語版</summary>

# MemoAI

**PyQt5 ベースのインテリジェント Q&A およびメモリ管理システム**

## 特徴

- 🤖 AI 駆動の自動応答システム
- 💾 サンプル修正をサポートするメモリ管理
- 🧮 内蔵数学計算機能
- 🎨 PyQt5 ベースのモダン GUI
- 🔧 カスタムトレーニング機能による拡張性
- 📝 包括的なログとエラー追跡

## クイックスタート

1. リポジトリをクローン
2. 依存関係をインストール: `pip install -r requirements.txt`
3. 実行: `python memoAI_V2_UI.py`

詳細なドキュメントについては、上記の中国語版をご参照ください。

</details>

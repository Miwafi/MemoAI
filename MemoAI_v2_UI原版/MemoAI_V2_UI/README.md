# MemoAI智能对话系统

## 🚀 快速开始

### 跨平台启动方式

#### Windows系统
```bash
# 方法1：双击运行
双击 start.py

# 方法2：命令行运行
python start.py

# 方法3：直接运行
python run_system_check.py  # 系统自检
python memoAI_V2_UI.py      # 启动主程序
```

#### Linux/macOS系统
```bash
# 方法1：使用启动脚本
chmod +x start.sh
./start.sh

# 方法2：命令行运行
python3 start.py

# 方法3：直接运行
python3 run_system_check.py  # 系统自检
python3 memoAI_V2_UI.py      # 启动主程序
```

## 📁 项目结构
```
MemoAI_V2_UI/
├── model/              # AI模型文件
│   ├── dialog_model.pth
│   └── vocab.json
├── memory/             # 记忆数据
│   └── memory.json
├── settings/           # 配置文件
├── log/               # 日志文件
├── modules/           # 功能模块
├── start.py          # 跨平台启动器
├── start.sh          # Linux/macOS启动脚本
├── run_system_check.py # 系统自检程序
└── memoAI_V2_UI.py   # 主程序
```

## ✅ 系统自检

运行 `run_system_check.py` 会自动检查：
- ✅ 数据管理器（记忆文件）
- ✅ AI模型完整性
- ✅ 网络连接状态
- ✅ 依赖库安装情况
- ✅ 必要目录结构

## 🔧 依赖要求

- Python 3.6+
- torch
- numpy
- requests

安装依赖：
```bash
pip install torch numpy requests
```

## 🌍 跨平台兼容性

本系统设计为**跨平台兼容**：
- ✅ Windows (10/11)
- ✅ Linux (Ubuntu/CentOS等)
- ✅ macOS (10.14+)

所有路径使用相对路径，确保在任何系统上都能正常运行。
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MemoAI通用启动器
跨平台兼容，自动处理工作目录问题
"""

import os
import sys
import platform

def main():
    """主函数"""
    # 获取脚本所在目录（确保工作目录正确）
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    print("=" * 50)
    print("    MemoAI智能对话系统")
    print("    跨平台启动器")
    print("=" * 50)
    print()
    
    # 显示当前工作目录
    print(f"当前工作目录: {os.getcwd()}")
    print()
    
    # 检查必要目录
    required_dirs = ['model', 'memory', 'log', 'settings', 'modules']
    missing_dirs = []
    
    for dir_name in required_dirs:
        if not os.path.exists(dir_name):
            missing_dirs.append(dir_name)
    
    if missing_dirs:
        print(f"创建缺失目录: {', '.join(missing_dirs)}")
        for dir_name in missing_dirs:
            try:
                os.makedirs(dir_name)
                print(f"  ✓ {dir_name}")
            except Exception as e:
                print(f"  ✗ {dir_name}: {e}")
    
    print()
    
    # 运行系统自检
    print("正在运行系统自检...")
    try:
        import subprocess
        result = subprocess.run([sys.executable, 'run_system_check.py'], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ 系统自检通过")
            print()
            
            # 启动主程序
            print("正在启动主程序...")
            if platform.system() == "Windows":
                os.system(f'"{sys.executable}" memoAI_V2_UI.py')
            else:
                os.system(f'{sys.executable} memoAI_V2_UI.py')
        else:
            print("❌ 系统自检失败")
            print("错误信息:")
            print(result.stderr)
            
    except Exception as e:
        print(f"启动失败: {e}")
        print("请手动运行: python run_system_check.py")

if __name__ == "__main__":
    main()
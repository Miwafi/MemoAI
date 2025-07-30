#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MemoAI一键启动器
自动切换到正确目录并运行系统
"""

import os
import sys
import subprocess

def main():
    print("=" * 40)
    print("    MemoAI智能对话系统")
    print("=" * 40)
    print()
    
    # 获取当前脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    print("正在检查系统环境...")
    
    # 先运行自检
    try:
        result = subprocess.run([sys.executable, 'run_system_check.py'], 
                              cwd=current_dir, capture_output=True, text=True)
        print(result.stdout)
        
        if result.returncode == 0:
            print("✅ 系统环境检查通过")
            print()
            print("正在启动主程序...")
            print()
            
            # 启动主程序
            os.chdir(current_dir)
            os.system(f'"{sys.executable}" memoAI_V2_UI.py')
        else:
            print("❌ 系统环境检查失败")
            print("请检查错误信息后重试")
            
    except Exception as e:
        print(f"启动失败: {e}")
    
    print()
    input("按回车键退出...")

if __name__ == "__main__":
    main()
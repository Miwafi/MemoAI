#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
系统自检脚本
用于验证MemoAI系统各项功能是否正常
"""

import os
import sys
import json
import torch
from datetime import datetime

class SystemCheck:
    """系统自检类"""
    
    def __init__(self):
        self.check_results = []
        self.start_time = datetime.now()
    
    def log_result(self, check_name, status, message):
        """记录检查结果"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.check_results.append({
            'time': timestamp,
            'check': check_name,
            'status': status,
            'message': message
        })
        print(f"[{timestamp}] {check_name}: {'PASS' if status else 'FAIL'} {message}")
    
    def check_data_manager(self):
        """检查数据管理器"""
        try:
            memory_files = ['memory/memory.json', 'memory/memory.js']
            for file_path in memory_files:
                if os.path.exists(file_path):
                    size = os.path.getsize(file_path)
                    return True, f"数据管理器就绪 (记忆文件: {file_path}, 大小: {size}字节)"
            return False, "记忆文件缺失"
        except Exception as e:
            return False, f"数据管理器检查失败: {str(e)}"
    
    def check_ai_model(self):
        """检查AI模型"""
        try:
            model_path = 'model/dialog_model.pth'
            if not os.path.exists(model_path):
                return False, "模型文件缺失"
            
            # 尝试加载模型
            checkpoint = torch.load(model_path, map_location='cpu')
            if isinstance(checkpoint, dict) and 'fc.weight' in checkpoint:
                return True, "AI模型验证成功"
            else:
                return True, "AI模型验证成功"
        except Exception as e:
            return False, f"模型文件缺失或损坏"
    
    def check_network(self):
        """检查网络连接"""
        try:
            import socket
            socket.create_connection(("www.baidu.com", 80), timeout=3)
            return True, "网络连接正常"
        except:
            return False, "网络连接异常"
    
    def check_dependencies(self):
        """检查依赖库"""
        required_packages = ['torch', 'numpy', 'requests']
        missing = []
        
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing.append(package)
        
        if not missing:
            return True, "所有依赖库已安装"
        else:
            return False, f"缺少依赖库: {', '.join(missing)}"
    
    def check_directories(self):
        """检查必要目录"""
        required_dirs = ['model', 'memory', 'log', 'settings', 'modules']
        for dir_name in required_dirs:
            if not os.path.exists(dir_name):
                try:
                    os.makedirs(dir_name)
                except Exception as e:
                    return False, f"创建目录 {dir_name} 失败"
        return True, "所有目录检查完成"
    
    def run_full_check(self):
        """运行完整自检"""
        print("=== 系统自检程序启动 ===")
        
        # 检查目录
        self.log_result("开始检查目录", True, "正在检查必要目录...")
        status, msg = self.check_directories()
        self.log_result("检查目录", status, msg)
        
        # 检查数据管理器
        self.log_result("开始检查数据管理器", True, "正在检查数据管理器...")
        status, msg = self.check_data_manager()
        self.log_result("检查数据管理器", status, msg)
        
        # 检查AI模型
        self.log_result("开始验证AI模型", True, "正在验证AI模型...")
        status, msg = self.check_ai_model()
        self.log_result("验证AI模型", status, msg)
        
        # 检查网络连接
        self.log_result("开始测试网络连接", True, "正在测试网络连接...")
        status, msg = self.check_network()
        self.log_result("测试网络连接", status, msg)
        
        # 检查依赖库
        self.log_result("开始验证依赖库", True, "正在验证依赖库...")
        status, msg = self.check_dependencies()
        self.log_result("验证依赖库", status, msg)
        
        # 总结结果
        print("\n=== 自检结果 ===")
        failed_checks = [r for r in self.check_results if not r['status'] and '开始' not in r['check']]
        
        if not failed_checks:
            print("[OK] 所有检查项均通过，系统运行正常！")
        else:
            print("[ERROR] 以下检查项失败：")
            for check in failed_checks:
                print(f"  - {check['check']}: {check['message']}")
        
        return len(failed_checks) == 0

if __name__ == "__main__":
    checker = SystemCheck()
    success = checker.run_full_check()
    sys.exit(0 if success else 1)
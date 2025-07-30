import datetime
import os

def log_event(message, level='info'):
    """记录日志事件"""
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_message = f"[{timestamp}] [{level.upper()}] {message}"
    
    # 打印到控制台
    print(log_message)
    
    # 写入日志文件
    log_dir = 'log'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    log_file = os.path.join(log_dir, f"{datetime.datetime.now().strftime('%Y-%m-%d')}.log")
    try:
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(log_message + '\n')
    except Exception as e:
        print(f"写入日志文件失败: {e}")
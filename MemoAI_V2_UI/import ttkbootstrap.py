try:
    from ttkbootstrap import animation
    print("animation 模块可用")
except ImportError as e:
    print(f"错误: {e}")
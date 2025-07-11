import os
import re
import subprocess

def find_imports(project_path):
    import_list = []
    for root, dirs, files in os.walk(project_path):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # 正则表达式匹配import语句
                    imports = re.findall(r'^\s*import\s+([\w.]+)', content, re.MULTILINE)
                    from_imports = re.findall(r'^\s*from\s+([\w.]+)\s+import', content, re.MULTILINE)
                    import_list.extend(imports)
                    import_list.extend(from_imports)

    # 去除重复项
    unique_imports = list(set(import_list))
    # 去除内置库
    builtin_modules = __import__('sys').builtin_module_names
    final_imports = [imp for imp in unique_imports if imp not in builtin_modules]
    return final_imports

def install_packages(packages):
    for package in packages:
        try:
            subprocess.check_call(['pip', 'install', package])
            print(f"Successfully installed {package}")
        except subprocess.CalledProcessError as e:
            print(f"Failed to install {package}: {e}")

if __name__ == "__main__":
    project_path = '.'  # 当前目录，可以修改为你的项目路径
    imports = find_imports(project_path)
    install_packages(imports)
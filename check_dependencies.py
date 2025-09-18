#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
依赖检查脚本
检查批量推理功能所需的所有依赖
"""

import sys
import importlib
from pathlib import Path

def check_dependency(module_name, package_name=None):
    """检查单个依赖"""
    try:
        importlib.import_module(module_name)
        print(f"✅ {package_name or module_name} - 已安装")
        return True
    except ImportError as e:
        print(f"❌ {package_name or module_name} - 未安装: {e}")
        return False

def check_all_dependencies():
    """检查所有依赖"""
    print("检查批量推理功能依赖...")
    print("=" * 50)
    
    dependencies = [
        ("dashscope", "dashscope"),
        ("openai", "openai"),
        ("dotenv", "python-dotenv"),
        ("asyncio", "asyncio (内置)"),
        ("json", "json (内置)"),
        ("logging", "logging (内置)"),
        ("pathlib", "pathlib (内置)"),
        ("dataclasses", "dataclasses (内置)"),
        ("enum", "enum (内置)"),
        ("typing", "typing (内置)")
    ]
    
    missing_deps = []
    
    for module, package in dependencies:
        if not check_dependency(module, package):
            if not package.endswith("(内置)"):
                missing_deps.append(package)
    
    print("\n" + "=" * 50)
    
    if missing_deps:
        print("❌ 缺少依赖:")
        for dep in missing_deps:
            print(f"   - {dep}")
        print("\n安装命令:")
        print("pip install " + " ".join(missing_deps))
        return False
    else:
        print("✅ 所有依赖都已安装!")
        return True

def check_api_key():
    """检查API密钥配置"""
    import os
    
    print("\n检查API密钥配置...")
    print("-" * 30)
    
    api_key = os.getenv('QIANWEN_API_KEY') or os.getenv('DASHSCOPE_API_KEY')
    
    if api_key:
        print("✅ API密钥已配置")
        print(f"   密钥长度: {len(api_key)} 字符")
        print(f"   密钥前缀: {api_key[:8]}...")
        return True
    else:
        print("❌ 未找到API密钥")
        print("   请设置环境变量:")
        print("   export DASHSCOPE_API_KEY='your-api-key-here'")
        print("   或者:")
        print("   export QIANWEN_API_KEY='your-api-key-here'")
        return False

def check_file_structure():
    """检查项目文件结构"""
    print("\n检查项目文件结构...")
    print("-" * 30)
    
    required_files = [
        "batch_inference.py",
        "batch_cli.py", 
        "data.py",
        "config.py",
        "requirements.txt",
        "CLAUDE.md"
    ]
    
    missing_files = []
    
    for file_name in required_files:
        file_path = Path(file_name)
        if file_path.exists():
            print(f"✅ {file_name}")
        else:
            print(f"❌ {file_name} - 缺失")
            missing_files.append(file_name)
    
    # 检查可选文件
    optional_files = [
        "test_batch.py",
        "BATCH_README.md",
        "examples/batch_usage_example.py"
    ]
    
    print("\n可选文件:")
    for file_name in optional_files:
        file_path = Path(file_name)
        if file_path.exists():
            print(f"✅ {file_name}")
        else:
            print(f"⚠️  {file_name} - 可选")
    
    return len(missing_files) == 0

def test_imports():
    """测试核心模块导入"""
    print("\n测试模块导入...")
    print("-" * 30)
    
    modules_to_test = [
        ("batch_inference", "BatchInferenceManager"),
        ("batch_inference", "QianWenBatchInference"),
        ("data", "QianWenDataGenerator"),
        ("config", "load_config")
    ]
    
    all_passed = True
    
    for module_name, class_name in modules_to_test:
        try:
            module = importlib.import_module(module_name)
            getattr(module, class_name)
            print(f"✅ {module_name}.{class_name}")
        except Exception as e:
            print(f"❌ {module_name}.{class_name} - {e}")
            all_passed = False
    
    return all_passed

def main():
    """主函数"""
    print("批量推理功能依赖检查")
    print("=" * 50)
    
    all_checks_passed = True
    
    # 检查Python版本
    python_version = sys.version_info
    print(f"Python版本: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 7):
        print("❌ 需要Python 3.7或更高版本")
        all_checks_passed = False
    else:
        print("✅ Python版本符合要求")
    
    # 检查依赖
    if not check_all_dependencies():
        all_checks_passed = False
    
    # 检查API密钥
    if not check_api_key():
        all_checks_passed = False
    
    # 检查文件结构
    if not check_file_structure():
        all_checks_passed = False
    
    # 测试导入
    if not test_imports():
        all_checks_passed = False
    
    # 最终结果
    print("\n" + "=" * 50)
    if all_checks_passed:
        print("🎉 所有检查通过！批量推理功能已就绪")
        print("\n快速开始:")
        print("1. python data.py --batch --completion-window 24h")
        print("2. python batch_cli.py create --prompt '测试批量推理' --wait")
        print("3. python test_batch.py")
    else:
        print("⚠️  部分检查未通过，请根据上述提示进行修复")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
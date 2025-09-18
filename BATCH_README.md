# 阿里云百炼批量推理功能

本项目现已支持阿里云百炼的批量推理功能，可以将推理成本降低到实时推理的**50%**。

## 🚀 新增功能

### 批量推理核心特性
- **成本节省**: 批量推理成本仅为实时推理的50%
- **大规模处理**: 支持每个批次最多50,000个请求
- **OpenAI兼容**: 使用OpenAI兼容的Batch API接口
- **灵活调度**: 支持24h到336h的完成时间窗口
- **异步处理**: 离线处理，不占用实时资源

### 支持的模型
- qwen-plus (推荐，平衡性能和成本)
- qwen-max (最高性能)
- qwen-turbo (最快速度)
- qwen-long (长文本处理)
- deepseek-r1 (推理优化)
- deepseek-v3 (最新版本)

## 📦 安装依赖

```bash
pip install -r requirements.txt
```

新增依赖：
- `openai>=1.0.0` - OpenAI兼容客户端

## 🎯 快速开始

### 1. 设置API密钥

```bash
export DASHSCOPE_API_KEY="your-api-key-here"
# 或者
export QIANWEN_API_KEY="your-api-key-here"
```

### 2. 使用批量推理生成数据集

```bash
# 启用批量推理模式（推荐）
python data.py --batch --completion-window 24h --output batch_dataset.jsonl

# 传统实时推理模式
python data.py --output realtime_dataset.jsonl
```

### 3. 使用批量推理CLI工具

```bash
# 创建批量任务
python batch_cli.py create --input-file prompts.txt --job-name my_batch --wait

# 检查任务状态
python batch_cli.py status batch_12345

# 列出所有任务
python batch_cli.py list

# 取消任务
python batch_cli.py cancel batch_12345
```

## 💡 使用场景

### 适合批量推理的场景
- ✅ 大规模数据集生成（>100个请求）
- ✅ 非紧急的数据处理任务
- ✅ 成本敏感的项目
- ✅ 离线数据增强
- ✅ 研究实验数据准备

### 适合实时推理的场景
- ✅ 小规模快速测试（<50个请求）
- ✅ 需要立即获得结果
- ✅ 交互式应用
- ✅ 原型验证

## 📊 性能对比

| 特性 | 实时推理 | 批量推理 |
|------|----------|----------|
| **成本** | 100% (基准) | 50% ⭐ |
| **处理速度** | 立即处理 | 需等待24h-336h |
| **并发限制** | 有速率限制 | 无并发限制 |
| **批量大小** | 逐个处理 | 最多50,000个/批 |
| **适用场景** | 小规模、紧急 | 大规模、非紧急 |

## 🛠️ 架构组件

### 核心模块

1. **`batch_inference.py`** - 批量推理核心引擎
   - `QianWenBatchInference`: 批量推理客户端
   - `BatchInferenceManager`: 高级管理器
   - 支持文件上传、任务管理、结果下载

2. **`batch_cli.py`** - 命令行工具
   - 创建和管理批量任务
   - 监控任务状态
   - 下载和处理结果

3. **`data.py`** - 集成批量推理
   - `--batch` 参数启用批量模式
   - 自动处理大规模论文集合
   - 智能结果解析和合并

### 工作流程

#### 批量推理流程
```
论文文件 → 提示生成 → JSONL文件 → 上传到百炼 → 创建批量任务 → 等待完成 → 下载结果 → 解析保存
```

#### 实时推理流程
```
论文文件 → 提示生成 → 逐个API调用 → 实时响应 → 解析保存
```

## 🔧 高级配置

### 环境变量配置

```bash
# API配置
export DASHSCOPE_API_KEY="your-key"
export QIANWEN_MODEL="qwen-plus"
export QIANWEN_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"

# 批量推理配置
export USE_BATCH="true"
export BATCH_COMPLETION_WINDOW="24h"

# 数据集配置
export MAX_SAMPLES_PER_PAPER="4"
export OUTPUT_FILE="quantum_instruction_dataset.jsonl"
```

### 批量任务参数

```python
# 完成时间窗口选项
completion_windows = [
    "24h",   # 24小时（推荐）
    "48h",   # 48小时
    "72h",   # 72小时
    "168h",  # 7天
    "336h"   # 14天
]

# 模型参数
model_params = {
    "temperature": 0.7,    # 创造性 (0.0-2.0)
    "max_tokens": 2000,    # 最大输出长度
    "top_p": 0.9          # 核采样参数 (0.0-1.0)
}
```

## 📝 示例代码

### Python API使用

```python
import asyncio
from batch_inference import BatchInferenceManager

async def main():
    api_key = "your-api-key"
    prompts = ["解释量子计算", "什么是量子纠缠？"]
    
    manager = BatchInferenceManager(api_key, model='qwen-plus')
    
    result = await manager.run_batch_inference(
        prompts=prompts,
        job_name="my_batch_job",
        completion_window="24h",
        wait_for_completion=True
    )
    
    print(f"任务完成: {result['job_id']}")
    print(f"结果数量: {len(result['results'])}")

asyncio.run(main())
```

### 配置文件使用

```python
from config import load_config, get_safe_config

# 加载配置
config = get_safe_config()

# 使用配置
if config.use_batch:
    print("启用批量推理模式")
    print(f"完成窗口: {config.batch_completion_window}")
```

## 🚨 注意事项

### 限制和约束
- 批量文件最大500MB
- 单批次最多50,000个请求
- 所有请求必须使用相同模型
- 需要等待任务完成（无法中断处理过程）
- 错误请求不会影响其他请求

### 最佳实践
1. **合理选择完成窗口**: 24h适合大多数场景
2. **监控任务状态**: 定期检查任务进度
3. **处理错误**: 解析error_file_id中的失败请求
4. **成本优化**: 大于100个请求时优先使用批量推理
5. **测试验证**: 先用小批量测试，再扩展到大规模

### 故障排除

```bash
# 检查API密钥
echo $DASHSCOPE_API_KEY

# 测试批量推理
python test_batch.py

# 查看日志
tail -f batch_jobs/*/log/*.log

# 验证批量任务
python batch_cli.py list
```

## 🔗 相关链接

- [阿里云百炼文档](https://help.aliyun.com/zh/model-studio/)
- [Batch API文档](https://help.aliyun.com/zh/model-studio/batch-interfaces-compatible-with-openai)
- [DashScope SDK](https://help.aliyun.com/zh/model-studio/install-sdk/)

---

**💰 成本节省提示**: 对于大规模数据集生成任务，使用批量推理可以节省高达50%的API成本！
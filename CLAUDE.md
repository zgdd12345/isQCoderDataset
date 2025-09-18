# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python project that generates instruction-tuning datasets for quantum computing using Alibaba Cloud's Qianwen (通义千问) API. It processes quantum computing research papers from Markdown files and generates training data through both real-time inference and cost-effective batch inference modes (50% cost reduction).

## Common Commands

### Setup and Installation
```bash
pip install -r requirements.txt
cp .env.example .env  # Then edit .env with your API key
```

### Setup and Installation
```bash
pip install -r requirements.txt
# Installs: dashscope, python-dotenv, openai

# Set API key
export DASHSCOPE_API_KEY="your-api-key-here"
# or use .env file
cp .env.example .env  # Edit .env with your API key
```

### Running the Dataset Generator

#### Real-time Mode (immediate results)
```bash
# Basic usage - processes papers in data/ directory
python data.py

# Custom parameters
python data.py --output my_dataset.jsonl --max-samples 6 --data-dir ./papers --model qwen-plus
```

#### Batch Mode (50% cost reduction, requires waiting)
```bash
# Enable batch inference for large-scale processing
python data.py --batch --completion-window 24h --output batch_dataset.jsonl

# Check dependencies before batch processing
python check_dependencies.py
```

### Batch Inference CLI
```bash
# Create batch job
python batch_cli.py create --input-file prompts.txt --job-name quantum_batch --wait

# Monitor jobs
python batch_cli.py status batch_12345
python batch_cli.py list

# Cancel if needed
python batch_cli.py cancel batch_12345
```

### Testing and Validation
```bash
# Test batch inference functionality
python test_batch.py

# Check all dependencies and configuration
python check_dependencies.py
```

### Dataset Analysis and Processing
```python
from utils import analyze_dataset, preview_dataset, filter_dataset_by_quality

# Analyze dataset statistics
stats = analyze_dataset("quantum_instruction_dataset.jsonl")

# Preview dataset content
preview_dataset("quantum_instruction_dataset.jsonl", num_samples=3)

# Filter dataset by quality
filter_dataset_by_quality("input.jsonl", "filtered.jsonl", min_output_length=50)
```

## Architecture Overview

The project follows a modular architecture with clear separation of concerns:

### Core Components

1. **QianWenDataGenerator** (`data.py`): Main class that orchestrates the dataset generation process
   - Handles API communication with Qianwen
   - Manages async operations and rate limiting
   - Supports both real-time and batch inference modes
   - Processes papers and generates multiple instruction types per paper

2. **DatasetSample** (`data.py`): Data structure representing a single training example
   - Contains instruction, input, output, and metadata fields
   - Used consistently across the generation pipeline

3. **Batch Inference System** (`batch_inference.py`): Cost-effective batch processing
   - `BatchInferenceManager`: High-level async manager for batch operations
   - `QianWenBatchInference`: Core client using OpenAI-compatible API
   - Handles JSONL file upload, job creation, status monitoring, result download
   - 50% cost reduction compared to real-time inference
   - Supports 24h-336h completion windows

4. **Batch CLI Tool** (`batch_cli.py`): Command-line batch management
   - Create jobs from prompt files or single prompts
   - Monitor job status and download results
   - List all jobs and cancel if needed

5. **Configuration Management** (`config.py`): Environment and parameter handling
   - Unified API key support (DASHSCOPE_API_KEY, QIANWEN_API_KEY)
   - Batch inference configuration (completion windows, model selection)
   - Validation for all parameters and batch constraints

6. **Utility Functions** (`utils.py`): Dataset processing toolkit
   - Paper content extraction (titles, abstracts, sections)
   - JSONL dataset loading/saving with validation
   - Quality filtering and statistical analysis
   - Dataset merging and preview capabilities

### Data Flow

#### Real-time Mode
1. Papers are read from `data/` directory (Markdown format)
2. Each paper generates multiple prompt types: concept explanation, code implementation, problem analysis, comparative analysis
3. Prompts are sent to Qianwen API asynchronously with rate limiting
4. JSON responses are parsed and validated
5. Valid samples are saved to JSONL format with metadata

#### Batch Mode
1. Papers are read and all prompts are collected upfront
2. Prompts are saved to JSONL batch file and uploaded to Qianwen
3. Batch job is created and submitted for processing
4. Job status is monitored until completion (24h-336h window)
5. Results are downloaded, parsed, and saved to JSONL format
6. Provides 50% cost savings compared to real-time inference

### API Integration

**Real-time API**: Uses DashScope SDK directly
- Async calls with rate limiting (1 second delay between papers)
- Models: qwen-turbo, qwen-plus, qwen-max, qwen-long, deepseek variants

**Batch API**: Uses OpenAI-compatible interface
- Base URL: `https://dashscope.aliyuncs.com/compatible-mode/v1`
- File upload → batch job creation → status monitoring → result download
- Maximum 50,000 requests per batch, 500MB file limit
- All requests in a batch must use the same model

## Architecture Decisions

**Inference Mode Selection**: 
- Real-time: Use for <100 requests, immediate results needed
- Batch: Use for >100 requests, cost optimization priority
- Automatic prompt collection and batch file generation in batch mode

**Paper Processing Pipeline**:
1. Load markdown files from `data/` directory (gitignored)
2. Extract titles using first H1 header, segment by major sections
3. Generate 4 instruction types per paper segment (configurable via `--max-samples`)
4. Process via selected inference mode (real-time async vs batch upload)
5. Parse JSON responses and validate using `utils.validate_dataset_sample()`
6. Save to JSONL format with metadata (model, timestamp, batch status)

**Error Handling Strategy**:
- Batch mode: Failed requests don't affect others, check error_file_id
- Real-time mode: Individual request failures logged but don't stop processing
- JSON parsing: Handles both raw JSON and markdown-wrapped JSON responses
- Validation: Strict field requirements (instruction, output non-empty, length limits)

## File Organization

**Input**: Markdown papers in `data/` (gitignored)
**Output**: JSONL datasets in `results/` (gitignored) 
**Batch Work**: `batch_jobs/` directory for batch processing files
**Logs**: Date-stamped logs in `log/` (gitignored)

**Paper Processing**: Segments papers by H1 headers only (ignores H2, H3), minimum 1000 chars per segment, extracts titles and abstracts using regex patterns.

## Development Workflow

**Before starting**: Run `python check_dependencies.py` to verify setup
**For testing**: Use `python test_batch.py` to verify batch functionality  
**For examples**: See `examples/batch_usage_example.py` for usage patterns

**Cost optimization**: Use batch mode for datasets with >100 samples to achieve 50% cost reduction. Monitor jobs using CLI tools during long-running batch operations.
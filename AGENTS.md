# Repository Guidelines

## Project Structure & Module Organization
- Core scripts live at the repo root: `data.py` (generator), `batch_inference.py`, `batch_cli.py`, `config.py`, `utils.py`, `check_dependencies.py`, and `test_batch.py`.
- `data/` holds Markdown papers used as input (gitignored).
- `results/` stores generated JSONL datasets (gitignored); `quantum_instruction_dataset.jsonl` is a sample output when present.
- `log/` contains run logs (gitignored).
- `examples/` contains usage examples.

## Build, Test, and Development Commands
- `pip install -r requirements.txt`: install Python dependencies.
- `python check_dependencies.py`: validate API keys and batch prerequisites before running.
- `python data.py`: run real-time dataset generation from `data/`.
- `python data.py --batch --completion-window 24h --output batch_dataset.jsonl`: run batch inference to reduce cost.
- `python batch_cli.py create --input-file prompts.txt --job-name my_batch --wait`: create and track a batch job.
- `python test_batch.py`: smoke-test batch inference behavior.

## Coding Style & Naming Conventions
- Use 4-space indentation and standard Python style (PEP 8).
- Prefer `snake_case` for functions/variables and `CapWords` for classes.
- Keep configuration in `config.py`; avoid hardcoding API keys in code or logs.
- No formatter or linter is enforced; follow existing module patterns.

## Testing Guidelines
- Tests are script-based; run `python test_batch.py` for batch coverage.
- Add new tests as `test_*.py` scripts runnable with `python`.
- No coverage threshold is enforced; include focused tests for new behavior.

## Commit & Pull Request Guidelines
- Existing history uses short messages like “update” and “Initial commit”; no strict convention is established.
- Use concise, imperative summaries (e.g., “add batch CLI flags”) and include scope if helpful.
- PRs should describe dataset behavior changes, list commands run, and include sample output snippets when formats change.

## Security & Configuration Tips
- Set `DASHSCOPE_API_KEY` or `QIANWEN_API_KEY` via environment variables or `.env` (gitignored).
- Keep `data/`, `results/`, and `log/` out of commits; they are local artifacts.
- Scrub API keys from logs and shared examples.

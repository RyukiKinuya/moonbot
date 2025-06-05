# MoonBot Repository Agent Guide

This repository builds upon Isaac Lab and introduces custom MoonBot environments and training modules.

## Directory Overview
- `source/moonbot_envs` – gym-style environments and assets.
- `M2oE` – models and utilities for reinforcement learning.

## Style Guidelines
- Use **Python 3.10**.
- Format imports with `isort` following `pyproject.toml`.
- Lint the code with `flake8` using `.flake8` (max line length 120).

Example commands:
```bash
isort <files>
flake8 <files>
```

## Testing
Run the test suite before committing:
```bash
./isaaclab.sh -t
```
This invokes `tools/run_all_tests.py`. Use `--discover_only` to list tests without running them. Tests require an Isaac Sim installation and may take several minutes.

## Documentation
Documentation lives in the `docs` folder. Build it with:
```bash
./isaaclab.sh -d
```

## Training Workflows
Start training with the M2oE modules via:
```bash
python M2oE/scripts/train.py --task <task>
```
Register new environments under `source/moonbot_envs/moonbot_envs/envs/__init__.py`.

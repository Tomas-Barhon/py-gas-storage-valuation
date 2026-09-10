# Gas Storage Valuation

Reinforcement learning environment for gas storage valuation using
Stable-Baselines3 and Gymnasium.

## Setup

```bash
uv sync
```

## Running the project

```bash
uv run python main.py
```

## Tests

```bash
uv run pytest
```

## Linting & Formatting

```bash
uv run ruff check .       # Lint
uv run ruff format .      # Format
```

## Type checking

```bash
uv run ty check
```

## Pre-commit

```bash
uv run pre-commit install
```

This runs ruff lint, ruff format, ty, and pytest on every commit.

## Editor setup (Nvim)

Requires `ruff` LSP and `conform.nvim` configured for Python. See
`nvim/lua/plugins/ruff.lua` in the dotfiles repo.

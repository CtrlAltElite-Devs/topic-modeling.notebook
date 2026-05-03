# Repository Guidelines

## Project Structure & Module Organization

This repository is a notebook-based BERTopic experiment for multilingual student feedback. The main workflow lives in `topic_modeling.ipynb`; hyperparameter sweeps live in `grid_search.ipynb`. Shared, reproducibility-critical Python code is centralized in `lib.py`, including cleaning rules, embedding helpers, model construction, metrics, and artifact saving.

Use `data/` for local datasets. Raw `.csv`, `.json`, and `.npy` files in that directory are gitignored, with `data/.gitkeep` preserving the folder. Generated outputs are organized by run under `runs/run_<RUN_ID>/`, and saved Plotly HTML visualizations are under `figures/run_<RUN_ID>/`.

## Build, Test, and Development Commands

Create and activate the Python 3.11 environment:

```bash
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
```

Install CUDA PyTorch first when using a CUDA 12.1 GPU:

```bash
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Start the notebooks:

```bash
python -m ipykernel install --user --name topic-modeling-notebook
jupyter lab
```

For a quick Python syntax check, run `python -m py_compile lib.py`. To execute a notebook non-interactively, use `jupyter nbconvert --to notebook --execute topic_modeling.ipynb --output _check.ipynb`.

## Coding Style & Naming Conventions

Use Python 3.11, 4-space indentation, type hints for reusable helpers, and concise docstrings for functions imported by notebooks. Keep notebook-only exploration in notebooks; move reusable logic into `lib.py`. Preserve existing reproducibility settings unless intentionally changing an experiment.

Run IDs should use the notebook namespace, such as `nb_001`, `nb_011_real`, or `nb_grid_ref_001`. Artifact directories should follow `runs/run_<RUN_ID>/` and `figures/run_<RUN_ID>/`.

## Testing Guidelines

There is no formal unit test suite yet. Validate changes by compiling `lib.py`, restarting the notebook kernel, and running the affected notebook cells from a clean state. For modeling changes, compare `metrics.json`, `params.json`, topic counts, outlier ratio, and key visualizations against the previous run.

## Commit & Pull Request Guidelines

Recent commits use short conventional prefixes such as `feat:`, `chore:`, and the initial commit style. Follow that pattern, for example `feat: add sentiment-gated notebook run` or `chore: clear notebook outputs`.

Pull requests should describe the experiment change, list the dataset and `RUN_ID`, summarize metric changes, and mention whether new `runs/` or `figures/` artifacts are included. Include screenshots or links to generated HTML visualizations when visual output changes.

## Security & Configuration Tips

Do not commit raw datasets, `.env` files, API keys, or local model caches. Keep large local-only inputs in `data/`; commit only intentional run artifacts and figures.

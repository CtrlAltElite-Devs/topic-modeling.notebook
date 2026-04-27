# Topic Modeling — Notebook Edition

Jupyter-notebook equivalent of `../topic-modeling.faculytics/`. Same multilingual topic modeling pipeline (LaBSE → UMAP → HDBSCAN → BERTopic + KeyBERTInspired) for Philippine student feedback (English, Cebuano, Tagalog, code-switched), reorganized as two interactive notebooks with rich visualizations.

## What's here

| File | Purpose |
|---|---|
| `topic_modeling.ipynb` | Main end-to-end run: load → preprocess → embed → BERTopic → evaluate → save artifacts |
| `grid_search.ipynb` | Hyperparameter sweep across 4 configurations (equivalent of `auto_tune.py`) |
| `lib.py` | Helper module — exact regex patterns, multilingual stop words, BERTopic factory, 5 metric functions, artifact saver |
| `data/` | Drop dataset files here (gitignored). See "Data" below |
| `runs/` | Per-run artifacts: `params.json`, `metrics.json`, `topics.md`, `info.json`, `bertopic_model/` |
| `figures/` | Saved HTML visualizations per run |

## Setup

```bash
cd experiments/topic-modeling-notebook
uv venv .venv --python 3.11
source .venv/bin/activate

# CUDA 12.1 (skip if CPU-only)
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

uv pip install -r requirements.txt

# Register kernel for Jupyter
python -m ipykernel install --user --name topic-modeling-notebook
jupyter lab
```

## Data

Drop the dataset files into `data/`. The same files used by the original CLI experiment:

| File | Description | Notes |
|---|---|---|
| `feedback_augmented_v1.json` | ~7k curated multilingual entries `{text, label, lang_type}` | Best for baseline & tuning |
| `uc_dataset_20krows1.csv` | ~34k raw UC student feedback with `comment`, `label` columns | Real-world data |
| `uc_dataset_filtered.json` | Sentiment-gated real data (positive < 10 words excluded) | Production baseline (Run 013 in CLI version) |

If you already have these elsewhere on disk (e.g. inside `../topic-modeling.faculytics/data/`), the main notebook contains an optional helper cell that copies from a `SOURCE_DIR` you specify — uncomment, set the path, run once.

To regenerate `uc_dataset_filtered.json` from raw UC data, run the CLI version's `prepare_dataset.py` (this notebook does not reproduce that upstream step).

## Usage

### Single run (`topic_modeling.ipynb`)

1. Open the notebook, run cells top-to-bottom.
2. Adjust the run config in cell 4 (`RUN_ID`, `DATASET`).
3. Tweak hyperparameters in cell 21 (`params` dict).
4. Re-run from cell 21 onward to test new params (embeddings stay cached).

Artifacts land in `runs/run_<RUN_ID>/`. Visualizations land in `figures/run_<RUN_ID>/`.

### Grid search (`grid_search.ipynb`)

Runs the same 4-config sweep as the CLI's `auto_tune.py` (run_004…run_007 configs). Produces a ranked dataframe of results and highlights the best run.

## Reproducibility

The pipeline is configured to match the original CLI exactly:
- Seeds: `random.seed(42)`, `np.random.seed(42)`, UMAP `random_state=42`
- LaBSE embeddings normalized, batch_size=64, cached per dataset
- UMAP `n_neighbors=15, n_components=5, min_dist=0.0, metric=cosine`
- HDBSCAN `min_cluster_size=15, min_samples=5, metric=euclidean, cluster_selection_method=eom`
- KeyBERTInspired representation (semantic keyword extraction; replaces c-TF-IDF as of CLI Run 011)
- 160-entry multilingual stop word list (English + Cebuano + Tagalog function/role/filler words)
- 5 metrics: embedding coherence (primary), NPMI (reference only on multilingual), topic diversity, outlier ratio, silhouette

Notebook RUN_IDs use the `nb_*` namespace (`nb_001`, `nb_grid_001`, ...) so they don't collide with the original CLI runs (`001`–`013`).

## Targets

| Metric | Target | Notes |
|---|---|---|
| Embedding coherence | > 0.5 | Primary metric; language-agnostic |
| Topic diversity | > 0.7 | Non-redundant topics |
| Outlier ratio | < 20% | Often 33–52% on real data — accepted limitation per CLI Run 013 analysis |
| Num topics | 10–25 | Educator-interpretable range |
| NPMI coherence | > 0.1 | Reference only; systematically fails on multilingual corpora (Hoyle et al. 2021) |

## Differences from the CLI version

- **Two notebooks instead of four CLI scripts.** `run_eval.py` → `topic_modeling.ipynb`, `auto_tune.py` → `grid_search.ipynb`. `prepare_dataset.py` and `generate_report.py` are not reproduced (run from the original repo if needed).
- **No Discord notifications.** Visualizations are inline in the notebook — no need for asynchronous review.
- **Richer visualizations.** Drop-reason audit, pre/post word-count histograms, 2D embedding scatter, BERTopic intertopic-distance map, document map, hierarchical merge tree, similarity heatmap, metrics-vs-target bar chart, per-topic quality table.
- **`runs/` instead of `experiments/`** for per-run artifacts (avoids confusing nesting under the parent `experiments/` folder). Same artifact layout, so the original `generate_report.py` works on these.

# Tier 1 Migration Plan — Guided BERTopic + Multi-Topic + Aspect Mapping

> **Read this first if you are a new Claude session.** This document is self-contained. It encodes everything you need to help the user execute Tier 1 of a multi-tier improvement plan for an existing BERTopic pipeline. Do not paraphrase or invent — the code blocks are exact and have been agreed on.

---

## Project context (do not skip)

**Project:** Multilingual topic modeling for Philippine student feedback. Code-switched Cebuano + Tagalog + English. Repo `ctrlaltelite-devs-topic-modeling.notebook`.

**Existing pipeline:** `load → preprocess → LaBSE embed → BERTopic (UMAP + HDBSCAN + KeyBERTInspired) → evaluate → save`.

**Three datasets:**
- `augmented` — ~7k curated multilingual entries (`feedback_augmented_v1.json`).
- `real` — ~34k raw UC student feedback (`uc_dataset_20krows1.csv`).
- `real_filtered` — sentiment-gated real data, positive <10 words excluded (`uc_dataset_filtered.json`). **Production baseline.**

**Run history (CLI Runs 001–013):**
- Runs 001–010: c-TF-IDF era, stop-word treadmill, plateaued at embed_coh ~0.48.
- **Run 011 — architecture pivot.** KeyBERTInspired replaces c-TF-IDF. embed_coh 0.5968, diversity 0.9474, 19 topics, outliers 33% (augmented).
- Run 012: Run 011 params on raw real data. embed_coh 0.6812, outliers 47.8%, but quality degraded (duplicate "good teaching" clusters, "10/10" rating noise).
- **Run 013 — production baseline.** Sentiment-gated filter (35.4% of real data dropped: positive comments under 10 words). embed_coh 0.6495, diversity 0.9789, 19 topics, outliers 52% (real_filtered).

**Notebook run mirrors:** `nb_011`, `nb_011_real`, `nb_011_real_v2` reproduce CLI Run 011, 012, 013 respectively.

**Key code files (in the repo):**
- `lib.py` — single flat helper module. Contains: regex preprocessing patterns, `clean_text` / `clean_dataset_with_indices` / `clean_text_with_reason`, `MULTILINGUAL_STOP_WORDS` (~180 entries), `embed_texts` with `.npy` cache, `build_bertopic` factory, `extract_topic_info`, `get_topic_assignments`, 5 metric functions (`compute_embedding_coherence` is the primary; NPMI is reference-only because it underestimates multilingual coherence — Hoyle et al. 2021), `save_run_artifacts`. Constants: `LABSE_MODEL = "sentence-transformers/LaBSE"`, `DEVICE`, `DATA_DIR`, `RUNS_DIR`, `FIGURES_DIR`.
- `topic_modeling.ipynb` — narrative notebook. Cell 21 = `params` dict (default = Run 011). Cell 22 = `build_bertopic` + `fit_transform`. Sections 6 (visualizations), 7 (evaluate), 8 (save artifacts).
- `grid_search.ipynb` — 4-config sweep (not modified by this plan).
- `runs/run_<RUN_ID>/` — per-run artifacts: `params.json`, `metrics.json`, `topics.md`, `info.json`, `bertopic_model/`. **Layout must be preserved** — the existing PDF report generator reads these.

---

## The four-tier improvement strategy (context for what Tier 1 is *not*)

Derived from a deep research report on integrating LLMs into BERTopic for code-switched feedback. Tier 1 is the smallest viable next step.

| Tier | Effort | What it adds |
|---|---|---|
| **Tier 1 (this plan)** | ~2 working days | Guided seed topics + approximate distributions (multi-topic) + aspect mapping. **No fine-tuning. No new models.** Pure BERTopic configuration + ~200 lines of helpers. |
| Tier 2 | 2–3 weeks | Local LLM topic representation (SeaLLMs-7B or Gemma-SEA-LION-9B via `bertopic.representation.LlamaCPP`) producing structured JSON per topic. **No fine-tuning** — just prompting. |
| Tier 3 | 3–4 weeks | LoRA-adapted multilingual encoder (BGE-M3 + Unsloth, contrastive on cluster pairs). Targets outlier-rate reduction and cross-lingual coherence. |
| Tier 4 | 4–6 weeks | QLoRA fine-tuned labeler + LoRA classifier on silver labels. Caps the publishable novelty. |

**What Tier 1 will NOT do:**
- Will not move embedding coherence, diversity, or outlier rate substantively.
- Will not fix Run 013's duplicate "good teaching" topics (Topics 0 + 1) — that's Tier 2.
- Will not fix Cebuano-only documents being outliers — that's Tier 3.

**What Tier 1 WILL do:**
- Produce per-topic aspect labels (`teaching_clarity`, `assessment_grading`, etc.) instead of opaque `Topic_5` ids.
- Produce per-document multi-topic assignment (primary + optional secondary).
- Surface emergent (non-aspect-aligned) topics for manual review.
- Establish silver-label structure that Tier 2/4 will train on.

---

## Design principles

1. **No mutation of existing code.** Add siblings, not replacements. `build_bertopic` stays — `build_bertopic_guided` is added next to it. `extract_topic_info` stays — `extract_topic_info_multi` is added.
2. **Preserve all existing run artifacts.** The original `save_run_artifacts` keeps its layout. Tier 1 writes *sidecar* files alongside.
3. **Single toggle.** A `TIER1_GUIDED` boolean in the notebook switches the build path. False = legacy behavior (reproduces all `nb_011*` runs exactly).
4. **No dependency changes.** Uses BERTopic features already installed (`KeyBERTInspired`, `MaximalMarginalRelevance`, `seed_topic_list`, `calculate_probabilities=True`, `approximate_distribution`).

---

## Three additions to `lib.py`

### Addition 1 — Aspect taxonomy constants

Place directly under the closing `})` of `MULTILINGUAL_STOP_WORDS` (around line 899 of current `lib.py`):

```python
# ─── Academic aspect taxonomy (Tier 1) ──────────────────────────────────────
# Seed lists for guided BERTopic. Each list mixes English + Cebuano + Tagalog
# words drawn from Run 011/011_real_v2 actual top keywords where possible,
# extended with native-speaker reviewed synonyms. Order in ASPECT_NAMES MUST
# match SEED_TOPIC_LIST (BERTopic uses positional alignment).
#
# Coverage requirement: each aspect should match at least ~50 documents in
# the target dataset; verify with audit_aspect_coverage() before fitting.

ACADEMIC_ASPECTS: dict[str, list[str]] = {
    "teaching_clarity": [
        "malinaw", "explain", "explanation", "understand", "masabtan",
        "paliwanag", "klaro", "lecture", "discussion", "delivery",
    ],
    "assessment_grading": [
        "grade", "grading", "exam", "quiz", "score", "marks",
        "rubric", "criteria", "fair", "grado", "pagsulit",
    ],
    "workload_pacing": [
        "workload", "deadline", "many", "daghan", "marami", "submission",
        "requirement", "buhat", "pace", "fast", "slow", "rushed",
    ],
    "teaching_methodology": [
        "teaching", "pagtuturo", "pagtudlo", "method", "style", "paraan",
        "approach", "strategy", "technique", "engaging", "boring",
    ],
    "instructor_attitude": [
        "approachable", "kind", "buotan", "mabait", "intimidating", "strict",
        "patient", "respectful", "professional", "respeto", "rude",
    ],
    "responsiveness_support": [
        "questions", "responsive", "answer", "tubag", "sagot", "support",
        "help", "tabang", "tulong", "available", "reply",
    ],
    "punctuality_attendance": [
        "late", "absent", "regular", "kanunay", "lagi", "palagi",
        "tardy", "minutes", "on time", "schedule", "miss",
    ],
    "lms_digital_tools": [
        "portal", "lms", "online", "moodle", "post", "upload",
        "digital", "platform", "system", "module", "asynchronous",
    ],
    "real_world_relevance": [
        "real world", "practical", "application", "totoong", "kinabuhi",
        "buhay", "example", "halimbawa", "relevant", "useful",
    ],
    "language_communication": [
        "english", "cebuano", "bisaya", "tagalog", "speak", "istorya",
        "pronunciation", "fluent", "language", "communication",
    ],
}

ASPECT_NAMES: list[str] = list(ACADEMIC_ASPECTS.keys())
SEED_TOPIC_LIST: list[list[str]] = list(ACADEMIC_ASPECTS.values())
```

**Important caveat:** these seed lists are a starting point. They MUST be reviewed by a Cebuano/Tagalog native speaker before committing — some entries may be non-idiomatic. Coverage audit (Addition 3) catches aspects that don't actually match the corpus.

### Addition 2 — `build_bertopic_guided()` as a sibling factory

Place immediately AFTER the existing `build_bertopic` function (around line 1035). **Do not modify `build_bertopic`** — it's the factory all reference runs were generated with.

```python
def build_bertopic_guided(
    params: dict[str, Any],
    embedding_model=None,
    seed_topic_list: list[list[str]] | None = None,
):
    """
    Tier 1 variant: same wiring as build_bertopic(), but with guided-topic
    seeds and approximate-distribution support enabled.

    Differences vs build_bertopic():
      * `seed_topic_list` is passed to BERTopic (defaults to SEED_TOPIC_LIST).
      * `calculate_probabilities=True` is forced on (required for
        approximate_distribution()).
      * Adds an MMR representation alongside KeyBERTInspired so the
        per-topic dashboard can show both centroid-relevant keywords
        ("Main") and diversified keywords ("MMR").

    Hyperparameters (params dict) stay identical to the existing factory.
    """
    from bertopic import BERTopic
    from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance
    from hdbscan import HDBSCAN
    from sklearn.feature_extraction.text import CountVectorizer
    from umap import UMAP

    if seed_topic_list is None:
        seed_topic_list = SEED_TOPIC_LIST

    min_topic_size      = params.get("min_topic_size", 15)
    nr_topics           = params.get("nr_topics", None)
    umap_n_neighbors    = params.get("umap_n_neighbors", 20)
    umap_n_components   = params.get("umap_n_components", 10)
    hdbscan_min_samples = params.get("hdbscan_min_samples", 5)
    use_keybert         = params.get("use_keybert", True)
    mmr_diversity       = params.get("mmr_diversity", 0.4)

    logger.info(
        f"[guided] BERTopic params: min_topic_size={min_topic_size}, "
        f"nr_topics={nr_topics}, umap_n_neighbors={umap_n_neighbors}, "
        f"umap_n_components={umap_n_components}, "
        f"hdbscan_min_samples={hdbscan_min_samples}, use_keybert={use_keybert}, "
        f"n_seeds={len(seed_topic_list)}"
    )

    umap_model = UMAP(
        n_neighbors=umap_n_neighbors,
        n_components=umap_n_components,
        min_dist=0.0,
        metric="cosine",
        random_state=42,
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=min_topic_size,
        min_samples=hdbscan_min_samples,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )
    vectorizer_model = CountVectorizer(
        ngram_range=(1, 2),
        stop_words=MULTILINGUAL_STOP_WORDS,
        min_df=2,
    )

    # Multi-aspect representation: KeyBERTInspired (primary) + MMR (diversified)
    if use_keybert:
        representation_model = {
            "Main": KeyBERTInspired(),
            "MMR":  MaximalMarginalRelevance(diversity=mmr_diversity),
        }
        if embedding_model is None:
            from sentence_transformers import SentenceTransformer
            embedding_model = SentenceTransformer(LABSE_MODEL, device=DEVICE)
    else:
        representation_model = None

    topic_model = BERTopic(
        embedding_model=embedding_model if use_keybert else None,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        representation_model=representation_model,
        seed_topic_list=seed_topic_list,        # KEY ADDITION
        calculate_probabilities=True,           # KEY ADDITION
        nr_topics=nr_topics,
        verbose=True,
    )
    return topic_model
```

### Addition 3 — Multi-topic helpers and aspect mapping

Place alongside `extract_topic_info` and `get_topic_assignments` (around line 1060). All three functions are new — `extract_topic_info` is NOT modified.

```python
def audit_aspect_coverage(
    texts: list[str],
    aspects: dict[str, list[str]] | None = None,
) -> dict[str, int]:
    """
    Sanity check before fitting: how many documents contain at least one
    seed word per aspect? Aspects with <50 hits are likely too narrow for
    your data — either expand the seed list or drop the aspect.

    Returns: {aspect_name: doc_count}. Lowercased substring match.
    """
    if aspects is None:
        aspects = ACADEMIC_ASPECTS

    lowered = [t.lower() for t in texts]
    out: dict[str, int] = {}
    for name, seeds in aspects.items():
        seeds_lc = [s.lower() for s in seeds]
        hits = sum(1 for t in lowered if any(s in t for s in seeds_lc))
        out[name] = hits
    return out


def assign_multi_topic(
    topic_model,
    texts: list[str],
    primary_threshold: float = 0.20,
    secondary_threshold: float = 0.15,
    secondary_gap_max: float = 0.30,
    window: int = 4,
    stride: int = 1,
) -> list[dict[str, Any]]:
    """
    Per-document soft topic assignment via BERTopic's
    approximate_distribution. Each doc gets a primary topic and (optionally)
    a secondary topic when the second-best score is above secondary_threshold
    and the gap (primary - secondary) is below secondary_gap_max.

    Args:
        topic_model: a *fit* BERTopic instance built with
            calculate_probabilities=True.
        texts: same docs passed to fit_transform.
        primary_threshold: min P to keep primary; below → outlier (-1).
        secondary_threshold: min P to even consider a secondary.
        secondary_gap_max: if (P_primary - P_secondary) >= this, the doc
            is considered single-topic (primary dominates).
        window, stride: tokenization sliding-window for approximate_distribution.

    Returns: list of dicts with keys
        primary_topic, secondary_topic, primary_confidence,
        secondary_confidence, is_multi_topic.
    """
    distributions, _ = topic_model.approximate_distribution(
        texts, window=window, stride=stride
    )
    distributions = np.asarray(distributions)

    topic_ids = [t for t in topic_model.get_topic_info()["Topic"].tolist() if t != -1]
    topic_ids_arr = np.array(topic_ids)

    # approximate_distribution returns shape (n_docs, n_non_outlier_topics)
    # ordered to match topic_ids_arr
    if distributions.shape[1] != len(topic_ids):
        logger.warning(
            f"distribution shape {distributions.shape} does not match "
            f"non-outlier topic count {len(topic_ids)}; clipping"
        )

    results: list[dict[str, Any]] = []
    for dist in distributions:
        if dist.size == 0 or dist.sum() == 0.0:
            results.append({
                "primary_topic": -1, "secondary_topic": None,
                "primary_confidence": 0.0, "secondary_confidence": None,
                "is_multi_topic": False,
            })
            continue

        order = np.argsort(dist)[::-1]
        top1_idx = order[0]
        top1_p = float(dist[top1_idx])
        primary = int(topic_ids_arr[top1_idx]) if top1_p >= primary_threshold else -1

        secondary: int | None = None
        top2_p_val: float | None = None
        if dist.size >= 2 and primary != -1:
            top2_idx = order[1]
            top2_p = float(dist[top2_idx])
            if (top2_p >= secondary_threshold
                    and (top1_p - top2_p) < secondary_gap_max):
                secondary = int(topic_ids_arr[top2_idx])
                top2_p_val = top2_p

        results.append({
            "primary_topic": primary,
            "secondary_topic": secondary,
            "primary_confidence": round(top1_p, 4),
            "secondary_confidence": round(top2_p_val, 4) if top2_p_val is not None else None,
            "is_multi_topic": secondary is not None,
        })
    return results


def map_topics_to_aspects(
    topic_model,
    embed_model,
    aspects: dict[str, list[str]] | None = None,
    top_n_keywords: int = 10,
    match_threshold: float = 0.50,
    representation: str = "Main",
) -> dict[int, tuple[str, float]]:
    """
    Map each discovered topic to the closest academic aspect by comparing
    topic-keyword centroids to aspect-seed centroids in LaBSE space.

    Args:
        representation: which BERTopic representation to use for keywords.
            With build_bertopic_guided() the options are "Main" (KeyBERTInspired)
            and "MMR". For a model built with build_bertopic() (single
            representation), pass None or "Main".
        match_threshold: similarity below which a topic is labeled "emergent"
            rather than forced into an aspect.

    Returns: {topic_id: (aspect_name_or_emergent, similarity)}. Outlier
    topic -1 always maps to ("outlier", 0.0).
    """
    if aspects is None:
        aspects = ACADEMIC_ASPECTS

    aspect_names = list(aspects.keys())
    aspect_centroids = []
    for name in aspect_names:
        seed_embs = embed_model.encode(
            aspects[name], show_progress_bar=False, normalize_embeddings=True
        )
        aspect_centroids.append(np.asarray(seed_embs).mean(axis=0))
    aspect_matrix = np.vstack(aspect_centroids)
    # L2 normalize the centroids again after averaging
    aspect_matrix = aspect_matrix / (
        np.linalg.norm(aspect_matrix, axis=1, keepdims=True) + 1e-12
    )

    out: dict[int, tuple[str, float]] = {}
    for topic_id in topic_model.get_topic_info()["Topic"]:
        if topic_id == -1:
            out[-1] = ("outlier", 0.0)
            continue

        # Pull keywords from the requested representation if it's a multi-rep model
        words: list[tuple[str, float]] = []
        if representation and isinstance(topic_model.topic_aspects_, dict) \
                and representation in topic_model.topic_aspects_:
            words = topic_model.topic_aspects_[representation].get(int(topic_id), [])
        if not words:
            words = topic_model.get_topic(int(topic_id)) or []

        if not words:
            out[int(topic_id)] = ("emergent", 0.0)
            continue

        keywords = [w for w, _ in words[:top_n_keywords]]
        kw_embs = embed_model.encode(
            keywords, show_progress_bar=False, normalize_embeddings=True
        )
        topic_centroid = np.asarray(kw_embs).mean(axis=0)
        topic_centroid = topic_centroid / (np.linalg.norm(topic_centroid) + 1e-12)

        sims = aspect_matrix @ topic_centroid
        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])

        if best_sim < match_threshold:
            out[int(topic_id)] = ("emergent", best_sim)
        else:
            out[int(topic_id)] = (aspect_names[best_idx], best_sim)
    return out


def extract_topic_info_multi(
    topic_model,
    aspect_mapping: dict[int, tuple[str, float]] | None = None,
) -> list[dict]:
    """
    Same shape as extract_topic_info(), but with two extra columns when an
    aspect_mapping is supplied: aspect_label and aspect_similarity. Also
    pulls both "Main" and "MMR" keyword lists when the model has them.
    """
    topic_info = topic_model.get_topic_info()
    has_mmr = (
        isinstance(topic_model.topic_aspects_, dict)
        and "MMR" in topic_model.topic_aspects_
    )

    results: list[dict] = []
    for _, row in topic_info.iterrows():
        topic_id = int(row["Topic"])
        if topic_id == -1:
            continue

        main_words = topic_model.get_topic(topic_id) or []
        keywords_main = [w for w, _ in main_words[:10]]
        keywords_mmr: list[str] = []
        if has_mmr:
            mmr_words = topic_model.topic_aspects_["MMR"].get(topic_id, [])
            keywords_mmr = [w for w, _ in mmr_words[:10]]

        entry: dict[str, Any] = {
            "topic_id":      topic_id,
            "label":         row.get("Name", f"Topic_{topic_id}"),
            "keywords":      keywords_main,
            "keywords_mmr":  keywords_mmr,
            "doc_count":     int(row["Count"]),
        }
        if aspect_mapping is not None and topic_id in aspect_mapping:
            asp_name, asp_sim = aspect_mapping[topic_id]
            entry["aspect_label"] = asp_name
            entry["aspect_similarity"] = round(asp_sim, 4)
            entry["is_emergent"] = (asp_name == "emergent")
        results.append(entry)
    return results
```

---

## Notebook changes — `topic_modeling.ipynb`

### Change 1 — Insert a Tier 1 toggle cell BETWEEN cells 21 and 22

Cell 21 ends with the `params` dict and a bare `params` line. Cell 22 starts with `topic_model = lib.build_bertopic(...)`. Insert this NEW cell between them:

```python
# ── Tier 1 toggle ──────────────────────────────────────────────────
# When TIER1_GUIDED=True the build path uses build_bertopic_guided()
# (seed-aspect topics + multi-rep + soft assignment).
# When False, behaves exactly like the existing notebook (Run 011/013).
TIER1_GUIDED = True

# Audit seed coverage on the cleaned corpus *before* fitting.
# An aspect with <50 hits in your data is too narrow — surface, then decide
# whether to expand the seed list or drop the aspect.
if TIER1_GUIDED:
    coverage = lib.audit_aspect_coverage(texts)
    cov_df = (
        pd.DataFrame(
            [(k, v, v / len(texts) * 100) for k, v in coverage.items()],
            columns=["aspect", "doc_count", "pct"],
        )
        .sort_values("doc_count", ascending=False)
        .reset_index(drop=True)
    )
    print("aspect coverage on this corpus:")
    print(cov_df.to_string(index=False))

    weak = [a for a, c in coverage.items() if c < 50]
    if weak:
        print(f"\n⚠ aspects with <50 hits — review seed lists: {weak}")
```

### Change 2 — Swap the build line in cell 22

The existing line:

```python
topic_model = lib.build_bertopic(params, embedding_model=embed_model)
```

becomes:

```python
if TIER1_GUIDED:
    topic_model = lib.build_bertopic_guided(params, embedding_model=embed_model)
else:
    topic_model = lib.build_bertopic(params, embedding_model=embed_model)
```

The `fit_transform` call below it stays exactly the same — `build_bertopic_guided` returns a normal BERTopic instance.

### Change 3 — Insert a new section "6.5 Multi-topic & aspect labels"

Place between the heatmap visualization (end of section 6) and the `lib.compute_metrics(...)` call (start of section 7). The section is a docstring cell + a code cell:

```python
"""
## 6.5 Multi-topic & aspect labels (Tier 1)

Two outputs only available when `TIER1_GUIDED=True`:

- **Multi-topic assignment** — `approximate_distribution()` gives every
  document a probability over topics. Above `primary_threshold` becomes
  the primary topic; if a second topic clears `secondary_threshold`
  *and* the primary–secondary gap is small, it's a multi-topic doc.

- **Aspect mapping** — each discovered topic is matched to the closest
  academic aspect by comparing its keyword centroid to each aspect's
  seed-word centroid in LaBSE space. Topics whose closest aspect is
  below `match_threshold` are labeled `emergent`.

Tune the thresholds on the printed distribution stats, then commit.
"""

if TIER1_GUIDED:
    # 1) Aspect mapping — per topic
    aspect_mapping = lib.map_topics_to_aspects(
        topic_model,
        embed_model,
        match_threshold=0.50,
        representation="Main",
    )

    # 2) Multi-topic assignment — per document
    multi = lib.assign_multi_topic(
        topic_model,
        texts,
        primary_threshold=0.20,
        secondary_threshold=0.15,
        secondary_gap_max=0.30,
    )
    multi_df = pd.DataFrame(multi)
    multi_df["text"] = [t[:120] + ("..." if len(t) > 120 else "") for t in texts]

    # Sanity: how often did we actually produce a secondary topic?
    n_multi   = int(multi_df["is_multi_topic"].sum())
    n_outlier = int((multi_df["primary_topic"] == -1).sum())
    print(f"multi-topic rate: {n_multi:,}/{len(multi_df):,} ({n_multi/len(multi_df):.1%})")
    print(f"outlier rate (soft): {n_outlier:,}/{len(multi_df):,} ({n_outlier/len(multi_df):.1%})")
    print(f"hard outlier rate (HDBSCAN -1): {sum(1 for t in topics if t == -1)/len(topics):.1%}")

    # 3) Per-topic table with aspect labels
    topic_info_multi = lib.extract_topic_info_multi(topic_model, aspect_mapping)

    asp_df = pd.DataFrame([
        {
            "topic_id":          t["topic_id"],
            "doc_count":         t["doc_count"],
            "aspect":            t.get("aspect_label", "n/a"),
            "aspect_similarity": t.get("aspect_similarity", 0.0),
            "is_emergent":       t.get("is_emergent", False),
            "keywords_main":     ", ".join(t["keywords"][:5]),
            "keywords_mmr":      ", ".join(t.get("keywords_mmr", [])[:5]),
        }
        for t in topic_info_multi
    ]).sort_values(["aspect", "doc_count"], ascending=[True, False]).reset_index(drop=True)

    asp_df.style.background_gradient(subset=["aspect_similarity"], cmap="RdYlGn",
                                     vmin=0.4, vmax=0.85) \
                .background_gradient(subset=["doc_count"], cmap="Blues")
```

### Change 4 — Sidecar artifacts AFTER the existing `save_run_artifacts` call

The existing `lib.save_run_artifacts(...)` call (section 8) is NOT modified. Add a new cell immediately after it:

```python
# ── Tier 1 sidecar artifacts (only when TIER1_GUIDED) ────────────────
if TIER1_GUIDED:
    # multi_topic_assignments.csv — one row per doc
    multi_path = RUN_DIR / "multi_topic_assignments.csv"
    multi_df.to_csv(multi_path, index=False)

    # aspect_mapping.json — topic_id → (aspect, similarity)
    asp_map_path = RUN_DIR / "aspect_mapping.json"
    with open(asp_map_path, "w") as f:
        json.dump(
            {str(k): {"aspect": v[0], "similarity": v[1]}
             for k, v in aspect_mapping.items()},
            f, indent=2,
        )

    # topic_info_multi.json — per-topic with aspect + keywords_main + keywords_mmr
    topic_info_path = RUN_DIR / "topic_info_multi.json"
    with open(topic_info_path, "w") as f:
        json.dump(topic_info_multi, f, indent=2, default=str)

    print("\nTier 1 sidecar artifacts:")
    print(f"  {multi_path.name}")
    print(f"  {asp_map_path.name}")
    print(f"  {topic_info_path.name}")
```

This preserves the original artifact layout (existing PDF report generator keeps working) and adds three new files for the dashboard / Tier 2 pipeline.

---

## Run ID convention for comparisons

To preserve existing runs and produce apples-to-apples comparisons, use:

| RUN_ID | DATASET | TIER1_GUIDED | Compares to |
|---|---|---|---|
| `nb_011` | augmented | False | (already exists — leave untouched) |
| `nb_011_real` | real | False | (already exists — leave untouched) |
| `nb_011_real_v2` | real_filtered | False | (already exists — production baseline) |
| `nb_014_aug` | augmented | True | `nb_011` |
| `nb_014_real` | real | True | `nb_011_real` |
| **`nb_014_real_filtered`** | **real_filtered** | **True** | **`nb_011_real_v2` ← key comparison for production decision** |

---

## Validation gates before committing Tier 1

Three checks, in order:

1. **`audit_aspect_coverage` output is sane.** Every aspect has >50 hits on `real_filtered`, OR the user has consciously expanded/dropped weak aspects. Aspects with <50 hits printed by the toggle cell.
2. **Native-speaker spot-check.** Print 30 documents per aspect from `multi_df` (filter by primary_topic, then look up its aspect via `aspect_mapping`). Have a Cebuano/Tagalog speaker confirm assignments are reasonable. This is the gate for committing thresholds.
3. **Existing PDF report still generates.** Run the existing report generator against `nb_014_real_filtered` to confirm Tier 1 sidecar files don't break the original artifact layout.

---

## Expected results when running `nb_014_real_filtered` vs `nb_011_real_v2`

- **Headline metrics nearly identical.** Embedding coherence (~0.65), diversity (~0.98), outlier rate (~52%). Tier 1 does NOT move these. Don't sell it as a metric improvement.
- **Aspect coverage:** 60–75% of topics map to a known aspect; 25–40% surface as `emergent`.
- **Multi-topic rate:** 15–30% of docs get a secondary label. Above 40% → thresholds too loose. Below 10% → too strict.
- **Run 013's duplicate "good teaching" issue (Topics 0 + 1) likely persists.** Tier 1 cannot fix that — it's a Tier 2 (LLM labeler) job.

---

## Effort estimate

- `lib.py` additions: 2–3 hours including testing
- Notebook integration: 1 hour
- Aspect taxonomy review with native speaker: 2–4 hours
- First validation run + threshold tuning: half a day
- Comparison report (`nb_014_real_filtered` vs `nb_011_real_v2`): half a day

**Total: ~2 working days** assuming no major taxonomy rewrites.

---

## How a future Claude session should use this document

1. **Read this entire file before suggesting any change.** The user has agreed to this exact plan; do not propose alternatives unless the user specifically asks.
2. **Treat the code blocks as canonical.** They are positioned exactly (line numbers, sibling functions, before/after relationships). Do not silently rewrite them.
3. **The user's existing baselines are sacred.** Do not propose modifying `build_bertopic`, `extract_topic_info`, `compute_metrics`, `save_run_artifacts`, the regex patterns, the stop-word list, or the cleaning functions. Sibling additions only.
4. **Tier 1 is configuration + helpers, not fine-tuning.** If the user starts asking about LoRA, QLoRA, encoder adaptation, or local LLMs, that is Tier 2/3/4 territory — flag it explicitly so the user knows they're moving past Tier 1.
5. **The user is the project owner.** Native-speaker review of the aspect taxonomy is their responsibility, not Claude's. Do not invent or "improve" the seed lists without being asked.

---

*Document version 1.0. Plan agreed in conversation prior to this file being written. If the user's repo state has changed (different `lib.py` line numbers, different existing functions, different run IDs), confirm with the user before applying these patches.*

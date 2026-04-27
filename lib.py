"""
Helper module for the topic-modeling-notebook experiments.

Mirrors the original `topic-modeling.faculytics/src/` package as a single flat
module so the two notebooks (topic_modeling.ipynb, grid_search.ipynb) can import
exact reproducibility-critical code (regex patterns, multilingual stop words,
BERTopic factory, evaluation metrics, artifact saver) without the user having
to scroll past 200+ lines of boilerplate in the notebook itself.

Source mapping (all original paths are relative to ../topic-modeling.faculytics/):
  src/preprocess.py    → clean_text, clean_dataset, clean_dataset_with_indices,
                          clean_text_with_reason (NEW), regex patterns
  src/topic_model.py   → MULTILINGUAL_STOP_WORDS, build_bertopic, extract_topic_info,
                          get_topic_assignments
  src/embed.py         → embed_texts, get_cache_path
  src/evaluate.py      → compute_npmi_coherence, compute_topic_diversity,
                          compute_outlier_ratio, compute_silhouette,
                          compute_embedding_coherence, compute_metrics
  scripts/run_eval.py  → save_run_artifacts (derived; same artifact layout)

No Discord wiring is included.
"""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ─── Paths & device ──────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).parent.resolve()
DATA_DIR = REPO_ROOT / "data"
RUNS_DIR = REPO_ROOT / "runs"
FIGURES_DIR = REPO_ROOT / "figures"

# Ensure directories exist (safe to call repeatedly)
DATA_DIR.mkdir(exist_ok=True)
RUNS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

LABSE_MODEL = "sentence-transformers/LaBSE"


def _detect_device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


DEVICE = _detect_device()

# Dataset paths (mirror the original config.py naming)
UC_DATASET_PATH = DATA_DIR / "uc_dataset_20krows1.csv"
AUGMENTED_DATASET_PATH = DATA_DIR / "feedback_augmented_v1.json"
UC_FILTERED_DATASET_PATH = DATA_DIR / "uc_dataset_filtered.json"


# ─── Preprocessing ───────────────────────────────────────────────────────────
# Verbatim from src/preprocess.py:12-19. Do not modify — these are tuned for
# Cebuano/Tagalog/English/code-switched feedback and changing them invalidates
# comparison with the original CLI runs.

EXCEL_ARTIFACT_PATTERN = re.compile(r"^#(NAME|VALUE|REF|DIV/0|NULL|NUM|N/A)\??$", re.IGNORECASE)
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
LAUGHTER_PATTERN = re.compile(
    r"\b(ha){2,}\b|\b(he){2,}\b|\b(hi){2,}\b|\blol+\b|\blmao+\b|\brofl+\b",
    re.IGNORECASE,
)
REPEATED_CHAR_PATTERN = re.compile(r"(.)\1{2,}")
PUNCTUATION_SPAM_PATTERN = re.compile(r"([!?.]){3,}")
BROKEN_EMOJI_PATTERN = re.compile("\ufffd+")
KEYBOARD_MASH_PATTERN = re.compile(r"^[asdfghjklqwertyuiopzxcvbnm]{5,}$", re.IGNORECASE)
WHITESPACE_PATTERN = re.compile(r"\s+")


def clean_text(text: str) -> Optional[str]:
    """Clean a single feedback entry. Returns cleaned text or None to drop."""
    if not isinstance(text, str):
        return None

    text = text.strip()
    if not text:
        return None
    if EXCEL_ARTIFACT_PATTERN.match(text):
        return None

    text = URL_PATTERN.sub("", text)
    text = BROKEN_EMOJI_PATTERN.sub("", text)
    text = LAUGHTER_PATTERN.sub("", text)
    text = REPEATED_CHAR_PATTERN.sub(r"\1", text)
    text = PUNCTUATION_SPAM_PATTERN.sub(r"\1", text)
    text = WHITESPACE_PATTERN.sub(" ", text).strip()

    if not text:
        return None

    text_no_space = text.replace(" ", "").lower()
    if len(text_no_space) >= 5 and KEYBOARD_MASH_PATTERN.match(text_no_space):
        vowels = sum(1 for c in text_no_space if c in "aeiou")
        if vowels / len(text_no_space) < 0.15:
            return None

    if len(text.split()) < 3:
        return None

    return text


def clean_text_with_reason(text: str) -> tuple[Optional[str], str]:
    """
    Same cleaning logic as clean_text, but returns (cleaned, reason) so the
    notebook can show *why* entries are dropped. Reasons:

      kept                — passed all checks
      not_string          — input was not a str (None / NaN / etc.)
      empty               — empty before any cleaning
      excel_artifact      — matches #NAME?, #VALUE!, etc.
      empty_after_clean   — became empty after URL/emoji/laughter/whitespace stripping
      keyboard_mash       — looks like random keypress with low vowel ratio
      too_short           — fewer than 3 words after cleaning
    """
    if not isinstance(text, str):
        return None, "not_string"

    raw = text.strip()
    if not raw:
        return None, "empty"
    if EXCEL_ARTIFACT_PATTERN.match(raw):
        return None, "excel_artifact"

    cleaned = URL_PATTERN.sub("", raw)
    cleaned = BROKEN_EMOJI_PATTERN.sub("", cleaned)
    cleaned = LAUGHTER_PATTERN.sub("", cleaned)
    cleaned = REPEATED_CHAR_PATTERN.sub(r"\1", cleaned)
    cleaned = PUNCTUATION_SPAM_PATTERN.sub(r"\1", cleaned)
    cleaned = WHITESPACE_PATTERN.sub(" ", cleaned).strip()

    if not cleaned:
        return None, "empty_after_clean"

    text_no_space = cleaned.replace(" ", "").lower()
    if len(text_no_space) >= 5 and KEYBOARD_MASH_PATTERN.match(text_no_space):
        vowels = sum(1 for c in text_no_space if c in "aeiou")
        if vowels / len(text_no_space) < 0.15:
            return None, "keyboard_mash"

    if len(cleaned.split()) < 3:
        return None, "too_short"

    return cleaned, "kept"


def clean_dataset(texts: list[str]) -> list[str]:
    """Clean a list of texts; drop entries that fail any check."""
    cleaned = [c for c in (clean_text(t) for t in texts) if c is not None]
    drop = len(texts) - len(cleaned)
    pct = (drop / len(texts) * 100) if texts else 0.0
    logger.info(f"Preprocessing: {len(texts)} → {len(cleaned)} ({drop} dropped, {pct:.1f}%)")
    return cleaned


def clean_dataset_with_indices(texts: list[str]) -> tuple[list[str], list[int]]:
    """Clean and also return original indices of kept entries (for label alignment)."""
    cleaned: list[str] = []
    indices: list[int] = []
    for i, t in enumerate(texts):
        result = clean_text(t)
        if result is not None:
            cleaned.append(result)
            indices.append(i)
    logger.info(f"Preprocessing: {len(texts)} → {len(cleaned)} ({len(texts) - len(cleaned)} dropped)")
    return cleaned, indices


# ─── Multilingual stop words ─────────────────────────────────────────────────
# Verbatim from src/topic_model.py:70-108. Do not paraphrase or "tidy up" —
# entries like "propesor"/"estudyante" exist because CLI Run 011 found a
# generic "propesor/estudyante" cluster that destroyed signal.

MULTILINGUAL_STOP_WORDS = list({
    # Role/title words — appear in almost all feedback, no topic signal
    "propesor", "estudyante", "guro", "magaaral", "maestra", "maestro",
    "students", "student", "maam", "sir", "professor", "teacher", "instructor",
    "faculty", "teacher", "atty", "miss",
    # English stop words (core subset)
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can", "that", "this",
    "these", "those", "it", "its", "i", "me", "my", "we", "our", "you",
    "your", "he", "she", "they", "his", "her", "their", "as", "not",
    "also", "so", "if", "than", "then", "when", "what", "how", "all",
    "very", "just", "more", "about", "up", "out", "no", "her", "him",
    # Cebuano function words + possessives + filler
    "ang", "nga", "sa", "ni", "si", "ug", "og", "kay", "ba", "na",
    "man", "lang", "ra", "jud", "gyud", "ko", "mo", "mi", "siya",
    "niya", "nako", "imo", "iya", "ato", "nato", "namo", "kamo",
    "sila", "nila", "kang", "kanang", "kanila", "dili", "wala",
    "naa", "adto", "diri", "didto", "pero",
    # Cebuano possessives (iyang/kanyang = his/her, among/atong = our)
    "iyang", "kanyang", "among", "atong", "ilang", "inyong", "among",
    # High-frequency Cebuano filler (kaayo=very, mao=that/it, bawat=every)
    "kaayo", "mao", "bawat", "aralin", "klase", "gawain",
    # Generic action verbs (high-freq across all topics, zero discrimination)
    "nagbibigay", "naguse", "nagtatakda", "naggagamit", "ginagamit",
    "nagbibigay ng", "ginagamit ng",
    # Generic English filler found in CLI Run 008 Topic 13
    "really", "much", "am", "way", "feel", "truly",
    # Generic time/context words
    "every", "during", "specific", "even",
    # Tagalog function words
    "ng", "mga", "ay", "nang", "rin", "din", "po", "ho", "yung",
    "kasi", "naman", "lang", "pa", "pag", "kung", "dahil", "para",
    "hindi", "at", "o", "ni", "niya", "siya", "namin", "natin",
    "nila", "ito", "iyon", "yon", "dito", "doon",
})


# ─── Embedding ───────────────────────────────────────────────────────────────


def get_cache_path(dataset_name: str, cache_dir: Path = DATA_DIR) -> Path:
    """Cache path for a given dataset name."""
    if dataset_name:
        return cache_dir / f"embeddings_cache_{dataset_name}.npy"
    return cache_dir / "embeddings_cache.npy"


def embed_texts(
    texts: list[str],
    embed_model=None,
    batch_size: int = 64,
    dataset_name: str = "",
    force: bool = False,
    cache_dir: Path = DATA_DIR,
) -> np.ndarray:
    """
    Generate LaBSE embeddings, with NPY caching per dataset.

    Differs from src/embed.py only in that the SentenceTransformer instance
    can be passed in (avoiding a second LaBSE load when the notebook already
    has one). If `embed_model` is None, LaBSE is loaded lazily.

    Returns: np.ndarray of shape (len(texts), 768), L2-normalized.
    """
    cache_path = get_cache_path(dataset_name, cache_dir)

    if not force and cache_path.exists():
        logger.info(f"Loading cached embeddings from {cache_path}")
        embeddings = np.load(cache_path)
        if embeddings.shape[0] == len(texts):
            logger.info(f"Loaded {embeddings.shape[0]} embeddings from cache")
            return embeddings
        logger.warning(
            f"Cache size mismatch ({embeddings.shape[0]} vs {len(texts)}), regenerating..."
        )

    if embed_model is None:
        from sentence_transformers import SentenceTransformer

        logger.info(f"Loading LaBSE model on {DEVICE}...")
        embed_model = SentenceTransformer(LABSE_MODEL, device=DEVICE)

    logger.info(f"Encoding {len(texts)} texts (batch_size={batch_size})...")
    embeddings = embed_model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    logger.info(f"Saving embeddings to {cache_path}")
    np.save(cache_path, embeddings)
    return embeddings


# ─── BERTopic factory ────────────────────────────────────────────────────────


def build_bertopic(params: dict[str, Any], embedding_model=None):
    """
    Construct and configure a BERTopic instance with the same UMAP/HDBSCAN/
    CountVectorizer/KeyBERTInspired wiring as src/topic_model.py:36-143.

    The caller still controls `fit_transform` (so the notebook can re-seed
    np.random immediately before fitting). Returns an unfit BERTopic.

    Params keys (all optional, defaults shown):
        min_topic_size      = 15
        nr_topics           = None  (None = auto via HDBSCAN)
        umap_n_neighbors    = 15
        umap_n_components   = 5
        hdbscan_min_samples = 5
        use_keybert         = True  (False → c-TF-IDF only)
    """
    from bertopic import BERTopic
    from bertopic.representation import KeyBERTInspired
    from hdbscan import HDBSCAN
    from sklearn.feature_extraction.text import CountVectorizer
    from umap import UMAP

    min_topic_size = params.get("min_topic_size", 15)
    nr_topics = params.get("nr_topics", None)
    umap_n_neighbors = params.get("umap_n_neighbors", 15)
    umap_n_components = params.get("umap_n_components", 5)
    hdbscan_min_samples = params.get("hdbscan_min_samples", 5)
    use_keybert = params.get("use_keybert", True)

    logger.info(
        f"BERTopic params: min_topic_size={min_topic_size}, nr_topics={nr_topics}, "
        f"umap_n_neighbors={umap_n_neighbors}, umap_n_components={umap_n_components}, "
        f"hdbscan_min_samples={hdbscan_min_samples}, use_keybert={use_keybert}"
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

    representation_model = KeyBERTInspired() if use_keybert else None

    if use_keybert and embedding_model is None:
        from sentence_transformers import SentenceTransformer

        embedding_model = SentenceTransformer(LABSE_MODEL, device=DEVICE)

    topic_model = BERTopic(
        embedding_model=embedding_model if use_keybert else None,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        representation_model=representation_model,
        nr_topics=nr_topics,
        verbose=True,
    )
    return topic_model


def extract_topic_info(model) -> list[dict]:
    """List topics with id, label, top-10 keywords, and doc count (excludes -1)."""
    topic_info = model.get_topic_info()
    results = []
    for _, row in topic_info.iterrows():
        topic_id = row["Topic"]
        if topic_id == -1:
            continue
        topic_words = model.get_topic(topic_id)
        keywords = [w for w, _ in topic_words[:10]] if topic_words else []
        results.append({
            "topic_id": int(topic_id),
            "label": row.get("Name", f"Topic_{topic_id}"),
            "keywords": keywords,
            "doc_count": int(row["Count"]),
        })
    return results


def get_topic_assignments(model, texts: list[str], embeddings: np.ndarray) -> list[int]:
    """Topic id per document (-1 = outlier)."""
    topics, _ = model.transform(texts, embeddings=embeddings)
    return list(topics)


# ─── Evaluation metrics ──────────────────────────────────────────────────────


def compute_npmi_coherence(model, texts: list[str], top_n: int = 10) -> float:
    """
    NPMI coherence (Lau et al. 2014, window_size=10). Reference only on
    multilingual data — semantically equivalent words across languages never
    co-occur, so NPMI systematically underestimates coherence (Hoyle et al. 2021).
    Use embedding_coherence as the real metric.
    """
    from gensim.corpora import Dictionary
    from gensim.models.coherencemodel import CoherenceModel

    tokenized = [t.lower().split() for t in texts]
    dictionary = Dictionary(tokenized)

    topic_words = []
    for topic_id in model.get_topic_info()["Topic"]:
        if topic_id == -1:
            continue
        words = model.get_topic(topic_id)
        if words:
            topic_words.append([w for w, _ in words[:top_n]])

    if not topic_words:
        logger.warning("No topics for NPMI calculation")
        return 0.0

    try:
        cm = CoherenceModel(
            topics=topic_words,
            texts=tokenized,
            dictionary=dictionary,
            coherence="c_npmi",
            window_size=10,
        )
        return cm.get_coherence()
    except Exception as e:
        logger.warning(f"NPMI calc failed: {e}")
        return 0.0


def compute_topic_diversity(model, top_n: int = 10) -> float:
    """Unique words / total words across top-N keywords of all non-outlier topics."""
    all_words: list[str] = []
    for topic_id in model.get_topic_info()["Topic"]:
        if topic_id == -1:
            continue
        words = model.get_topic(topic_id)
        if words:
            all_words.extend([w for w, _ in words[:top_n]])
    if not all_words:
        return 0.0
    return len(set(all_words)) / len(all_words)


def compute_outlier_ratio(model) -> float:
    """Fraction of documents in topic -1."""
    topic_info = model.get_topic_info()
    outlier_row = topic_info[topic_info["Topic"] == -1]
    if outlier_row.empty:
        return 0.0
    outlier_count = outlier_row["Count"].values[0]
    total_count = topic_info["Count"].sum()
    return outlier_count / total_count


def compute_silhouette(topics: list[int], embeddings: np.ndarray) -> float:
    """Silhouette score (cosine), excluding outliers."""
    from sklearn.metrics import silhouette_score

    valid_mask = np.array(topics) != -1
    valid_topics = np.array(topics)[valid_mask]
    valid_embeddings = embeddings[valid_mask]

    if len(set(valid_topics)) < 2 or len(valid_topics) < 10:
        logger.warning("Not enough valid clusters for silhouette score")
        return 0.0

    try:
        return silhouette_score(valid_embeddings, valid_topics, metric="cosine")
    except Exception as e:
        logger.warning(f"Silhouette failed: {e}")
        return 0.0


def compute_embedding_coherence(model, embed_model=None, top_n: int = 10) -> float:
    """
    Average pairwise cosine similarity between top-N keyword embeddings per topic.
    Language-agnostic alternative to NPMI. Manual normalize+dot loop matches
    src/evaluate.py:124-159 (NOT sklearn.cosine_similarity — exact reproducibility).
    Target > 0.5.
    """
    if embed_model is None:
        return 0.0

    scores = []
    for topic_id in model.get_topic_info()["Topic"]:
        if topic_id == -1:
            continue
        words = model.get_topic(topic_id)
        if not words or len(words) < 2:
            continue
        keywords = [w for w, _ in words[:top_n]]
        try:
            vecs = embed_model.encode(
                keywords, show_progress_bar=False, normalize_embeddings=True
            )
            n = len(vecs)
            sim_sum = 0.0
            count = 0
            for i in range(n):
                for j in range(i + 1, n):
                    sim_sum += float(np.dot(vecs[i], vecs[j]))
                    count += 1
            if count > 0:
                scores.append(sim_sum / count)
        except Exception:
            continue

    return round(float(np.mean(scores)), 4) if scores else 0.0


def compute_per_topic_coherence(model, embed_model, top_n: int = 10) -> dict[int, float]:
    """
    Per-topic embedding coherence — useful for the topic-quality table to
    identify weak topics. Same formula as compute_embedding_coherence but
    per-topic instead of averaged.
    """
    if embed_model is None:
        return {}

    out: dict[int, float] = {}
    for topic_id in model.get_topic_info()["Topic"]:
        if topic_id == -1:
            continue
        words = model.get_topic(topic_id)
        if not words or len(words) < 2:
            out[int(topic_id)] = 0.0
            continue
        keywords = [w for w, _ in words[:top_n]]
        try:
            vecs = embed_model.encode(
                keywords, show_progress_bar=False, normalize_embeddings=True
            )
            n = len(vecs)
            sim_sum = 0.0
            count = 0
            for i in range(n):
                for j in range(i + 1, n):
                    sim_sum += float(np.dot(vecs[i], vecs[j]))
                    count += 1
            out[int(topic_id)] = round(sim_sum / count, 4) if count > 0 else 0.0
        except Exception:
            out[int(topic_id)] = 0.0
    return out


def compute_metrics(
    model,
    topics: list[int],
    texts: list[str],
    embeddings: np.ndarray,
    embed_model=None,
) -> dict[str, Any]:
    """Aggregate the 5 metrics + run-level counts. Same shape as src/evaluate.py."""
    logger.info("Computing evaluation metrics...")

    num_topics = len(set(topics)) - (1 if -1 in topics else 0)

    metrics = {
        "embedding_coherence": compute_embedding_coherence(model, embed_model=embed_model),
        "npmi_coherence": round(compute_npmi_coherence(model, texts), 4),
        "topic_diversity": round(compute_topic_diversity(model), 4),
        "outlier_ratio": round(compute_outlier_ratio(model), 4),
        "num_topics": num_topics,
        "silhouette_score": round(compute_silhouette(topics, embeddings), 4),
        "total_documents": len(texts),
        "outlier_count": sum(1 for t in topics if t == -1),
    }

    logger.info(
        f"Metrics: emb_coh={metrics['embedding_coherence']}, "
        f"NPMI={metrics['npmi_coherence']}, "
        f"diversity={metrics['topic_diversity']}, "
        f"outliers={metrics['outlier_ratio']:.1%}, "
        f"topics={metrics['num_topics']}"
    )
    return metrics


# ─── Artifact saving ─────────────────────────────────────────────────────────


def save_run_artifacts(
    run_dir: Path,
    params: dict,
    metrics: dict,
    topic_info: list[dict],
    model=None,
    n_texts: int | None = None,
) -> None:
    """
    Persist a run to disk. Same artifact layout as scripts/run_eval.py so the
    original PDF report generator (`generate_report.py`) reads these as-is.

    Writes: params.json, metrics.json, topics.md, info.json, bertopic_model/
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "params.json", "w") as f:
        json.dump(params, f, indent=2, default=str)

    with open(run_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    with open(run_dir / "topics.md", "w") as f:
        f.write(f"# Topics for {run_dir.name}\n\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n\n")
        for topic in topic_info:
            f.write(f"## Topic {topic['topic_id']}: {topic['label']}\n")
            f.write(f"- Documents: {topic['doc_count']}\n")
            f.write(f"- Keywords: {', '.join(topic['keywords'])}\n\n")

    with open(run_dir / "info.json", "w") as f:
        info_payload: dict[str, Any] = {"timestamp": datetime.now().isoformat()}
        if n_texts is not None:
            info_payload["n_texts"] = n_texts
        json.dump(info_payload, f, indent=2)

    if model is not None:
        try:
            model.save(
                str(run_dir / "bertopic_model"),
                serialization="safetensors",
                save_ctfidf=True,
                save_embedding_model=False,
            )
            logger.info(f"BERTopic model saved → {run_dir / 'bertopic_model'}")
        except Exception as e:
            logger.warning(f"Could not persist BERTopic model: {e}")

    logger.info(f"Artifacts saved to {run_dir}")

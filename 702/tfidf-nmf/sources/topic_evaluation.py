"""
Utilities to evaluate the quality of NMF topics.

Metrics computed per topic configuration (k):
  - topic coherence (c_v, c_npmi via gensim).

These metrics are meant to be used with the W / H matrices
produced by `sources.topic_modeling_nmf`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.auto import tqdm
from joblib import Parallel, delayed

# Import N_JOBS from params
import sys
import os
# Add parent directory to path to import params
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from params import N_JOBS

try:
    # gensim is used only for coherence metrics
    from gensim.corpora import Dictionary
    from gensim.models import CoherenceModel

    _GENSIM_AVAILABLE = True
except Exception:  # ImportError and less common cases
    Dictionary = None  # type: ignore
    CoherenceModel = None  # type: ignore
    _GENSIM_AVAILABLE = False


TokenizedTexts = List[List[str]]
TopicsWords = List[List[str]]


def prepare_corpus_for_coherence(model_texts: Sequence[str]) -> tuple[TokenizedTexts, "Dictionary", list]:
    """
    Prepare internal structures needed to compute topic coherence.

    Args:
        model_texts: lemmatized texts (after spaCy), one string per document.

    Returns:
        tokenized_texts: list of tokenized documents (lists of str)
        dictionary: gensim dictionary
        corpus: BoW corpus
    """
    if not _GENSIM_AVAILABLE:
        raise RuntimeError(
            "The 'gensim' module is not available. "
            "Install it to enable coherence metrics "
            "(e.g. pip install gensim)."
        )

    print("\nPreparing corpus for coherence metrics...")

    # Texts are already lemmatized and filtered: a simple split is enough
    tokenized_texts: TokenizedTexts = []
    for text in tqdm(
        model_texts,
        desc="Tokenization (coherence)",
        unit="doc",
    ):
        tokens = [tok for tok in (text or "").split() if tok]
        tokenized_texts.append(tokens)

    print(f"  - Tokenized {len(tokenized_texts)} documents")
    print(f"  - Average tokens per document: {sum(len(t) for t in tokenized_texts) / len(tokenized_texts) if tokenized_texts else 0:.1f}")
    
    dictionary: Dictionary = Dictionary(tokenized_texts)
    print(f"  - Dictionary size: {len(dictionary)} unique tokens")
    
    corpus = [
        dictionary.doc2bow(doc)
        for doc in tqdm(
            tokenized_texts,
            desc="BoW construction (coherence)",
            unit="doc",
        )
    ]
    
    print(f"  - BoW corpus created: {len(corpus)} documents")

    return tokenized_texts, dictionary, corpus


def get_top_words_per_topic(
    H: np.ndarray,
    feature_names: np.ndarray,
    top_n: int,
) -> TopicsWords:
    """
    Extract the list of top_n words for each topic.
    """
    n_topics, n_terms = H.shape
    top_n = min(top_n, n_terms)

    topics_words: TopicsWords = []
    for topic_idx in range(n_topics):
        topic = H[topic_idx]
        top_indices = topic.argsort()[-top_n:][::-1]
        top_words = [str(feature_names[i]) for i in top_indices]
        topics_words.append(top_words)

    return topics_words


def _compute_single_coherence_metric(
    metric_name: str,
    topics_words: TopicsWords,
    tokenized_texts: TokenizedTexts,
    dictionary: "Dictionary",
    corpus: list,
) -> Tuple[str, float]:
    """
    Helper function to compute a single coherence metric for parallel execution.
    
    Args:
        metric_name: name of the coherence metric ('c_v' or 'c_npmi')
        topics_words: list of top words per topic
        tokenized_texts: tokenized corpus
        dictionary: gensim dictionary
        corpus: BoW corpus
    
    Returns:
        tuple of (metric_name, score)
    """
    try:
        print(f"  Computing {metric_name}...")
        print(f"    - Number of topics: {len(topics_words)}")
        print(f"    - Number of documents: {len(tokenized_texts)}")
        print(f"    - Dictionary size: {len(dictionary) if dictionary else 0}")
        
        cm = CoherenceModel(
            topics=topics_words,
            texts=tokenized_texts,
            dictionary=dictionary,
            coherence=metric_name,
        )
        score = float(cm.get_coherence())
        print(f"  ✓ {metric_name} = {score:.4f}")
    except Exception as e:
        # Log the error with full details
        print(f"  ⚠️ ERROR computing {metric_name}: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        score = float("nan")
    
    return metric_name, score


def compute_coherence_scores(
    topics_words: TopicsWords,
    tokenized_texts: TokenizedTexts,
    dictionary: "Dictionary",
    corpus: list,
    metrics: Sequence[str],
) -> Dict[str, float]:
    """
    Compute global coherence scores for a list of topics (in parallel).

    Only the following metrics are supported (case‑insensitive):
      - 'c_v' or 'cv'
      - 'c_npmi' or 'npmi'

    The returned dictionary always uses normalized keys:
      - 'c_v', 'c_npmi'
    """
    scores: Dict[str, float] = {}

    # Normalize requested metrics and keep only supported ones
    metrics_norm = {m.lower() for m in metrics}
    want_c_v = any(m in {"c_v", "cv"} for m in metrics_norm)
    want_c_npmi = any(m in {"c_npmi", "npmi"} for m in metrics_norm)

    if not (want_c_v or want_c_npmi):
        return scores

    # If gensim is not available or we have no topics, return NaN for requested metrics
    if not _GENSIM_AVAILABLE or not topics_words:
        if want_c_v:
            scores["c_v"] = float("nan")
        if want_c_npmi:
            scores["c_npmi"] = float("nan")
        return scores

    # Compute requested metrics sequentially (gensim handles its own multiprocessing)
    requested = []
    if want_c_v:
        requested.append("c_v")
    if want_c_npmi:
        requested.append("c_npmi")

    print(f"  Computing {len(requested)} coherence metric(s): {requested}")
    print(f"  (gensim will use its own multiprocessing)")
    
    # Compute metrics sequentially - gensim will parallelize internally where needed
    for metric_name in requested:
        _, score = _compute_single_coherence_metric(
            metric_name, topics_words, tokenized_texts, dictionary, corpus
        )
        scores[metric_name] = score

    return scores


def compute_topic_diversity(topics_words: TopicsWords) -> Dict[str, float]:
    """
    Compute a simple lexical diversity metric for topics.

    - unique_words: number of distinct words across all topics.
    - diversity_ratio: unique_words / (total number of positions).

    diversity_ratio is between 0 and 1:
      - close to 1: little lexical overlap between topics;
      - close to 0: topics share many words.
    """
    if not topics_words:
        return {"unique_words": 0.0, "diversity_ratio": float("nan")}

    all_words = [w for topic in topics_words for w in topic]
    num_unique = len(set(all_words))
    max_possible = len(all_words)

    diversity_ratio = float(num_unique) / max_possible if max_possible > 0 else float("nan")

    return {
        "unique_words": float(num_unique),
        "diversity_ratio": diversity_ratio,
    }


def compute_w_entropy_stats(W: np.ndarray) -> Dict[str, float]:
    """
    Compute (normalized) entropy statistics on rows of W.

    - Each row of W is normalized into a probability distribution p over k topics.
    - Entropy is normalized by log(k) so it lies between 0 and 1.
      * entropy close to 0: very concentrated distribution (one dominant topic).
      * entropy close to 1: uniform distribution (documents less specialized).
    """
    if W.size == 0:
        return {
            "entropy_mean": float("nan"),
            "entropy_median": float("nan"),
            "entropy_min": float("nan"),
            "entropy_max": float("nan"),
            "entropy_std": float("nan"),
        }

    # Row‑wise normalization
    row_sums = W.sum(axis=1, keepdims=True)
    k = W.shape[1]

    # For zero rows, force a uniform distribution
    with np.errstate(divide="ignore", invalid="ignore"):
        P = np.divide(W, row_sums, out=np.zeros_like(W), where=row_sums != 0)

    zero_row_mask = (row_sums.squeeze(-1) == 0)
    if np.any(zero_row_mask):
        P[zero_row_mask, :] = 1.0 / float(k) if k > 0 else 0.0

    # Shannon entropy
    eps = 1e-12
    P_safe = np.clip(P, eps, 1.0)
    ent = -(P_safe * np.log(P_safe)).sum(axis=1)

    # Normalization by log(k)
    if k > 1:
        ent_norm = ent / np.log(float(k))
    else:
        ent_norm = ent

    return {
        "entropy_mean": float(np.mean(ent_norm)),
        "entropy_median": float(np.median(ent_norm)),
        "entropy_min": float(np.min(ent_norm)),
        "entropy_max": float(np.max(ent_norm)),
        "entropy_std": float(np.std(ent_norm)),
    }


def compute_topic_cosine_stats(W: np.ndarray) -> Dict[str, float]:
    """
    Compute cosine similarity statistics between columns of W.

    Each column of W is seen as a topic profile over documents.
    We compute cosine similarity for all pairs of columns.
    """
    n_docs, n_topics = W.shape
    if n_topics < 2:
        return {
            "cosine_mean": float("nan"),
            "cosine_min": float("nan"),
            "cosine_max": float("nan"),
        }

    # Cosine similarity between column vectors
    sim_matrix = cosine_similarity(W.T)
    # Keep only the strictly upper triangular part (pairs i < j)
    iu = np.triu_indices_from(sim_matrix, k=1)
    sims = sim_matrix[iu]

    if sims.size == 0:
        return {
            "cosine_mean": float("nan"),
            "cosine_min": float("nan"),
            "cosine_max": float("nan"),
        }

    return {
        "cosine_mean": float(np.mean(sims)),
        "cosine_min": float(np.min(sims)),
        "cosine_max": float(np.max(sims)),
    }


def _evaluate_single_config(
    n_topics: int,
    W: np.ndarray,
    H: np.ndarray,
    feature_names: np.ndarray,
    top_n_words: int,
    coherence_metrics: Sequence[str],
    tokenized_texts: TokenizedTexts,
    dictionary: "Dictionary",
    corpus: list,
) -> Tuple[int, Dict[str, object]]:
    """
    Helper function to evaluate a single topic configuration for parallel execution.
    
    Args:
        n_topics: number of topics in this configuration
        W: documents × topics matrix
        H: topics × words matrix
        feature_names: TF-IDF feature names
        top_n_words: number of top words to extract per topic
        coherence_metrics: list of coherence metrics to compute
        tokenized_texts: tokenized corpus for coherence
        dictionary: gensim dictionary
        corpus: BoW corpus
    
    Returns:
        tuple of (n_topics, metrics_dict)
    """
    # Words per topic
    topics_words = get_top_words_per_topic(H, feature_names, top_n=top_n_words)
    
    print(f"\n  Configuration: {n_topics} topics")
    print(f"    - Extracted {len(topics_words)} topics")
    if topics_words:
        print(f"    - Words per topic: {len(topics_words[0])}")
        print(f"    - Example topic 1 words: {topics_words[0][:5]}")

    # Global topic coherence (only if metrics are requested)
    if coherence_metrics:
        coherence_scores = compute_coherence_scores(
            topics_words=topics_words,
            tokenized_texts=tokenized_texts,
            dictionary=dictionary,
            corpus=corpus,
            metrics=coherence_metrics,
        )
    else:
        coherence_scores = {}

    metrics = {
        "n_topics": int(n_topics),
        "n_docs": int(W.shape[0]),
        "coherence": coherence_scores,
    }
    
    return n_topics, metrics


def evaluate_all_topic_configs(
    model_texts: Sequence[str],
    W_dict: Mapping[int, np.ndarray],
    H_dict: Mapping[int, np.ndarray],
    feature_names: np.ndarray,
    top_n_words: int,
    coherence_metrics: Sequence[str],
) -> Dict[int, Dict[str, object]]:
    """
    Evaluate all NMF topic configurations (k) in parallel.

    This function should be called right after TF‑IDF + NMF, before W_dict
    is re‑assigned to near‑duplicates and before any metadata‑based
    deduplication.

    Returns:
        dict {k -> metrics}, where each entry contains at least:
          - "n_topics"
          - "n_docs"
          - "coherence": dict of coherence scores (c_v, c_npmi)
    """
    if not W_dict or not H_dict:
        return {}

    # Prepare corpus for coherence only if at least one coherence metric is requested
    tokenized_texts: TokenizedTexts
    dictionary: "Dictionary"
    corpus: list

    if coherence_metrics:
        if not _GENSIM_AVAILABLE:
            raise RuntimeError(
                "Coherence metrics (c_v, c_npmi) require the 'gensim' package. "
                "Install it (e.g. pip install gensim) or clear COHERENCE_METRICS in 'params.py' "
                "to disable coherence computation."
            )
        tokenized_texts, dictionary, corpus = prepare_corpus_for_coherence(model_texts)
    else:
        # No coherence metric requested
        tokenized_texts, dictionary, corpus = [], None, []  # type: ignore

    print(
        f"\nEvaluating topic quality for {len(W_dict)} topic configuration(s) sequentially..."
    )
    print("(Gensim will handle multiprocessing internally for coherence calculations)\n")

    # Evaluate all configurations sequentially (gensim will parallelize internally)
    results: Dict[int, Dict[str, object]] = {}
    for n_topics, W in sorted(W_dict.items(), key=lambda x: x[0]):
        if n_topics in H_dict:
            _, metrics = _evaluate_single_config(
                n_topics,
                W,
                H_dict[n_topics],
                feature_names,
                top_n_words,
                coherence_metrics,
                tokenized_texts,
                dictionary,
                corpus,
            )
            results[n_topics] = metrics

    return results


def _flatten_metrics_row(k: int, metrics: Mapping[str, object]) -> Dict[str, object]:
    """
    Flatten the nested metrics structure into a single row.
    """
    row: Dict[str, object] = {"n_topics": int(k)}

    for key, value in metrics.items():
        if key == "n_topics":
            # already carried by argument k
            continue

        if key == "coherence" and isinstance(value, Mapping):
            # Keep all coherence sub‑metrics (typically c_v, c_npmi)
            for sub_key, sub_val in value.items():
                col_name = f"coherence_{sub_key}"
                row[col_name] = sub_val
        elif not isinstance(value, Mapping):
            # Any other simple scalar metric (none in the default pipeline)
            row[key] = value

    return row


def save_topic_quality_summary(
    topic_quality_by_k: Mapping[int, Mapping[str, object]],
    results_root: Path,
    dataset_name: str,
    top_n_words: int,
) -> Path:
    """
    Save a global summary file of metrics per number of topics.

    File created:
      results/<DATASET_NAME>/topic_quality_summary_top{top_n_words}.csv
    """
    if not topic_quality_by_k:
        return results_root / f"topic_quality_summary_top{top_n_words}.csv"

    rows = [
        _flatten_metrics_row(k, metrics)
        for k, metrics in sorted(topic_quality_by_k.items(), key=lambda kv: kv[0])
    ]
    df = pd.DataFrame(rows)

    # Do not keep n_docs in the global summary file
    if "n_docs" in df.columns:
        df = df.drop(columns=["n_docs"])

    results_root = Path(results_root)
    results_root.mkdir(parents=True, exist_ok=True)

    out_path = results_root / f"topic_quality_summary_top{top_n_words}.csv"
    df.to_csv(out_path, index=False)
    return out_path


def save_topic_quality_per_config(
    topic_quality_by_k: Mapping[int, Mapping[str, object]],
    results_root: Path,
    dataset_name: str,
) -> None:
    """
    Save a small metrics file per topic configuration.

    For each k:
      results/<DATASET_NAME>/<k>t/topic_quality_config.csv
    """
    if not topic_quality_by_k:
        return

    base_root = Path(results_root)
    for k, metrics in topic_quality_by_k.items():
        config_dir = base_root / f"{k}t"
        config_dir.mkdir(parents=True, exist_ok=True)

        row = _flatten_metrics_row(k, metrics)
        df = pd.DataFrame([row])
        out_path = config_dir / "topic_quality_config.csv"
        df.to_csv(out_path, index=False)



#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Topic modeling with TF‑IDF + NMF.

Analysis of medical (or other) articles with SOTA lemmatization.
"""

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
import spacy
from typing import List, Tuple, Iterable
import warnings
import html
import re
import sys
import argparse
import pickle
import hashlib
import json
from pathlib import Path
from datasketch import MinHash, MinHashLSH
from tqdm.auto import tqdm
from joblib import Parallel, delayed
import multiprocessing
from .params import (
    SOURCE_TYPE,
    LANGUAGE,
    HTML_FOLDER,
    ISTEX_FOLDER,
    CSV_FILE,
    CSV_DATE_FORMAT,
    CSV_SEPARATOR,
    CSV_ENCODING,
    CSV_GROUPING_COLUMN,
    MIN_CHARS,
    MAX_CHARS,
    TOPIC_LIST,
    N_TOP_WORDS,
    KEEP_POS,
    FAST_LEMMATIZATION,
    TFIDF_MAX_DF,
    TFIDF_MIN_DF,
    TFIDF_MAX_FEATURES,
    TFIDF_SUBLINEAR_TF,
    TFIDF_NORM,
    TFIDF_SMOOTH_IDF,
    GO_REMOVE_NEAR_DUPLICATES,
    LSH_THRESHOLD,
    LSH_NUM_PERM,
    LSH_MAX_TOKENS,
    DATASET_NAME,
    RESULTS_ROOT,
    CACHE_DIR,
    CACHE_ENABLED,
    EVAL_TOP_N_WORDS,
    COHERENCE_METRICS,
    SAVE_TOPIC_LEVEL_METRICS,
    N_JOBS,
)
from .europresse_utils import (
    remove_urls_hashtags_emojis_mentions_emails,
    transform_text,
    extract_metadata_from_soup,
    extract_republication_sources,
)
from .istex_utils import (
    extract_istex_articles,
    format_istex_metadata,
)
from .topic_evaluation import (
    evaluate_all_topic_configs,
    save_topic_quality_summary,
    save_topic_quality_per_config,
)

warnings.filterwarnings("ignore")

# Threshold below which NMF coefficients are treated as zero
EPS_NMF = 1e-4

def find_near_duplicates_lsh(
    texts: List[str],
    threshold: float = LSH_THRESHOLD,
    num_perm: int = LSH_NUM_PERM,
    max_tokens: int = LSH_MAX_TOKENS,
) -> Tuple[List[int], List[int]]:
    """
    Detect near‑duplicate texts using MinHash LSH.

    Args:
        texts: list of cleaned raw texts on which duplicates are detected.
        threshold: Jaccard similarity threshold to consider two texts as near‑identical.
        num_perm: number of permutations for MinHash.
        max_tokens: maximum number of tokens considered per document (from the beginning).

    Returns:
        canonical_indices: list of indices of documents kept as canonical
                           (used to train TF‑IDF + NMF).
        rep_for_doc: list of length len(texts) such that rep_for_doc[i] is
                     the canonical document index representing document i.
    """
    n_docs = len(texts)
    if n_docs == 0:
        return [], []

    print("\n" + "=" * 80)
    print("TEXTUAL SIMILARITY DEDUPLICATION (MinHash LSH)")
    print("=" * 80)
    print(f"  - Number of input documents: {n_docs}")
    print(f"  - Jaccard threshold: {threshold}")
    print(f"  - Number of MinHash permutations: {num_perm}")
    print(f"  - Max tokens per document: {max_tokens}")

    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    canonical_indices: List[int] = []
    rep_for_doc: List[int] = list(range(n_docs))

    for i, text in enumerate(tqdm(texts, desc="LSH deduplication", unit="doc")):

        # Very simple tokenization: take the first max_tokens words
        tokens = (text or "").split()
        if max_tokens is not None and max_tokens > 0:
            tokens = tokens[:max_tokens]

        m = MinHash(num_perm=num_perm)
        for tok in tokens:
            m.update(tok.lower().encode("utf-8", errors="ignore"))

        # Query the LSH index to find already‑seen similar documents
        candidates = lsh.query(m)
        if not candidates:
            # New canonical document
            canonical_indices.append(i)
            rep_for_doc[i] = i
        else:
            # Use the oldest canonical document as representative
            candidate_indices = sorted(int(c) for c in candidates)
            rep_idx = candidate_indices[0]
            # If that candidate is itself a duplicate, follow its canonical representative
            rep_idx = rep_for_doc[rep_idx]
            rep_for_doc[i] = rep_idx

        # Always insert the current document's MinHash to capture similarity chains
        lsh.insert(str(i), m)

    n_canon = len(canonical_indices)
    print(
        f"\n✓ LSH deduplication complete: {n_docs} documents → {n_canon} canonical "
        f"({n_docs - n_canon} near‑duplicates detected)."
    )

    return canonical_indices, rep_for_doc

def load_spacy_model(language: str = "fr"):
    """
    Load the appropriate spaCy model for the given language.

    Args:
        language: language code ('fr' for French, 'en' for English)

    Returns:
        The loaded and configured spaCy model.
    """
    # Select model size based on FAST_LEMMATIZATION parameter
    model_size = "sm" if FAST_LEMMATIZATION else "lg"
    
    models = {
        "fr": f"fr_core_news_{model_size}",
        "en": f"en_core_web_{model_size}"
    }
    
    if language not in models:
        raise ValueError(f"Unsupported language: {language}. Use 'fr' or 'en'.")

    model_name = models[language]
    lang_name = {"fr": "French", "en": "English"}[language]
    mode_desc = "fast/small" if FAST_LEMMATIZATION else "accurate/large"

    print(f"Loading spaCy {lang_name} model ({mode_desc})...")
    try:
        nlp = spacy.load(model_name)
    except OSError:
        # Model not installed: try automatic download
        print(f"Downloading model {model_name} (one‑time operation)...")
        import subprocess

        try:
            # Use the same Python interpreter as the one running this script
            subprocess.run(
                [sys.executable, "-m", "spacy", "download", model_name],
                check=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            raise RuntimeError(
                f"Automatic download of spaCy model '{model_name}' failed.\n"
                f"Please install it manually with:\n"
                f"    {sys.executable} -m spacy download {model_name}\n"
                f"Original error: {e}"
            ) from e

        # Retry loading after installation
        nlp = spacy.load(model_name)

    # Disable only expensive components (parser, ner, etc.),
    # keeping those needed for POS‑tagging and lemmatization.
    components_to_keep = {"tok2vec", "tagger", "morphologizer", "attribute_ruler", "lemmatizer"}
    nlp.disable_pipes(
        [pipe for pipe in nlp.pipe_names if pipe not in components_to_keep]
    )

    print(f"✓ Model {model_name} loaded successfully")
    return nlp


def extract_europresse_articles(
    html_file: str,
    min_chars: int = 500,
    max_chars: int = 200000,
) -> Tuple[List[str], List[BeautifulSoup]]:
    """
    Dedicated extraction for Europresse exports,
    inspired by load_documents(..., source_type='europresse') in the notebook.

    Returns:
      - articles_texts: list of cleaned article bodies,
      - article_soups: list of corresponding BeautifulSoup objects (for metadata).
    """
    print(f"\nExtracting Europresse articles from {html_file}...")

    with open(html_file, "r", encoding="utf-8", errors="xmlcharrefreplace") as f:
        document_europresse = f.read()

    # Decode entities and repair article separators
    document_europresse = html.unescape(document_europresse)
    document_europresse = document_europresse.replace(
        "</article> <article>", "</article><article>"
    )
    documents_europresse = document_europresse.split("</article><article>")

    articles: List[str] = []
    soups: List[BeautifulSoup] = []
    nb_not_occur = 0
    nb_too_short = 0
    nb_too_long = 0

    for d in documents_europresse:
        soup = BeautifulSoup(d, features="html.parser")

        # Remove short \"Lire aussi ...\" paragraphs that contain URLs
        for p in soup.find_all("p"):
            p_text = p.get_text()
            if (
                "Lire aussi" in p_text
                and ("http" in p_text or "https" in p_text)
                and len(p_text) <= 1000
            ):
                p.decompose()

        # If we find a div with class docOcurrContainer, it holds the main text
        if len(soup("div", {"class": "docOcurrContainer"})) > 0:
            # Fix missing end‑of‑paragraph punctuation
            for p in soup.find_all("p"):
                next_char_match = re.search(
                    r"(?<=" + re.escape(p.text) + r")\s*(?:<[^>]*>)*\s*([a-zA-Z])",
                    str(soup),
                )
                if (
                    not p.text.endswith(".")
                    and next_char_match
                    and next_char_match.group(1).isupper()
                ):
                    p.string = p.text + ". "

            # Rebuild a clean soup after modifications
            soup = BeautifulSoup(str(soup), features="html.parser")

            candidate_text = soup("div", {"class": "docOcurrContainer"})[0].get_text()
            length = len(candidate_text or "")
            if min_chars <= length < max_chars:
                candidate_text = remove_urls_hashtags_emojis_mentions_emails(
                    candidate_text
                )
                candidate_text = transform_text(candidate_text)
                articles.append(candidate_text)
                soups.append(soup)
            else:
                if length < min_chars:
                    nb_too_short += 1
                elif length >= max_chars:
                    nb_too_long += 1
        else:
            nb_not_occur += 1

    print(
        f"✓ {len(articles)} Europresse articles extracted "
        f"(with docOcurrContainer present), {nb_not_occur} without main text."
    )
    if nb_too_short or nb_too_long:
        print(
            f"  ({nb_too_short} articles ignored because < {min_chars} characters, "
            f"{nb_too_long} articles ignored because ≥ {max_chars} characters)"
        )
    return articles, soups


def preprocess_and_lemmatize(texts: List[str], nlp, keep_pos: List[str] = ['NOUN', 'PROPN', 'VERB']) -> List[str]:
    """
    Preprocess and lemmatize texts, keeping only selected POS tags.

    Args:
        texts: list of texts to process
        nlp: loaded spaCy model
        keep_pos: POS tags to keep (NOUN, PROPN, VERB)

    Returns:
        list of lemmatized and filtered texts
    """
    total_cores = multiprocessing.cpu_count()
    print(f"\nUsing {N_JOBS} CPU core(s) out of {total_cores} available")
    print(f"Lemmatization and POS filtering {keep_pos}...")
    processed_texts = []
    
    # Truncate texts before processing to avoid overly long documents
    truncated_texts = [text[:1000000] for text in texts]
    
    # Use nlp.pipe with multi-processing for parallel CPU processing
    # n_process controlled by N_JOBS parameter (leaves at least 1 core free by default)
    # batch_size=50 balances memory usage and throughput
    for doc in tqdm(
        nlp.pipe(truncated_texts, batch_size=50, n_process=N_JOBS),
        desc="Lemmatization",
        unit="doc",
        total=len(texts)
    ):
        # Extract lemmas filtered by POS
        lemmas = [
            token.lemma_.lower() 
            for token in doc 
            if token.pos_ in keep_pos 
            and not token.is_stop 
            and not token.is_punct 
            and not token.is_space
            and len(token.lemma_) > 2  # Words longer than 2 characters
            and token.lemma_.isalpha()  # Letters only
        ]
        
        processed_texts.append(' '.join(lemmas))
    
    print(f"✓ {len(processed_texts)} documents processed")
    return processed_texts


def _train_single_nmf(tfidf_matrix, n_topics: int) -> Tuple[int, np.ndarray, np.ndarray, NMF]:
    """
    Helper function to train a single NMF model for parallel execution.
    
    Args:
        tfidf_matrix: TF-IDF matrix (docs × terms)
        n_topics: number of topics for this NMF model
    
    Returns:
        tuple of (n_topics, W, H, nmf_model)
    """
    # Create a writable copy to avoid "WRITEBACKIFCOPY base is read-only" error
    # when joblib passes the matrix to parallel workers
    tfidf_matrix = tfidf_matrix.copy()
    
    nmf_model = NMF(
        n_components=n_topics,
        random_state=42,
        init="nndsvda",
        max_iter=5000,
        alpha_W=0.0,
        alpha_H=0.0,
        l1_ratio=0.0,
        solver="mu",
        beta_loss="frobenius",
    )

    W = nmf_model.fit_transform(tfidf_matrix)
    H = nmf_model.components_

    # Zero‑out tiny values (numerical noise)
    W[W < EPS_NMF] = 0.0
    H[H < EPS_NMF] = 0.0

    return n_topics, W, H, nmf_model


def apply_tfidf_and_multi_nmf(
    texts: List[str],
    topic_list: List[int],
    n_top_words: int = 20,
    max_df: float = 0.95,
    min_df: int = 3,
    max_features: int = 5000,
    sublinear_tf: bool = True,
    norm: str = "l2",
    smooth_idf: bool = True,
) -> Tuple[dict[int, np.ndarray], dict[int, np.ndarray], np.ndarray, dict[int, NMF], TfidfVectorizer]:
    """
    Apply TF‑IDF once, then train an NMF model for each
    n_topics value in topic_list (in parallel).

    Returns:
      - W_matrices: dict {n_topics -> W matrix (docs × topics)}
      - H_matrices: dict {n_topics -> H matrix (topics × words)}
      - feature_names: TF‑IDF feature names
      - nmf_models: dict {n_topics -> NMF model}
      - tfidf_vectorizer: the fitted vectorizer
    """
    print(f"\nApplying TF‑IDF...")

    tfidf_vectorizer = TfidfVectorizer(
        max_df=max_df,
        min_df=min_df,
        max_features=max_features,
        sublinear_tf=sublinear_tf,
        norm=norm,
        smooth_idf=smooth_idf,
    )
    tfidf_matrix = tfidf_vectorizer.fit_transform(texts)
    feature_names = tfidf_vectorizer.get_feature_names_out()

    print(f"✓ TF‑IDF matrix created: {tfidf_matrix.shape}")
    print(f"  {tfidf_matrix.shape[0]} documents, {tfidf_matrix.shape[1]} terms")
    print(
        f"  Matrix density: "
        f"{(tfidf_matrix.nnz / (tfidf_matrix.shape[0] * tfidf_matrix.shape[1]) * 100):.2f}%"
    )

    # Train multiple NMF models in parallel
    print(f"\nTraining {len(topic_list)} NMF model(s) in parallel...")
    
    results = Parallel(n_jobs=N_JOBS, backend='loky', verbose=10)(
        delayed(_train_single_nmf)(tfidf_matrix, n_topics)
        for n_topics in topic_list
    )

    # Organize results into dictionaries
    W_matrices: dict[int, np.ndarray] = {}
    H_matrices: dict[int, np.ndarray] = {}
    nmf_models: dict[int, NMF] = {}

    for n_topics, W, H, nmf_model in results:
        W_matrices[n_topics] = W
        H_matrices[n_topics] = H
        nmf_models[n_topics] = nmf_model
        
        print(f"\n✓ NMF finished for {n_topics} topics")
        print(f"  Reconstruction error: {nmf_model.reconstruction_err_:.4f}")
        print(f"  Number of iterations: {nmf_model.n_iter_}")

    return W_matrices, H_matrices, feature_names, nmf_models, tfidf_vectorizer


def apply_tfidf_nmf(
    texts: List[str],
    n_topics: int = 10,
    n_top_words: int = 20,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, NMF, TfidfVectorizer]:
    """
    Wrapper to keep the historical signature: TF‑IDF + a single NMF.
    Uses apply_tfidf_and_multi_nmf internally.
    """
    W_dict, H_dict, feature_names, nmf_models, tfidf_vectorizer = apply_tfidf_and_multi_nmf(
        texts, [n_topics], n_top_words=n_top_words
    )
    W = W_dict[n_topics]
    H = H_dict[n_topics]
    nmf_model = nmf_models[n_topics]
    return W, H, feature_names, nmf_model, tfidf_vectorizer


def compute_cache_key() -> str:
    """
    Compute a unique cache key based on parameters affecting extraction and lemmatization.
    
    The cache key is a hash of:
      - SOURCE_TYPE
      - Data file paths and their modification times
      - MIN_CHARS, MAX_CHARS
      - LANGUAGE
      - KEEP_POS
      - FAST_LEMMATIZATION
    
    Returns:
        Hex string representing the cache key.
    """
    key_parts = [
        f"source_type={SOURCE_TYPE}",
        f"language={LANGUAGE}",
        f"min_chars={MIN_CHARS}",
        f"max_chars={MAX_CHARS}",
        f"keep_pos={','.join(sorted(KEEP_POS))}",
        f"fast_lemmatization={FAST_LEMMATIZATION}",
    ]
    
    # Add data file paths and modification times
    if SOURCE_TYPE == "europresse":
        html_path = Path(HTML_FOLDER)
        if html_path.exists():
            html_files = sorted(html_path.glob("*.HTML"))
            for f in html_files:
                if f.exists():
                    key_parts.append(f"file={f.name}|mtime={f.stat().st_mtime}")
    elif SOURCE_TYPE == "istex":
        istex_path = Path(ISTEX_FOLDER)
        if istex_path.exists():
            key_parts.append(f"folder={ISTEX_FOLDER}|mtime={istex_path.stat().st_mtime}")
    elif SOURCE_TYPE == "csv":
        csv_path = Path(CSV_FILE)
        if csv_path.exists():
            key_parts.append(f"file={CSV_FILE}|mtime={csv_path.stat().st_mtime}")
    
    # Generate hash
    key_string = "|".join(key_parts)
    cache_key = hashlib.sha256(key_string.encode("utf-8")).hexdigest()
    return cache_key


def get_cache_path(cache_key: str) -> Path:
    """
    Get the cache file path for a given cache key.
    
    Args:
        cache_key: hex string representing the cache key
    
    Returns:
        Path to the cache file
    """
    cache_dir = Path(CACHE_DIR)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"preprocessing_{cache_key}.pkl"


def compute_config_hash() -> str:
    """
    Compute a unique configuration hash based on all parameters affecting
    the modeling results (TF-IDF, NMF, lemmatization, filtering, LSH).
    
    This hash is used to create separate result folders for different
    configurations, preventing results from being overwritten when
    parameters change.
    
    Parameters included:
      - TF-IDF: TFIDF_MAX_DF, TFIDF_MIN_DF, TFIDF_MAX_FEATURES,
                TFIDF_SUBLINEAR_TF, TFIDF_NORM, TFIDF_SMOOTH_IDF
      - Lemmatization: LANGUAGE, KEEP_POS, FAST_LEMMATIZATION
      - Document filtering: MIN_CHARS, MAX_CHARS
      - LSH deduplication: GO_REMOVE_NEAR_DUPLICATES, LSH_THRESHOLD,
                           LSH_NUM_PERM, LSH_MAX_TOKENS
    
    Returns:
        Short hash string (8 characters) representing the configuration.
    """
    config_parts = [
        # TF-IDF parameters
        f"tfidf_max_df={TFIDF_MAX_DF}",
        f"tfidf_min_df={TFIDF_MIN_DF}",
        f"tfidf_max_features={TFIDF_MAX_FEATURES}",
        f"tfidf_sublinear_tf={TFIDF_SUBLINEAR_TF}",
        f"tfidf_norm={TFIDF_NORM}",
        f"tfidf_smooth_idf={TFIDF_SMOOTH_IDF}",
        # Lemmatization parameters
        f"language={LANGUAGE}",
        f"keep_pos={','.join(sorted(KEEP_POS))}",
        f"fast_lemmatization={FAST_LEMMATIZATION}",
        # Document filtering
        f"min_chars={MIN_CHARS}",
        f"max_chars={MAX_CHARS}",
        # LSH deduplication
        f"go_remove_near_duplicates={GO_REMOVE_NEAR_DUPLICATES}",
        f"lsh_threshold={LSH_THRESHOLD}",
        f"lsh_num_perm={LSH_NUM_PERM}",
        f"lsh_max_tokens={LSH_MAX_TOKENS}",
    ]
    
    config_string = "|".join(config_parts)
    config_hash = hashlib.sha256(config_string.encode("utf-8")).hexdigest()
    # Return short hash (8 characters)
    return config_hash[:8]


def save_config_params(config_dir: Path, config_hash: str) -> None:
    """
    Save configuration parameters to a JSON file for traceability.
    
    This file documents which parameters were used to generate the results
    in this configuration folder, making it easy to reproduce or compare
    different runs.
    
    Args:
        config_dir: directory where the config file will be saved
        config_hash: the configuration hash (for verification)
    """
    config_data = {
        "config_hash": config_hash,
        "dataset_name": DATASET_NAME,
        "source_type": SOURCE_TYPE,
        "language": LANGUAGE,
        # TF-IDF parameters
        "tfidf": {
            "max_df": TFIDF_MAX_DF,
            "min_df": TFIDF_MIN_DF,
            "max_features": TFIDF_MAX_FEATURES,
            "sublinear_tf": TFIDF_SUBLINEAR_TF,
            "norm": TFIDF_NORM,
            "smooth_idf": TFIDF_SMOOTH_IDF,
        },
        # Lemmatization parameters
        "lemmatization": {
            "language": LANGUAGE,
            "keep_pos": KEEP_POS,
            "fast_lemmatization": FAST_LEMMATIZATION,
        },
        # Document filtering
        "document_filtering": {
            "min_chars": MIN_CHARS,
            "max_chars": MAX_CHARS,
        },
        # LSH deduplication
        "lsh_deduplication": {
            "enabled": GO_REMOVE_NEAR_DUPLICATES,
            "threshold": LSH_THRESHOLD,
            "num_perm": LSH_NUM_PERM,
            "max_tokens": LSH_MAX_TOKENS,
        },
        # Topic modeling (for reference, not included in hash)
        "topic_modeling": {
            "topic_list": TOPIC_LIST,
            "n_top_words": N_TOP_WORDS,
        },
        # Evaluation (for reference, not included in hash)
        "evaluation": {
            "eval_top_n_words": EVAL_TOP_N_WORDS,
            "coherence_metrics": COHERENCE_METRICS,
        },
    }
    
    config_file = config_dir / "config_params.json"
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Configuration parameters saved to '{config_file}'")


def load_from_cache(cache_key: str) -> dict:
    """
    Load cached preprocessing results if available and valid.
    
    Args:
        cache_key: hex string representing the expected cache key
    
    Returns:
        Dictionary containing cached data, or None if cache is invalid/missing.
        Expected keys:
          - 'cache_key': the cache key used to generate this cache
          - 'articles': list of raw extracted texts
          - 'article_soups': list of BeautifulSoup objects (or empty for ISTEX/CSV)
          - 'columns_dict': metadata dict (for ISTEX only, empty dict otherwise)
          - 'csv_metadata_all': metadata list (for CSV only, empty list otherwise)
          - 'processed_texts': lemmatized texts after preprocessing
    """
    cache_path = get_cache_path(cache_key)
    
    if not cache_path.exists():
        return None
    
    try:
        with open(cache_path, "rb") as f:
            cached_data = pickle.load(f)
        
        # Validate cache key
        if cached_data.get("cache_key") != cache_key:
            print(f"⚠️  Cache key mismatch, cache invalidated")
            return None
        
        # Validate structure
        required_keys = {
            "cache_key", "articles", "article_soups",
            "columns_dict", "csv_metadata_all", "processed_texts"
        }
        if not required_keys.issubset(cached_data.keys()):
            print(f"⚠️  Cache structure invalid, cache invalidated")
            return None
        
        return cached_data
    
    except Exception as e:
        print(f"⚠️  Error loading cache: {e}")
        return None


def save_to_cache(
    cache_key: str,
    articles: List[str],
    article_soups: List[BeautifulSoup],
    columns_dict: dict,
    csv_metadata_all: list,
    processed_texts: List[str],
) -> None:
    """
    Save preprocessing results to cache.
    
    Args:
        cache_key: hex string representing the cache key
        articles: list of raw extracted texts
        article_soups: list of BeautifulSoup objects (or empty for ISTEX/CSV)
        columns_dict: metadata dict (for ISTEX only, empty dict otherwise)
        csv_metadata_all: metadata list (for CSV only, empty list otherwise)
        processed_texts: lemmatized texts after preprocessing
    """
    cache_path = get_cache_path(cache_key)
    
    cached_data = {
        "cache_key": cache_key,
        "articles": articles,
        "article_soups": article_soups,
        "columns_dict": columns_dict,
        "csv_metadata_all": csv_metadata_all,
        "processed_texts": processed_texts,
    }
    
    try:
        with open(cache_path, "wb") as f:
            pickle.dump(cached_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        # Report cache size
        cache_size_mb = cache_path.stat().st_size / (1024 * 1024)
        print(f"\n✓ Preprocessing results cached to: {cache_path}")
        print(f"  Cache size: {cache_size_mb:.2f} MB")
    
    except Exception as e:
        print(f"⚠️  Error saving cache: {e}")


def compute_journal_date_span(meta_df: pd.DataFrame, grouping_column: str = "Journal_normalized") -> pd.DataFrame:
    """
    Compute, for each journal/grouping, the first and last dates
    it appears in the corpus, based on the final metadata table.

    This computation relies on the columns:
      - grouping_column (default: 'Journal_normalized', or CSV_GROUPING_COLUMN for CSV mode)
      - 'Date_normalized'

    Args:
        meta_df: DataFrame with metadata
        grouping_column: Name of the column to group by (e.g., "Journal_normalized" or "channel_title")

    Returns:
        DataFrame with columns [grouping_column, "Date_min", "Date_max"]

    The DataFrame passed must correspond to the final state of metadata
    after republication duplication and final deduplication.
    """
    # DEBUG: Check what we receive
    print(f"\n🔍 DEBUG compute_journal_date_span() ENTRY:")
    print(f"  grouping_column: {grouping_column}")
    print(f"  meta_df is None: {meta_df is None}")
    if meta_df is not None:
        print(f"  meta_df.empty: {meta_df.empty}")
        print(f"  meta_df columns: {meta_df.columns.tolist()}")
        print(f"  meta_df shape: {meta_df.shape}")
    
    if meta_df is None or meta_df.empty:
        print("  ❌ Returning early: meta_df is None or empty")
        return pd.DataFrame(columns=[grouping_column, "Date_min", "Date_max"])

    required_cols = {grouping_column, "Date_normalized"}
    if not required_cols.issubset(meta_df.columns):
        print(f"  ❌ Returning early: missing required columns")
        print(f"     Required: {required_cols}")
        print(f"     Available: {set(meta_df.columns)}")
        return pd.DataFrame(columns=[grouping_column, "Date_min", "Date_max"])

    df = meta_df.copy()

    # DEBUG: Initial state
    print(f"\n🔍 DEBUG compute_journal_date_span:")
    print(f"  Input rows: {len(df)}")
    print(f"  Columns: {df.columns.tolist()}")

    # Keep only rows where journal + date are non‑empty
    df[grouping_column] = df[grouping_column].astype(str).str.strip()
    df["Date_normalized"] = df["Date_normalized"].astype(str).str.strip()

    # DEBUG: Check values before filtering
    print(f"  Sample {grouping_column}: {df[grouping_column].head(3).tolist()}")
    print(f"  Sample Date_normalized: {df['Date_normalized'].head(3).tolist()}")

    df = df[
        df[grouping_column].notna()
        & df["Date_normalized"].notna()
        & (df[grouping_column] != "")
        & (df["Date_normalized"] != "")
    ]

    print(f"  After filtering empty values: {len(df)} rows")

    if df.empty:
        print("  ❌ DataFrame empty after filtering empty values")
        return pd.DataFrame(columns=[grouping_column, "Date_min", "Date_max"])

    # Robust date conversion; unparseable values are discarded
    df["_date_dt"] = pd.to_datetime(df["Date_normalized"], errors="coerce", dayfirst=True)
    print(f"  Successfully parsed to datetime: {df['_date_dt'].notna().sum()} rows")
    print(f"  Failed to parse: {df['_date_dt'].isna().sum()} rows")
    if df["_date_dt"].isna().any():
        print(f"  Sample unparseable dates: {df.loc[df['_date_dt'].isna(), 'Date_normalized'].head(5).tolist()}")
    
    df = df[df["_date_dt"].notna()]

    if df.empty:
        print("  ❌ DataFrame empty after date parsing")
        return pd.DataFrame(columns=[grouping_column, "Date_min", "Date_max"])

    grouped = (
        df.groupby(grouping_column)["_date_dt"]
        .agg(["min", "max"])
        .reset_index()
        .rename(columns={"min": "Date_min", "max": "Date_max"})
    )

    # Date formatting:
    #  - Europresse: keep full YYYY‑MM‑DD format
    #  - ISTEX     : show only the year (YYYY), as month/day are unavailable
    if SOURCE_TYPE == "istex":
        grouped["Date_min"] = grouped["Date_min"].dt.strftime("%Y")
        grouped["Date_max"] = grouped["Date_max"].dt.strftime("%Y")
    else:
        grouped["Date_min"] = grouped["Date_min"].dt.strftime("%Y-%m-%d")
        grouped["Date_max"] = grouped["Date_max"].dt.strftime("%Y-%m-%d")

    grouped = grouped.sort_values(grouping_column).reset_index(drop=True)
    return grouped


def main(reset_cache: bool = False):
    """
    Main entry point.
    
    Args:
        reset_cache: if True, ignore cached preprocessing and force reprocessing.
    """
    # ------------------------------------------------------------------
    # NOTE FOR STUDENTS
    # ------------------------------------------------------------------
    # All main project parameters (source type, language, topic counts,
    # TF‑IDF hyperparameters, etc.) are defined in `params.py`.
    #
    # For your experiments, DO NOT modify this file: only edit `params.py`
    # (in the same `sources/` folder).

    print("="*80)
    print("TOPIC MODELING: TF‑IDF + NMF")
    print(
        f"With SOTA lemmatization (spaCy) in "
        f"{'French' if LANGUAGE == 'fr' else 'English'} and POS filtering"
    )
    print(
        f"Document length filter: {MIN_CHARS} ≤ number of characters ≤ {MAX_CHARS}"
    )
    print("="*80)

    # ------------------------------------------------------------------
    # CACHE CHECK: Try to load preprocessing results from cache
    # ------------------------------------------------------------------
    articles = []
    article_soups = []
    columns_dict = {}
    csv_metadata_all: list[dict] = []
    processed_texts = []
    cache_hit = False
    
    if CACHE_ENABLED and not reset_cache:
        cache_key = compute_cache_key()
        print(f"\n🔍 Checking cache (key: {cache_key[:16]}...)")
        
        cached_data = load_from_cache(cache_key)
        if cached_data is not None:
            print("✓ Loading preprocessing results from cache...")
            articles = cached_data["articles"]
            article_soups = cached_data["article_soups"]
            columns_dict = cached_data["columns_dict"]
            csv_metadata_all = cached_data["csv_metadata_all"]
            processed_texts = cached_data["processed_texts"]
            cache_hit = True
            
            print(f"  - {len(articles)} articles loaded")
            print(f"  - {len(processed_texts)} preprocessed texts loaded")
            
            # Show corpus statistics for cached data
            print(f"\n📈 Corpus statistics (from cache):")
            print(f"  - Number of articles: {len(articles)}")
            print(f"  - Average length: {np.mean([len(a) for a in articles]):.0f} characters")
            print(f"  - Min length: {min([len(a) for a in articles])} characters")
            print(f"  - Max length: {max([len(a) for a in articles])} characters")
        else:
            print("⚠️  No valid cache found, will process from scratch")
    elif reset_cache:
        print("\n🔄 Cache reset requested (--reset flag), processing from scratch")
    
    # ------------------------------------------------------------------
    # EXTRACTION & LEMMATIZATION (only if cache miss)
    # ------------------------------------------------------------------
    if not cache_hit:
        # Step 1: scan files/folders according to the source type
        if SOURCE_TYPE == "europresse":
            html_files = sorted(Path(HTML_FOLDER).glob("*.HTML"))
            if not html_files:
                print(f"❌ No HTML files found in {HTML_FOLDER}")
                return
            
            print(f"\n📁 {len(html_files)} HTML file(s) found:")
            for f in html_files:
                print(f"  - {f.name}")
            
            # Step 2: Europresse extraction (text + soups)
            articles = []
            article_soups = []
            columns_dict = {}
            
            for html_file in html_files:
                print(f"\n🔄 Processing {html_file.name}...")
                arts, soups = extract_europresse_articles(
                    str(html_file),
                    min_chars=MIN_CHARS,
                    max_chars=MAX_CHARS,
                )
                articles.extend(arts)
                article_soups.extend(soups)
            
            print(f"\n✓ Total: {len(articles)} articles extracted from {len(html_files)} file(s)")
        
        elif SOURCE_TYPE == "istex":
            istex_path = Path(ISTEX_FOLDER)
            if not istex_path.exists():
                print(f"❌ Folder {ISTEX_FOLDER} does not exist")
                return
            
            # List subdirectories
            subdirs = [d for d in istex_path.iterdir() if d.is_dir()]
            if not subdirs:
                print(f"❌ No subdirectories found in {ISTEX_FOLDER}")
                return
            
            print(f"\n📁 {len(subdirs)} ISTEX subdirectory(ies) found")
            
            # Step 2: ISTEX extraction (text + metadata)
            articles = []
            article_soups = []  # Vide pour ISTEX
            all_columns_dict = []
            
            for subdir in subdirs:
                print(f"\n🔄 Processing {subdir.name}...")
                arts, cols_dict = extract_istex_articles(
                    str(subdir),
                    min_chars=MIN_CHARS,
                    max_chars=MAX_CHARS,
                )
                articles.extend(arts)
                all_columns_dict.append(cols_dict)
            
            # Merge per‑subdirectory metadata dictionaries
            columns_dict = {}
            for dico in all_columns_dict:
                for key, value in dico.items():
                    if key not in columns_dict:
                        columns_dict[key] = []
                    columns_dict[key].extend(value)
            
            print(f"\n✓ Total: {len(articles)} ISTEX articles extracted from {len(subdirs)} subdirectory(ies)")

        elif SOURCE_TYPE == "csv":
            csv_path = Path(CSV_FILE)
            if not csv_path.exists():
                print(f"❌ CSV file '{CSV_FILE}' does not exist")
                return

            print(f"\n📁 Loading CSV file: {csv_path}")
            try:
                df = pd.read_csv(
                    csv_path,
                    sep=CSV_SEPARATOR,
                    encoding=CSV_ENCODING,
                    dtype={'text': str, 'date': str},
                    na_filter=False,
                    engine='c'
                )
            except UnicodeDecodeError:
                print(f"⚠️  Encoding '{CSV_ENCODING}' failed, trying with 'latin1' fallback...")
                try:
                    df = pd.read_csv(
                        csv_path,
                        sep=CSV_SEPARATOR,
                        encoding='latin1',
                        dtype={'text': str, 'date': str},
                        na_filter=False,
                        engine='c'
                    )
                except Exception as e:
                    print(f"❌ Error while reading CSV file '{CSV_FILE}': {e}")
                    return
            except Exception as e:
                print(f"❌ Error while reading CSV file '{CSV_FILE}': {e}")
                return

            required_cols = {"text", "date"}
            missing = required_cols - set(df.columns)
            if missing:
                print(
                    "❌ CSV file is missing required column(s): "
                    + ", ".join(sorted(missing))
                )
                print("   Required columns are: 'text' and 'date' (format dd-mm-yyyy).")
                return

            # Validate date format (dd-mm-yyyy)
            parsed_dates = pd.to_datetime(
                df["date"],
                format=CSV_DATE_FORMAT,
                errors="coerce",
                dayfirst=False,
            )
            invalid_mask = parsed_dates.isna()
            if invalid_mask.any():
                invalid_values = df.loc[invalid_mask, "date"].unique()
                sample_values = ", ".join(map(str, invalid_values[:5]))
                print(
                    "❌ Some dates do not match the expected format "
                    f"'{CSV_DATE_FORMAT}' (dd-mm-yyyy)."
                )
                print(f"   Example invalid value(s): {sample_values}")
                return

            # Filter documents by length (vectorized approach - much faster than iterrows)
            df["text_length"] = df["text"].str.len()
            nb_too_short = (df["text_length"] < MIN_CHARS).sum()
            nb_too_long = (df["text_length"] >= MAX_CHARS).sum()
            
            # Create mask for valid documents
            valid_mask = (df["text_length"] >= MIN_CHARS) & (df["text_length"] < MAX_CHARS)
            df_valid = df[valid_mask].copy()

            # Clean texts (vectorized)
            df_valid["text_cleaned"] = df_valid["text"].apply(
                lambda x: transform_text(remove_urls_hashtags_emojis_mentions_emails(x))
            )
            
            # Extract articles
            articles = df_valid["text_cleaned"].tolist()
            article_soups: list[BeautifulSoup] = []  # unused for CSV

            # Build CSV metadata (fully vectorized)
            df_meta = df_valid.copy()
            # Convert CSV dates to standardized YYYY-MM-DD format
            parsed_dates = pd.to_datetime(
                df_meta["date"],
                format=CSV_DATE_FORMAT,
                errors="coerce"
            )
            df_meta["Date_normalized"] = parsed_dates.dt.strftime("%Y-%m-%d")
            # Keep original date string for rows where parsing failed
            df_meta.loc[parsed_dates.isna(), "Date_normalized"] = df_meta.loc[parsed_dates.isna(), "date"].astype(str)
            
            # DEBUG: Check dates parsing
            print(f"\n🔍 DEBUG Date parsing:")
            print(f"  Total rows: {len(df_meta)}")
            print(f"  Successfully parsed dates: {(~parsed_dates.isna()).sum()}")
            print(f"  Failed to parse: {(parsed_dates.isna()).sum()}")
            if (parsed_dates.isna()).any():
                print(f"  Sample failed dates: {df_meta.loc[parsed_dates.isna(), 'date'].head(5).tolist()}")
            print(f"  Sample Date_normalized: {df_meta['Date_normalized'].head(5).tolist()}")
            
            df_meta["Num_characters"] = df_meta["text_length"].astype(int).astype(str)

            # Check if grouping column exists (for statistics and date span computation)
            # Note: In CSV mode, we keep the original column name (e.g., "channel_title")
            # instead of creating Journal_original/Journal_normalized columns
            if CSV_GROUPING_COLUMN:
                if CSV_GROUPING_COLUMN not in df_meta.columns:
                    print(f"⚠️  Configured grouping column '{CSV_GROUPING_COLUMN}' not found in CSV. Continuing without grouping statistics.")

            df_meta = df_meta.drop(columns=["text", "text_cleaned", "text_length", "date"])
            csv_metadata_all = df_meta.to_dict("records")

            print(f"\n✓ Total: {len(articles)} CSV documents kept after length filtering")
            if nb_too_short or nb_too_long:
                print(
                    f"  ({nb_too_short} documents ignored because < {MIN_CHARS} characters, "
                    f"{nb_too_long} documents ignored because ≥ {MAX_CHARS} characters)"
                )
        
        else:
            print(
                f"❌ SOURCE_TYPE '{SOURCE_TYPE}' not recognized. "
                "Use 'europresse', 'istex' or 'csv'."
            )
            return
        
        if not articles:
            print("❌ No article extracted. Check your input files.")
            return
        
        print(f"\n📈 Corpus statistics:")
        print(f"  - Number of articles: {len(articles)}")
        print(f"  - Average length: {np.mean([len(a) for a in articles]):.0f} characters")
        print(f"  - Min length: {min([len(a) for a in articles])} characters")
        print(f"  - Max length: {max([len(a) for a in articles])} characters")
        
        # Load spaCy model according to configured language
        nlp = load_spacy_model(LANGUAGE)
        
        # Step 3: lemmatization and POS filtering
        processed_texts = preprocess_and_lemmatize(
            articles,
            nlp,  # Pass the loaded nlp model
            keep_pos=KEEP_POS  # Common nouns, proper nouns, verbs
        )
        
        # ------------------------------------------------------------------
        # SAVE TO CACHE
        # ------------------------------------------------------------------
        if CACHE_ENABLED:
            cache_key = compute_cache_key()
            save_to_cache(
                cache_key=cache_key,
                articles=articles,
                article_soups=article_soups,
                columns_dict=columns_dict,
                csv_metadata_all=csv_metadata_all,
                processed_texts=processed_texts,
            )
    
    # Filter out empty documents after preprocessing, keeping alignment
    valid_texts: List[str] = []
    valid_articles: List[str] = []
    valid_soups: List[BeautifulSoup] = []
    valid_indices: List[int] = []  # For ISTEX/CSV: keep track of original indices
    
    if SOURCE_TYPE == "europresse":
        for text, raw, soup in zip(processed_texts, articles, article_soups):
            if text and text.strip():
                valid_texts.append(text)
                valid_articles.append(raw)
                valid_soups.append(soup)
    else:  # ISTEX / CSV (no soup, rely on indices)
        for i, (text, raw) in enumerate(zip(processed_texts, articles)):
            if text and text.strip():
                valid_texts.append(text)
                valid_articles.append(raw)
                valid_indices.append(i)
                # valid_soups stays empty for ISTEX/CSV
    
    print(f"\n📊 Valid documents after preprocessing: {len(valid_texts)}")
    
    if len(valid_texts) < 2:
        print("❌ Not enough valid documents to run the analysis.")
        return
    
    # ------------------------------------------------------------------
    # Step 3bis: UPSTREAM DEDUPLICATION BY TEXTUAL SIMILARITY (LSH)
    # ------------------------------------------------------------------
    # We apply deduplication on cleaned raw texts (valid_articles),
    # but use lemmatized texts (valid_texts) for TF‑IDF + NMF.
    if GO_REMOVE_NEAR_DUPLICATES:
        canonical_indices, rep_for_doc = find_near_duplicates_lsh(
            valid_articles,
            threshold=LSH_THRESHOLD,
            num_perm=LSH_NUM_PERM,
            max_tokens=LSH_MAX_TOKENS,
        )
    else:
        canonical_indices = list(range(len(valid_texts)))
        rep_for_doc = list(range(len(valid_texts)))
    
    print("\n" + "=" * 80)
    print("LSH DEDUPLICATION SUMMARY (BEFORE NMF)")
    print("=" * 80)
    print(f"  - Valid documents (after preprocessing): {len(valid_texts)}")
    print(f"  - Canonical documents for TF‑IDF + NMF: {len(canonical_indices)}")
    print(f"  - Near‑duplicates reassigned (LSH): {len(valid_texts) - len(canonical_indices)}")
    
    if len(canonical_indices) < 2:
        print(
            "❌ Not enough canonical documents after LSH deduplication "
            f"({len(canonical_indices)}). Analysis impossible."
        )
        return
    
    # Texts actually used for TF‑IDF + NMF (deduplicated corpus)
    model_texts: List[str] = [valid_texts[i] for i in canonical_indices]
        
    # Step 4: TF‑IDF + multi‑NMF (on canonical documents only)
    print("\n" + "=" * 80)
    print("TF‑IDF + NMF ON DEDUPLICATED CORPUS")
    print("=" * 80)
    print(f"  - Number of training documents (canonical): {len(model_texts)}")
    
    W_dict, H_dict, feature_names, nmf_models, vectorizer = apply_tfidf_and_multi_nmf(
        model_texts,
        topic_list=TOPIC_LIST,
        n_top_words=N_TOP_WORDS,
        max_df=TFIDF_MAX_DF,
        min_df=TFIDF_MIN_DF,
        max_features=TFIDF_MAX_FEATURES,
        sublinear_tf=TFIDF_SUBLINEAR_TF,
        norm=TFIDF_NORM,
        smooth_idf=TFIDF_SMOOTH_IDF,
    )

    # ------------------------------------------------------------------
    # Step 4ter: TOPIC QUALITY EVALUATION (on canonical corpus)
    # ------------------------------------------------------------------
    # For each topic configuration k, compute quality metrics:
    #   - coherence (c_v, c_npmi) via gensim.
    #
    # Evaluation is done on W/H defined only on canonical documents
    # (model_texts), before score re‑assignment to near‑duplicates and
    # before metadata‑based deduplication.
    topic_quality_by_k = evaluate_all_topic_configs(
        model_texts=model_texts,
        W_dict=W_dict,
        H_dict=H_dict,
        feature_names=feature_names,
        top_n_words=EVAL_TOP_N_WORDS,
        coherence_metrics=COHERENCE_METRICS,
    )
    
    # ------------------------------------------------------------------
    # Step 4bis: REASSIGN SCORES TO ORIGINAL DOCUMENTS
    # ------------------------------------------------------------------
    # W_dict contains a W matrix per topic configuration, with one row
    # per canonical document (len(canonical_indices)).
    # Here we rebuild W_full so there is one row per valid document
    # (len(valid_texts)), copying the canonical representative's scores
    # to all its near‑duplicates.
    num_valid_docs = len(valid_texts)
    doc_index_to_row = {
        doc_idx: row_idx for row_idx, doc_idx in enumerate(canonical_indices)
    }
    
    print("\n" + "=" * 80)
    print("REASSIGNING NMF SCORES TO NEAR‑DUPLICATES")
    print("=" * 80)
    print(f"  - Canonical documents (rows of W from NMF): {len(canonical_indices)}")
    print(f"  - Final documents (valid_texts / valid_articles): {num_valid_docs}")
    print("  - Each final document receives the scores of its canonical representative")
    print("    before republication handling (Europresse) and metadata‑based deduplication.")
    
    for n_topics, W_canonical in W_dict.items():
        if W_canonical.shape[0] != len(canonical_indices):
            raise ValueError(
                "Internal inconsistency: W matrix does not match the number "
                "of canonical documents."
            )
        W_full = np.zeros((num_valid_docs, W_canonical.shape[1]), dtype=W_canonical.dtype)
        for j in range(num_valid_docs):
            rep_idx = rep_for_doc[j]
            row_idx = doc_index_to_row.get(rep_idx)
            if row_idx is None:
                raise ValueError(
                    f"Canonical representative not found for document {j} "
                    f"(rep_idx={rep_idx})."
                )
            W_full[j, :] = W_canonical[row_idx, :]
        W_dict[n_topics] = W_full
    
    # From here on, all W matrices have one row per valid document
    # (including near‑duplicates), guaranteeing alignment with metadata
    # and later deduplication steps by title/journal/date.
    
    # Prepare metadata aligned with valid_articles/valid_soups
    # (or columns_dict / csv_metadata_all for ISTEX / CSV)
    metadata_rows = []
    if SOURCE_TYPE == "europresse":
        for raw_text, soup in zip(valid_articles, valid_soups):
            metadata_rows.append(extract_metadata_from_soup(soup, raw_text))
    elif SOURCE_TYPE == "istex":
        # Pour ISTEX, on utilise columns_dict et les valid_indices
        for idx in valid_indices:
            metadata_rows.append(
                format_istex_metadata(columns_dict, idx, articles[idx])
            )
    elif SOURCE_TYPE == "csv":
        # Pour CSV, on relie les métadonnées du CSV aux documents valides
        for idx in valid_indices:
            metadata_rows.append(csv_metadata_all[idx])
    
    # Sanity check: alignment documents ↔ metadata
    assert len(metadata_rows) == len(valid_texts), (
        "Inconsistency between the number of valid texts and the number of metadata rows."
    )
    
    # Step 5: extract republication information (per article)
    #         -> list of normalized journals from `apd-wrapper` blocks
    if SOURCE_TYPE == "europresse":
        repub_journals_per_doc: list[list[str]] = [
            extract_republication_sources(soup) for soup in valid_soups
        ]
    else:
        # Pour ISTEX, pas de republication
        repub_journals_per_doc: list[list[str]] = [[] for _ in range(len(valid_texts))]
    
    assert len(repub_journals_per_doc) == len(
        metadata_rows
    ), "Inconsistency between metadata size and repub_journals_per_doc."

    # ------------------------------------------------------------------
    # Step 5bis: COMPUTE TOTAL NUMBER OF PUBLICATIONS PER ARTICLE
    # ------------------------------------------------------------------
    # Objective: for each initial article (LSH duplicates/near‑duplicates
    # group), determine how many times it appears in the corpus by combining:
    #   - the number of documents in the group (duplicates/near‑duplicates),
    #   - republications coming from `apd-wrapper` blocks,
    #   - while avoiding counting twice republications already represented
    #     by a document in the group.
    num_docs_meta = len(metadata_rows)
    if len(rep_for_doc) != num_docs_meta:
        raise ValueError(
            "Internal inconsistency: rep_for_doc and metadata_rows do not have "
            f"the same length ({len(rep_for_doc)} vs {num_docs_meta})."
        )

    # Group documents by canonical representative (LSH)
    group_docs_by_rep: dict[int, list[int]] = {}
    for doc_idx, rep_idx in enumerate(rep_for_doc):
        group_docs_by_rep.setdefault(rep_idx, []).append(doc_idx)

    # Pre‑compute Num_publications for each canonical group
    num_publications_per_doc: list[int] = [1] * num_docs_meta
    for rep_idx, doc_indices in group_docs_by_rep.items():
        # Set of normalized journals observed in the group
        journaux_set: set[str] = set()
        for i in doc_indices:
            journal_raw = (metadata_rows[i].get("Journal_normalized") or "").strip().lower()
            if journal_raw:
                journaux_set.add(journal_raw)

        # Union of normalized republications
        repub_set: set[str] = set()
        for i in doc_indices:
            for repub_journal in repub_journals_per_doc[i]:
                norm_repub = (repub_journal or "").strip().lower()
                if norm_repub:
                    repub_set.add(norm_repub)

        # Republications not already present via a duplicate in the group
        repub_extra = repub_set - journaux_set
        num_duplicates = len(doc_indices)
        num_publications_group = num_duplicates + len(repub_extra)

        for i in doc_indices:
            num_publications_per_doc[i] = num_publications_group

    # Attach Num_publications to each metadata row
    for i, meta in enumerate(metadata_rows):
        meta["Num_publications"] = num_publications_per_doc[i]

    # ------------------------------------------------------------------
    # Step 6: DEDUPLICATION AFTER TF‑IDF / NMF
    # ------------------------------------------------------------------
    # An article is considered exactly the same if:
    #   - same title,
    #   - same normalized journal,
    #   (regardless of date).
    # We keep the *first* occurrence of each (Title, Journal_normalized)
    # pair and merge republication journal lists into that row.
    #
    # This deduplication happens here, after TF‑IDF / NMF, but BEFORE:
    #   - duplication for republications,
    #   - writing W to disk.

    num_docs_before_dedup = len(metadata_rows)

    seen_keys: dict[tuple[str, str], int] = {}
    keep_indices: list[int] = []
    merged_repubs: dict[int, list[str]] = {}

    for idx, (meta, repubs) in enumerate(zip(metadata_rows, repub_journals_per_doc)):
        title_raw = (meta.get("Title") or "").strip()
        journal_raw = (meta.get("Journal_normalized") or "").strip()

        # Deduplication key:
        #   - Europresse / ISTEX: (Title, Journal_normalized)
        #   - CSV: per-row key so that this deduplication step has no effect
        if SOURCE_TYPE == "csv":
            key = (f"csv_row_{idx}", f"csv_row_{idx}")
        else:
            key = (title_raw.lower(), journal_raw.lower())

        if key not in seen_keys:
            # First occurrence of this article
            seen_keys[key] = idx
            keep_indices.append(idx)
            # Initialize the (already normalized) list of republications
            merged_repubs[idx] = list(dict.fromkeys([r for r in repubs if r]))
        else:
            # Duplicate: do NOT keep a new row, but merge republications
            # into the first article's entry.
            first_idx = seen_keys[key]
            base_list = merged_repubs[first_idx]
            for r in repubs:
                if r and r not in base_list:
                    base_list.append(r)

    # If no deduplication is needed, do not filter W
    if keep_indices and len(keep_indices) < num_docs_before_dedup:
        # Filter metadata and republications
        new_metadata_rows: list[dict] = []
        new_repubs_list: list[list[str]] = []
        for orig_idx in keep_indices:
            meta = dict(metadata_rows[orig_idx])
            new_metadata_rows.append(meta)
            new_repubs_list.append(merged_repubs[orig_idx])

        metadata_rows = new_metadata_rows
        repub_journals_per_doc = new_repubs_list

        # Filter all W matrices (one per topic configuration)
        keep_indices_arr = np.array(keep_indices, dtype=int)
        for n_topics in W_dict:
            W_dict[n_topics] = W_dict[n_topics][keep_indices_arr, :]

        print(
            "✓ Deduplication by (Title, Journal_normalized): "
            f"{num_docs_before_dedup} → {len(metadata_rows)} documents "
            f"({num_docs_before_dedup - len(metadata_rows)} duplicates removed)."
        )

    # Step 7: display and save results for each topic configuration
    
    print("\n" + "="*80)
    print("SAVING RESULTS (multi‑config NMF)")
    print("="*80)

    # Root folder for all results of this run
    results_root = Path(RESULTS_ROOT)
    results_root.mkdir(parents=True, exist_ok=True)

    # Folder dedicated to the current dataset, e.g. results/menopause/
    dataset_dir = results_root / DATASET_NAME
    dataset_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Configuration-specific folder to prevent overwriting results
    # ------------------------------------------------------------------
    # Compute configuration hash from all parameters affecting results
    config_hash = compute_config_hash()
    config_dir = dataset_dir / f"config_{config_hash}"
    config_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📁 Configuration folder: {config_dir.relative_to(results_root)}")
    print(f"   Config hash: {config_hash}")
    
    # Save configuration parameters for traceability
    save_config_params(config_dir, config_hash)

    # ------------------------------------------------------------------
    # Global summary of topic quality metrics (all configs)
    # ------------------------------------------------------------------
    if topic_quality_by_k:
        summary_path = save_topic_quality_summary(
            topic_quality_by_k=topic_quality_by_k,
            results_root=config_dir,
            dataset_name=DATASET_NAME,
            top_n_words=EVAL_TOP_N_WORDS,
        )
        print(
            f"✓ Global summary of topic quality metrics saved to "
            f"'{summary_path}'"
        )

    # Boolean flag so we compute the min/max dates table only once
    # (it does not depend on the number of topics).
    journal_date_span_saved = False

    for n_topics in TOPIC_LIST:
        print(f"\n--- NMF configuration: {n_topics} topics ---")
    
        W = W_dict[n_topics]
        H = H_dict[n_topics]
    
        # Build enriched W DataFrame
        assert W.shape[0] == len(metadata_rows), (
            f"Number of rows in W ({W.shape[0]}) different "
            f"from number of metadata rows ({len(metadata_rows)})."
        )
    
        W_df = pd.DataFrame(
            W,
            columns=[f'Topic_{i+1}' for i in range(W.shape[1])],
        )
    
        # Topic dominant par document (index 1-based)
        main_topic_index = W_df.values.argmax(axis=1) + 1
        W_df.insert(0, "Main_topic_index", main_topic_index)

        meta_df = pd.DataFrame(metadata_rows)

        # ------------------------------------------------------------------
        # DUPLICATE ROWS FOR REPUBLICATIONS
        # ------------------------------------------------------------------
        # For each article, we add:
        #   - the original row (main journal),
        #   - a cloned row for each republication journal detected
        #     in `apd-wrapper` / `source-name-APD` blocks.
        # For these duplicated rows, we replace both
        #   - "Journal_original"
        #   - "Journal_normalized"
        # with the normalized name returned by `extract_republication_sources`
        # (we do not have access here to the raw HTML string of the republication).

        duplicated_meta_rows: list[dict] = []
        duplicated_W_rows: list[dict] = []
        duplicated_flags: list[int] = []  # 0 = original row, 1 = added via also published

        meta_records = meta_df.to_dict(orient="records")
        w_records = W_df.to_dict(orient="records")

        for meta_row, w_row, repub_list in zip(
            meta_records, w_records, repub_journals_per_doc
        ):
            # Toujours ajouter la ligne originale
            duplicated_meta_rows.append(meta_row)
            duplicated_W_rows.append(w_row)
            duplicated_flags.append(0)

            base_journal = (meta_row.get("Journal_normalized") or "").strip().lower()
            seen_journals = {base_journal} if base_journal else set()

            # Add a cloned row for each republication journal
            for repub_journal in repub_list:
                norm_repub = (repub_journal or "").strip().lower()
                if not norm_repub or norm_repub in seen_journals:
                    continue
                seen_journals.add(norm_repub)

                new_meta = dict(meta_row)
                # Store the normalized version returned by
                # `extract_republication_sources` (already normalized) in
                # both journal‑related columns to keep consistency between
                # "original" and "normalized" on duplicated rows.
                new_meta["Journal_original"] = repub_journal
                new_meta["Journal_normalized"] = repub_journal
                duplicated_meta_rows.append(new_meta)
                duplicated_W_rows.append(dict(w_row))
                duplicated_flags.append(1)

        meta_df_expanded = pd.DataFrame(duplicated_meta_rows)
        # Explicit indicator: 0 = main article, 1 = duplicated row
        meta_df_expanded["Is_republication"] = duplicated_flags

        # ------------------------------------------------------------------
        # CLEAN TEXT FIELDS
        # ------------------------------------------------------------------
        # Objective:
        #   - remove multiple spaces,
        #   - trim leading/trailing spaces
        #   in all textual fields of the W table.
        #
        # Apply this cleaning only to values of type str,
        # leaving numeric values unchanged.
        def _clean_text_value(v):
            if isinstance(v, str):
                v = re.sub(r"\s+", " ", v)
                return v.strip()
            return v

        meta_df_expanded = meta_df_expanded.applymap(_clean_text_value)

        # ------------------------------------------------------------------
        # DROP AUTHOR COLUMNS
        # ------------------------------------------------------------------
        # Remove Authors / Raw_authors columns from the exported W table.
        for col in ["Authors", "Raw_authors"]:
            if col in meta_df_expanded.columns:
                meta_df_expanded = meta_df_expanded.drop(columns=[col])

        W_df_expanded = pd.DataFrame(duplicated_W_rows)

        assert meta_df_expanded.shape[0] == W_df_expanded.shape[0], (
            "Inconsistency between metadata row count and W after duplication."
        )

        # ------------------------------------------------------------------
        # FINAL DEDUPLICATION ON (Title, Journal_normalized, Date_normalized)
        # ------------------------------------------------------------------
        # Objective: keep only one row per unique combination of
        # (Title, Journal_normalized, Date_normalized), including rows
        # created by republication.
        # Prefer original rows (Is_republication == 0) when possible.
        
        num_rows_before_final_dedup = meta_df_expanded.shape[0]
        
        if all(col in meta_df_expanded.columns for col in ["Title", "Journal_normalized", "Date_normalized"]):
            # Build a deduplication key
            dedup_keys = (
                meta_df_expanded["Title"].fillna("").str.strip().str.lower() + "|||" +
                meta_df_expanded["Journal_normalized"].fillna("").str.strip().str.lower() + "|||" +
                meta_df_expanded["Date_normalized"].fillna("").str.strip().str.lower()
            )
            
            # Build a temporary DataFrame with keys and republication flags
            temp_df = pd.DataFrame({
                "dedup_key": dedup_keys,
                "is_repub": meta_df_expanded["Is_republication"],
                "original_index": range(len(meta_df_expanded))
            })
            
            # Group by key and keep the first occurrence, favoring Is_republication == 0
            temp_df_sorted = temp_df.sort_values(by=["dedup_key", "is_repub", "original_index"])
            keep_final_indices = temp_df_sorted.groupby("dedup_key", sort=False).first()["original_index"].tolist()
            
            # Filter both DataFrames
            meta_df_expanded = meta_df_expanded.iloc[keep_final_indices].reset_index(drop=True)
            W_df_expanded = W_df_expanded.iloc[keep_final_indices].reset_index(drop=True)
            
            num_rows_after_final_dedup = meta_df_expanded.shape[0]
            if num_rows_after_final_dedup < num_rows_before_final_dedup:
                print(
                    f"✓ Final deduplication (Title, Journal_normalized, Date_normalized): "
                    f"{num_rows_before_final_dedup} → {num_rows_after_final_dedup} rows "
                    f"({num_rows_before_final_dedup - num_rows_after_final_dedup} duplicates removed)."
                )

        W_enriched = pd.concat([meta_df_expanded, W_df_expanded], axis=1)
        W_enriched.index = [f"Doc_{i+1}" for i in range(W_enriched.shape[0])]

        # Column Is_republication is only used for final deduplication.
        # Do not export it to final CSVs.
        if "Is_republication" in W_enriched.columns:
            W_enriched = W_enriched.drop(columns=["Is_republication"])

        # In CSV mode, drop Journal_original and Journal_normalized columns
        # (they are not used and the original grouping column is kept instead)
        if SOURCE_TYPE == "csv":
            cols_to_drop = [col for col in ["Journal_original", "Journal_normalized"] if col in W_enriched.columns]
            if cols_to_drop:
                W_enriched = W_enriched.drop(columns=cols_to_drop)

        # ------------------------------------------------------------------
        # FINAL UNIQUENESS CHECK
        # ------------------------------------------------------------------
        # Ensure there are no duplicates on (Title, Journal_normalized, Date_normalized)
        if all(col in W_enriched.columns for col in ["Title", "Journal_normalized", "Date_normalized"]):
            duplicates = W_enriched.duplicated(
                subset=["Title", "Journal_normalized", "Date_normalized"],
                keep=False,
            )
            num_duplicates = duplicates.sum()
            assert num_duplicates == 0, (
                f"❌ ERROR: {num_duplicates} duplicates detected on "
                f"(Title, Journal_normalized, Date_normalized) after final deduplication!"
            )
            print(
                "✓ Uniqueness check: no duplicates on "
                "(Title, Journal_normalized, Date_normalized)"
            )

        # ------------------------------------------------------------------
        # MIN/MAX DATES PER JOURNAL TABLE (once)
        # ------------------------------------------------------------------
        if not journal_date_span_saved:
            # Determine the grouping column to use based on SOURCE_TYPE
            if SOURCE_TYPE == "csv":
                # For CSV mode, use the configured grouping column if it exists
                if CSV_GROUPING_COLUMN and CSV_GROUPING_COLUMN in meta_df_expanded.columns:
                    grouping_col = CSV_GROUPING_COLUMN
                else:
                    grouping_col = None  # Skip if no valid grouping column
            else:
                # For Europresse/ISTEX, use Journal_normalized
                grouping_col = "Journal_normalized"
            
            if grouping_col:
                journal_span_df = compute_journal_date_span(meta_df_expanded, grouping_column=grouping_col)
                journal_span_path = config_dir / "journal_min_max_dates.csv"

                if not journal_span_df.empty:
                    journal_span_df.to_csv(journal_span_path, index=False)
                    print(
                        f"✓ Min/max dates per {grouping_col} saved to '{journal_span_path}'"
                    )
                else:
                    print(
                        f"⚠️ Could not compute min/max dates per {grouping_col} "
                        "(no valid dates found)."
                    )
            else:
                print(
                    "⚠️ Skipping min/max dates computation: no valid grouping column available."
                )

            journal_date_span_saved = True

        # Topic-specific folder for this configuration and number of topics,
        # e.g. results/menopause/config_a3f2b1c4/5t/
        topic_dir = config_dir / f"{n_topics}t"
        topic_dir.mkdir(parents=True, exist_ok=True)

        w_filename = topic_dir / "W_documents_topics.csv"
        # Do not export the index column (Doc_x) in the CSV
        W_enriched.to_csv(w_filename, index=False)
        print(f"✓ Enriched W matrix (with republications) saved to '{w_filename}'")
    
        # Build compact H DataFrame: top N_TOP_WORDS words per topic
        # Keep two columns per topic: <Topic_k_Term>, <Topic_k_Weight>
        n_top = min(N_TOP_WORDS, H.shape[1])
        h_data: dict[str, list] = {}

        for topic_idx, topic in enumerate(H):
            # Indices of the n_top most important words for this topic
            top_indices = topic.argsort()[-n_top:][::-1]
            top_words = [feature_names[i] for i in top_indices]
            top_weights = [float(topic[i]) for i in top_indices]

            col_word = f"Topic_{topic_idx+1}_Term"
            col_weight = f"Topic_{topic_idx+1}_Weight"

            h_data[col_word] = top_words
            h_data[col_weight] = top_weights

        H_df = pd.DataFrame(h_data)
        # Index encodes rank (1 = most important word)
        H_df.index = range(1, n_top + 1)

        h_filename = topic_dir / "H_topics_terms.csv"
        H_df.to_csv(h_filename, index=False)
        print(
            f"✓ H matrix (top {n_top} words per topic, compact format) saved to '{h_filename}'"
        )

    # Optional: save a small metrics file per configuration
    if topic_quality_by_k and SAVE_TOPIC_LEVEL_METRICS:
        save_topic_quality_per_config(
            topic_quality_by_k=topic_quality_by_k,
            results_root=config_dir,
            dataset_name=DATASET_NAME,
        )
    
    print("\n" + "="*80)
    print("✅ ANALYSIS COMPLETED SUCCESSFULLY FOR ALL CONFIGURATIONS!")
    print("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Topic modeling with TF-IDF + NMF and lemmatization caching"
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Force reprocessing by ignoring cached extraction and lemmatization results"
    )
    args = parser.parse_args()
    main(reset_cache=args.reset)


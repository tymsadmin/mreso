"""
Main parameters for TF‑IDF + NMF topic modeling.
"""

# ----------------------------------------------------------------------
# General parameters
# ----------------------------------------------------------------------

# Source type: "europresse", "istex" or "csv"
SOURCE_TYPE: str = "csv"

# Corpus language: "fr" (French) or "en" (English)
LANGUAGE: str = "en"

# Human‑readable corpus / experiment name (used to organize results)
# Examples: "menopause", "hospital_2022", "covid_fr_2020"
DATASET_NAME: str = "usa_youtube"

# Root folder where all results (matrices, summaries, etc.) will be stored
RESULTS_ROOT: str = "results"

# Cache folder for storing preprocessed data (extraction + lemmatization)
CACHE_DIR: str = "cache"

# Enable/disable caching of extraction and lemmatization results
CACHE_ENABLED: bool = True

# Data folders
HTML_FOLDER: str = "html_sources"   # For Europresse
ISTEX_FOLDER: str = "istex_sources"  # For ISTEX
CSV_FILE: str = "csv_sources/usa_youtube.csv"  # For CSV input when SOURCE_TYPE == "csv"

# Expected date format in the 'date' column when SOURCE_TYPE == "csv"
CSV_DATE_FORMAT: str = "%d-%m-%Y"

# CSV separator character (default ';' for European format)
CSV_SEPARATOR: str = ";"

# CSV encoding (utf-8, latin1, cp1252, etc.)
CSV_ENCODING: str = "utf-8"

# CSV grouping column (optional): column name to use for grouping documents
# This column will be mapped to Journal_normalized for statistics and date span computation.
# 
# Examples:
#   CSV_GROUPING_COLUMN: str | None = None              # No grouping (default)
#   CSV_GROUPING_COLUMN: str | None = "channel_title"   # For YouTube data
#   CSV_GROUPING_COLUMN: str | None = "source"          # For other datasets
#
# If the specified column doesn't exist in your CSV, a warning will be displayed
# and the analysis will continue without grouping statistics.
CSV_GROUPING_COLUMN: str | None = "channel_title"

# Document length filter (character count)
MIN_CHARS: int = 500
MAX_CHARS: int = 1000_000

# ----------------------------------------------------------------------
# CSV source format (when SOURCE_TYPE == "csv")
# ----------------------------------------------------------------------
# The input CSV file must contain at least the following columns:
#   - "text": raw textual content of the document
#   - "date": publication date in format dd-mm-yyyy
# Any additional columns are kept as-is and propagated to the final
# W_documents_topics.csv file (one column per original CSV field).


# ----------------------------------------------------------------------
# TF‑IDF parameters
# ----------------------------------------------------------------------

# Max document frequency: ignore words present in >95% of documents
TFIDF_MAX_DF: float = 0.95

# Min document frequency: ignore words present in <3 documents (too rare)
TFIDF_MIN_DF: int = 3

# Maximum number of terms kept
TFIDF_MAX_FEATURES: int = 5000

# If True: down‑weights very frequent words in the corpus (weighting 1 + log(tf))
TFIDF_SUBLINEAR_TF: bool = False

# Normalization of TF‑IDF document vectors:
#   - "l2": L2 (Euclidean) normalization → each document vector has unit norm (default, recommended)
#   - "l1": L1 (Manhattan) normalization → sum of absolute values = 1
#   - None: no normalization → keeps raw TF-IDF values
TFIDF_NORM: str | None = None

# Smooth IDF weights by adding 1 to document frequencies:
#   - True: smooth IDF = log((n + 1) / (df + 1)) + 1 → avoids zero divisions, reduces weight of very rare terms (default)
#   - False: raw IDF = log(n / df) + 1 → no smoothing
TFIDF_SMOOTH_IDF: bool = True


# ----------------------------------------------------------------------
# NMF parameters
# ----------------------------------------------------------------------

# List of topic counts to test
# Example: [5, 7, 10]
TOPIC_LIST: list[int] = [5, 7, 12, 20, 30, 40, 50]

# Number of words displayed / exported per topic
# (used for the compact H matrix: top N terms per topic)
N_TOP_WORDS: int = 50


# ----------------------------------------------------------------------
# Lemmatization / POS filtering
# ----------------------------------------------------------------------

# spaCy POS tags to keep
# Common nouns, proper nouns, verbs
KEEP_POS: list[str] = ["NOUN", "PROPN", "VERB", "ADJ"]

# Fast lemmatization mode: use small spaCy models (sm) instead of large (lg)
# True = faster processing with small models, False = better accuracy with large models
FAST_LEMMATIZATION: bool = True


# ----------------------------------------------------------------------
# Topic quality evaluation
# ----------------------------------------------------------------------
#
# Number of words per topic used for coherence metrics (c_v, c_npmi).
EVAL_TOP_N_WORDS: int = 20
#
# List of coherence metrics to compute with gensim
# (global average coherence over all topics).
# Only c_v and c_npmi are supported in the default pipeline.
COHERENCE_METRICS: list[str] = ["c_v", "c_npmi"]
#
# If True, in addition to the global summary file, write a small
# metrics file per configuration (in each
# `results/<DATASET_NAME>/<k>t/` folder).
SAVE_TOPIC_LEVEL_METRICS: bool = False


# ----------------------------------------------------------------------
# Near‑duplicate removal via textual similarity (MinHash LSH)
# ----------------------------------------------------------------------

# Enable / disable LSH‑based deduplication
GO_REMOVE_NEAR_DUPLICATES: bool = True

# Jaccard similarity threshold
LSH_THRESHOLD: float = 0.8

# Number of permutations for MinHash
LSH_NUM_PERM: int = 256

# Maximum number of tokens considered per document
LSH_MAX_TOKENS: int = 100


# ----------------------------------------------------------------------
# Parallel processing
# ----------------------------------------------------------------------

# Number of CPU cores to use for parallel processing
# Examples with 8 cores:
#   N_JOBS: int = max(1, os.cpu_count() - 2) if os.cpu_count() else 1  # Uses 6 cores (default)
#   N_JOBS: int = max(1, os.cpu_count() - 1) if os.cpu_count() else 1  # Uses 7 cores
#   N_JOBS: int = os.cpu_count() if os.cpu_count() else 1              # Uses all 8 cores
#   N_JOBS: int = 4                                                     # Uses exactly 4 cores
import os
N_JOBS: int = max(1, os.cpu_count() - 2) if os.cpu_count() else 1




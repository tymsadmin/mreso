## tfidf-nmf: topic modeling with TF‑IDF + NMF

Automatic topic modeling on text corpora (medical or otherwise) from Europresse, ISTEX or CSV files, with:
- robust lemmatization via spaCy,
- TF‑IDF vectorization,
- NMF factorization to extract topics,
- quality metrics to help choose the best number of topics.

This repository is designed for students/practitioners who want to **quickly run topic modeling experiments** without digging into implementation details.

---

## 1. General idea

- **TF‑IDF**: turns each document into a vector, weighting words by their frequency in the document and in the corpus.
- **NMF (Non‑negative Matrix Factorization)**: decomposes the TF‑IDF matrix into two matrices:
  - \( W \) (Documents × Topics): importance of each topic in each document,
  - \( H \) (Topics × Words): importance of each word in each topic.
- **Several numbers of topics** are tested in a single run (`TOPIC_LIST`).
- **Topic quality**: topic coherences `c_v` and `c_npmi` to help choose the best number of topics.

Everything is driven by the configuration file `sources/params.py`.

---

## 2. Installation with uv

The project uses **Python 3.13** (see `pyproject.toml`) and **[uv](https://github.com/astral-sh/uv)** as dependency manager.

### 2.1 Prerequisites

- `uv` installed (see the official `uv` documentation).
  - `uv` will automatically use/install a Python 3.13 compatible with `pyproject.toml`.

### 2.2 Installing dependencies

From the project root (`tfidf-nmf`):

```bash
uv sync
```

This creates a virtual environment (by default `.venv/`) and installs dependencies from `pyproject.toml`:
- `numpy`, `pandas`, `scikit-learn`,
- `spacy`,
- `beautifulsoup4`, `lxml`,
- `datasketch`,
- `gensim` (for topic coherence metrics),
- etc.

---

## 3. Data preparation

The document source is configured via `SOURCE_TYPE` in `sources/params.py`:
- `"europresse"`: Europresse HTML exports,
- `"istex"`: ISTEX corpus (text + JSON files),
- `"csv"`: a simple CSV file with one row per document.

### 3.1 Europresse (HTML)

1. Place your HTML files in a folder, for example `html_sources/`.
2. In `sources/params.py`:

```python
SOURCE_TYPE = "europresse"
HTML_FOLDER = "html_sources"
```

The script:
- splits each HTML file into articles,
- cleans the text in depth (URLs, multiple spaces, removal of all unnecessary HTML tags, etc.),
- extracts metadata (title, journal, date, authors, etc.),
- detects republications (same article appearing in other newspapers).

### 3.2 ISTEX (JSON + text)

#### Downloading from ISTEX platform

1. Go to the [ISTEX platform](https://www.istex.fr/) and perform your corpus query.
2. When exporting results, **select both formats**: `cleaned` (cleaned full text) and `json` (metadata).
3. Download the resulting archive(s) and extract them.
4. Place the extracted subdirectories directly into the `istex_sources/` folder.

The resulting structure will look like:

```text
istex_sources/
├── corpus1/
│   ├── article1.cleaned
│   ├── article1.json
│   ├── article2.cleaned
│   └── article2.json
└── corpus2/
    ├── article3.cleaned
    └── article3.json
```

#### Configuration

In `sources/params.py`:

```python
SOURCE_TYPE = "istex"
ISTEX_FOLDER = "istex_sources"
```

The script automatically detects all `.cleaned` / `.json` pairs in subdirectories. For each pair:
- loads the cleaned full text,
- extracts relevant metadata from the JSON files (title, authors, date, journal, etc.),
- filters out documents that are too short/long (`MIN_CHARS` / `MAX_CHARS`).

### 3.3 CSV (tabular text + metadata)

For a CSV source, you provide a **single CSV file** with one row per document.

Required columns:
- `text`: raw textual content of the document,
- `date`: publication date in format `dd-mm-yyyy`.

Any other column is allowed (e.g. `id_document`, `source`, `category`, …) and will be **propagated as‑is** to the final `W_documents_topics.csv` table.

In `sources/params.py`:

```python
SOURCE_TYPE = "csv"
CSV_FILE = "csv_sources/documents.csv"
CSV_DATE_FORMAT = "%d-%m-%Y"
```

Notes:
- the `date` column is strictly validated against `CSV_DATE_FORMAT`; if at least one value does not match, the CSV is rejected with an explicit error message,
- `MIN_CHARS` / `MAX_CHARS` are applied to the `text` content, exactly like for Europresse / ISTEX.

---

## 4. Experiment configuration (`sources/params.py`)

This is the **only file you need to edit** to run experiments. Key parameters:

- **Data source**
  - `SOURCE_TYPE`: `"europresse"`, `"istex"` or `"csv"`.
  - `HTML_FOLDER`, `ISTEX_FOLDER`: folders where the raw data is stored.
  - `CSV_FILE`: path to the input CSV file when `SOURCE_TYPE = "csv"`.
  - `CSV_DATE_FORMAT`: expected format for the `date` column (default `"%d-%m-%Y"`).

- **Language and corpus**
  - `LANGUAGE`: `"fr"` or `"en"` (selects the spaCy model).
  - `DATASET_NAME`: human‑readable name for the corpus (used in the results directory tree).
  - `RESULTS_ROOT`: root folder where results are stored (default `"results"`).

- **Document filtering**
  - `MIN_CHARS`, `MAX_CHARS`: min/max document length (in characters).

- **TF‑IDF**
  - `TFIDF_MAX_DF`: ignore words present in more than X% of documents (default 0.95).
  - `TFIDF_MIN_DF`: ignore words present in fewer than N documents (default 3).
  - `TFIDF_MAX_FEATURES`: maximum vocabulary size (default 5000).
  - `TFIDF_SUBLINEAR_TF`: if `True`, down-weights very frequent words using \(1 + \log(tf)\) instead of raw term frequency (default `False`).
  - `TFIDF_NORM`: normalization of TF-IDF document vectors. Options: `"l2"` (Euclidean norm, recommended), `"l1"` (Manhattan norm), or `None` (no normalization, keeps raw values). Default: `None`.
  - `TFIDF_SMOOTH_IDF`: if `True`, smooths IDF weights by adding 1 to document frequencies to avoid zero divisions and reduce weight of very rare terms (default `True`).

- **NMF**
  - `TOPIC_LIST`: list of topic counts to test (e.g. `[5, 10, 20]`).
  - `N_TOP_WORDS`: number of words exported per topic in the compact H matrix.

- **Linguistic pre‑processing**
  - `KEEP_POS`: POS tags to keep (default `["NOUN", "PROPN", "VERB"]`).

- **Evaluation and deduplication**
  - `EVAL_TOP_N_WORDS`: number of words used for coherence metrics (`c_v`, `c_npmi`).
  - `COHERENCE_METRICS`: list of coherence metrics to compute (default `["c_v", "c_npmi"]`).
  - `GO_REMOVE_NEAR_DUPLICATES`: enable near‑duplicate removal (MinHash LSH).

For most pedagogical uses, you only need to:
- adjust `SOURCE_TYPE`, `LANGUAGE`, `DATASET_NAME`,
- choose `TOPIC_LIST` and optionally tweak `MIN_CHARS` / `MAX_CHARS`.

---

## 5. Run topic modeling

From the project root:

```bash
uv run python -m sources.topic_modeling_nmf
```

The script automatically performs:
1. **Extraction** of articles from Europresse, ISTEX or CSV.
2. **Cleaning & lemmatization** with spaCy + POS filtering (`KEEP_POS`).
3. **Document filtering** (`MIN_CHARS` / `MAX_CHARS`, removal of empty documents).
4. **Deduplication by similarity** (MinHash LSH) if `GO_REMOVE_NEAR_DUPLICATES = True`.
5. **TF‑IDF + NMF** for each `k` in `TOPIC_LIST` (training on the deduplicated corpus).
6. **Topic quality evaluation** (coherences `c_v` and `c_npmi`).
7. **Score propagation** to near‑duplicates, handling of republications and fine‑grained deduplication.
8. **Export** of matrices and metric tables as CSV files.

At the end, a success message is printed and all results are stored in `RESULTS_ROOT/DATASET_NAME/config_XXXXXXXX/` (where `XXXXXXXX` is the configuration hash, see section 7).

---

## 6. Cache system

The script uses a **smart caching mechanism** to avoid reprocessing data when only modeling parameters change.

### How it works

When you run the script, preprocessing results (extraction + lemmatization) are cached in the `cache/` directory:

```text
cache/
└── preprocessing_XXXXXXXX.pkl
```

The cache key (8-character hash) is computed from:
- `SOURCE_TYPE` (europresse, istex, or csv)
- `LANGUAGE` (fr, en)
- `MIN_CHARS` and `MAX_CHARS`
- `KEEP_POS` (POS tags to keep)
- `FAST_LEMMATIZATION`
- Source file paths and their modification times

### Benefits

If you change only TF-IDF or NMF parameters (e.g., `TFIDF_MAX_DF`, `TOPIC_LIST`, `LSH_THRESHOLD`), the cache is reused and **extraction/lemmatization are skipped**, saving significant time on large corpora.

### Cache reset

To force reprocessing from scratch (e.g., after updating source files):

```bash
uv run python -m sources.topic_modeling_nmf --reset
```

This deletes the cache and regenerates all preprocessing data.

---

## 7. Organization of output files

For a dataset named `DATASET_NAME` (e.g. `"menopause"`) and a results folder `RESULTS_ROOT` (default `"results"`), the structure is:

```text
cache/
└── preprocessing_XXXXXXXX.pkl

results/
└── DATASET_NAME/
    └── config_XXXXXXXX/
        ├── config_params.json
        ├── topic_quality_summary_top{EVAL_TOP_N_WORDS}.csv
        ├── journal_min_max_dates.csv
        ├── 5t/
        │   ├── W_documents_topics.csv
        │   └── H_topics_terms.csv
        ├── 7t/
        │   ├── W_documents_topics.csv
        │   └── H_topics_terms.csv
        └── ...
```

### Configuration-specific folders

Results are organized by **configuration hash** to prevent overwriting when you change parameters. Each `config_XXXXXXXX/` folder corresponds to a unique combination of:
- TF-IDF parameters (`TFIDF_MAX_DF`, `TFIDF_MIN_DF`, `TFIDF_MAX_FEATURES`, `TFIDF_SUBLINEAR_TF`, `TFIDF_NORM`, `TFIDF_SMOOTH_IDF`)
- Lemmatization parameters (`LANGUAGE`, `KEEP_POS`, `FAST_LEMMATIZATION`)
- Document filtering (`MIN_CHARS`, `MAX_CHARS`)
- LSH deduplication settings (`GO_REMOVE_NEAR_DUPLICATES`, `LSH_THRESHOLD`, `LSH_NUM_PERM`, `LSH_MAX_TOKENS`)

This allows you to **compare different configurations** for the same dataset without losing previous results.

### Files in each configuration folder

- **`config_params.json`**: complete traceability file containing all parameters used for this configuration (TF-IDF, NMF, lemmatization, evaluation). Essential for reproducibility and comparison between runs.

- **`topic_quality_summary_top{EVAL_TOP_N_WORDS}.csv`**: global summary of quality metrics for each `k` in `TOPIC_LIST` (one row per number of topics). The filename includes the `EVAL_TOP_N_WORDS` parameter (e.g., `topic_quality_summary_top15.csv`).

- **`journal_min_max_dates.csv`**: for each journal, min/max publication dates in the corpus (useful for temporal analyses, generated for Europresse sources only).

- **For each `k` in `TOPIC_LIST`** (e.g., `5t/`, `7t/`):
  - `W_documents_topics.csv`: W matrix **enriched** with metadata (title, journal, date, etc. for Europresse/ISTEX; original CSV columns for `SOURCE_TYPE="csv"`) + `Topic_i` columns; includes `Main_topic_index` (dominant topic).
  - `H_topics_terms.csv`: compact H matrix, containing only the `N_TOP_WORDS` best terms per topic (two columns per topic: `Topic_k_Term`, `Topic_k_Weight`).
  - `topic_quality_config.csv` (optional, if `SAVE_TOPIC_LEVEL_METRICS = True`): detailed metrics for this specific `k`.

---

## 8. Interpreting the W and H matrices

- **W matrix (documents × topics)**:
  - each row = one document,
  - each column = one topic,
  - each value = weight of the topic in the document.
  - a `Main_topic_index` column indicates the dominant topic per document.

- **H matrix (topics × words)**:
  - each pair of columns `Topic_k_Term` / `Topic_k_Weight` contains the most important words for topic \(k\),
  - rows are ordered from the most to the least important word.

In practice, for qualitative analysis:
- use `H_topics_terms.csv` to **read the topics** (in the corresponding `{k}t/` folder, e.g., `5t/` for 5 topics);
- use `W_documents_topics.csv` to **see which documents carry which topics**, filtering by journal, date, etc.

---

## 9. Going further

- Change `TOPIC_LIST` and observe how metrics evolve in `topic_quality_summary_top{EVAL_TOP_N_WORDS}.csv`.
- Experiment with `TFIDF_MAX_DF`, `TFIDF_MIN_DF` and `TFIDF_MAX_FEATURES` to adjust topic granularity.
- Change `KEEP_POS` (e.g. add adjectives) to see the effect on topics.
- Disable deduplication (`GO_REMOVE_NEAR_DUPLICATES = False`) to understand its impact on results.

The low‑level code (extraction, deduplication, metrics) is intentionally factored and commented in `sources/` so it can serve as a **pedagogical support** if you want to dive into internal details.


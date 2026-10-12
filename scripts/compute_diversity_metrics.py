"""
Compute the lexical and semantic diversity metrics of the generated narratives and write them IN PLACE
into the results files.

Port of scripts/metrics-regression/compute_diversity_metrics.py (self_bleu and semantic_div per response
group) plus the per-response lexical metrics of scripts/metrics-regression/regression_utils.py
(min_distinct_n, trigram_repetition_rate), with the in-place file handling of extract_gender.py.

Columns written to every row (re-running overwrites them with the same values; nothing else changes):

  Per response group: all rows sharing model_key, profile_id, temperature and prompt_case, i.e. the 10
  'given' or 20 'assumed' samples of one prompt. The group value is repeated on each row of the group.
    self_bleu                lexical: BLEU (geometric mean over n = 2..5) of each response against the
                             rest of its group, averaged. HIGHER = more overlap = LESS diverse; the one
                             metric here whose direction is inverted.
    semantic_div             semantic: mean pairwise cosine distance of all-MiniLM-L6-v2 sentence
                             embeddings within the group. Higher = more varied.
  Per response:
    min_distinct_n           lexical: minimum distinct-n over n = 1..5, in practice the unigram
                             type-token ratio. Higher = more varied; falls with length by construction.
    trigram_repetition_rate  lexical: share of word trigrams that are repeats; values near 1 mean a
                             repetition loop (the degeneration audit flagged > 0.3). Null under 20 words.

A metric that cannot be computed (a group with fewer than 2 responses, a response too short) is null.

In-place handling, as in extract_gender.py: rows are read and written with the json module so every other
field round-trips exactly, and each file is rewritten atomically (temp file + os.replace). Run it only
once generation of a file is complete, and never while another script is rewriting the same file.

Embedding length: all-MiniLM-L6-v2 ships with max_seq_length 256 word pieces and silently drops the rest;
these narratives have a median of ~308, so at 256 about 98.5% are embedded from their opening ~83% only.
The default here is 512 (whole narratives). The semantic_div of the original thesis data was computed at
256: pass --max-seq-length 256 to reproduce those values.

Usage (from olmo_generation_pipeline/, after generation):
  python scripts/compute_diversity_metrics.py                     # given + assumed results files
  python scripts/compute_diversity_metrics.py --inputs data/results/olmo7b_temp_results_given.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

script_dir = Path(__file__).resolve().parent
root_dir = script_dir.parent                    # .py < scripts < olmo_generation_pipeline
sys.path.insert(0, str(root_dir))

from utils.results_io import finite_or_none, read_rows, write_rows_atomic  # noqa: E402

results_dir = root_dir / "data" / "results"
DEFAULT_INPUTS = [
    results_dir / "olmo7b_temp_results_given.jsonl",
    results_dir / "olmo7b_temp_results_assumed.jsonl",
]

# The bash script exports OLMO_MODEL_ROOT to the node-local model cache; fall back to a
# pipeline-level models/ dir so the script also runs unmodified on a laptop.
models_dir = Path(os.getenv("OLMO_MODEL_ROOT", str(root_dir / "models")))

ST_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_MAX_SEQ_LENGTH = 512

GROUP_METRICS = ("self_bleu", "semantic_div")
ROW_METRICS = ("min_distinct_n", "trigram_repetition_rate")

# ---------------------------------------------------------------------------
# Per-response lexical metrics (from scripts/metrics-regression/regression_utils.py)
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"\b\w+\b")
_DISTINCT_N_VALUES = range(1, 6)


def tokenize(text: str) -> list[str]:
    return _WORD_RE.findall(str(text).lower())


def distinct_n(tokens: list[str], n: int) -> float:
    if len(tokens) < n:
        return float("nan")
    ngrams = list(zip(*[tokens[i:] for i in range(n)]))
    return len(set(ngrams)) / len(ngrams)


def min_distinct_n(text: str) -> float:
    """Minimum distinct-n over n = 1..5, within a single response.

    Note the minimum is attained at n = 1 in essentially every real narrative
    (distinct-1 <= distinct-2 <= ...), so this is the unigram type-token ratio in
    practice. TTR falls with text length by construction, so it is length-sensitive.
    """
    tokens = tokenize(text)
    if not tokens:  # np.nanmin would warn on an all-NaN slice; the result is NaN either way
        return float("nan")
    return float(np.nanmin([distinct_n(tokens, n) for n in _DISTINCT_N_VALUES]))


def trigram_repetition_rate(text: str, n: int = 3, min_tokens: int = 20) -> float:
    """Share of n-grams in a response that are repeats.

    0 means every n-gram is unique; values approaching 1 indicate a repetition loop.
    Short responses return NaN rather than a misleadingly extreme value.
    """
    tokens = tokenize(text)
    if len(tokens) < max(min_tokens, n + 1):
        return float("nan")
    ngrams = list(zip(*[tokens[i:] for i in range(n)]))
    return 1.0 - len(set(ngrams)) / len(ngrams)


# ---------------------------------------------------------------------------
# Per-group metrics (from scripts/metrics-regression/compute_diversity_metrics.py)
# ---------------------------------------------------------------------------


def ensure_nltk_punkt() -> None:
    try:
        from nltk.tokenize import word_tokenize
        word_tokenize("test")
    except LookupError:
        import nltk
        print("NLTK punkt not found. Downloading...")
        try:
            nltk.download("punkt_tab")
        except Exception:
            nltk.download("punkt")


def self_bleu(texts: list[str], ns: list[int] = [2, 3, 4, 5]) -> float:
    """
    Average BLEU of each text against the other texts in the group.
    Higher = responses overlap more, i.e. more repetitive (less diverse).
    Returns NaN for groups of size < 2.
    """
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from nltk.tokenize import word_tokenize

    if len(texts) < 2:
        return float("nan")

    tokenized = [word_tokenize(t.lower()) for t in texts]
    smoothing = SmoothingFunction().method1
    scores_per_n: dict[int, list[float]] = {n: [] for n in ns}

    for i, hyp in enumerate(tokenized):
        refs = [t for j, t in enumerate(tokenized) if j != i]
        for n in ns:
            weights = tuple(1 / n for _ in range(n))
            score = sentence_bleu(refs, hyp, weights=weights, smoothing_function=smoothing)
            scores_per_n[n].append(score)

    mean_scores = [float(np.mean(scores_per_n[n])) for n in ns]
    # Geometric mean across n-gram orders
    return float(np.exp(np.mean(np.log(np.clip(mean_scores, 1e-10, None)))))


def semantic_diversity(model, texts: list[str], batch_size: int = 64) -> float:
    """
    Average pairwise cosine distance of sentence-transformer embeddings.
    Higher = responses are more semantically varied.
    Returns NaN for groups of size < 2.
    """
    from scipy.spatial.distance import pdist

    if len(texts) < 2:
        return float("nan")

    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=False)
    pairwise_distances = pdist(embeddings, metric="cosine")
    return float(np.mean(pairwise_distances))


_GROUP_CANDIDATES = ["model_key", "profile_id", "temperature", "prompt_case", "prompt_style"]


def detect_group_cols(rows: list[dict]) -> list[str]:
    """Generation-context columns present on every row."""
    return [c for c in _GROUP_CANDIDATES if all(c in row for row in rows)]


# ---------------------------------------------------------------------------
# In-place enrichment
# ---------------------------------------------------------------------------


def enrich_rows(rows: list[dict], group_cols: list[str], st_model, batch_size: int, name: str) -> dict:
    """Add/refresh the four metric keys on every row (mutates `rows`, changes nothing else)."""
    groups: dict[tuple, list[int]] = {}
    for i, row in enumerate(rows):
        groups.setdefault(tuple(row[c] for c in group_cols), []).append(i)

    sizes = Counter(len(idx) for idx in groups.values())
    print(f"[{name}] {len(rows)} rows in {len(groups)} groups by {group_cols}; group sizes {dict(sorted(sizes.items()))}")
    if len(sizes) > 1:
        print(f"[{name}] WARNING: groups differ in size. Is generation of this file complete?")

    started = time.time()
    n_singletons = 0
    for idx in tqdm(groups.values(), desc=name[:28], unit="group"):
        texts = [rows[i]["response"] for i in idx if isinstance(rows[i]["response"], str) and rows[i]["response"].strip()]
        if len(texts) < 2:
            n_singletons += 1
            values = {"self_bleu": float("nan"), "semantic_div": float("nan")}
        else:
            values = {
                "self_bleu": self_bleu(texts),
                "semantic_div": semantic_diversity(st_model, texts, batch_size),
            }
        for i in idx:
            for key in GROUP_METRICS:
                rows[i][key] = finite_or_none(values[key])

    for row in rows:
        row["min_distinct_n"] = finite_or_none(min_distinct_n(row["response"]))
        row["trigram_repetition_rate"] = finite_or_none(trigram_repetition_rate(row["response"]))
    elapsed = time.time() - started

    nulls = {key: sum(row[key] is None for row in rows) for key in GROUP_METRICS + ROW_METRICS}
    print(f"[{name}] {elapsed:.1f}s; null values per metric: {nulls}")
    return {
        "n_records": len(rows),
        "n_groups": len(groups),
        "group_sizes": {str(k): v for k, v in sorted(sizes.items())},
        "n_singleton_groups": n_singletons,
        "null_counts": nulls,
        "seconds": round(elapsed, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--inputs", type=Path, nargs="+", default=DEFAULT_INPUTS,
                        help="Results files to enrich in place "
                             "(default: data/results/olmo7b_temp_results_{given,assumed}.jsonl)")
    parser.add_argument("--group-by", type=str, default=None,
                        help="Comma-separated columns defining response groups "
                             "(default: those of model_key, profile_id, temperature, prompt_case, prompt_style on every row)")
    parser.add_argument("--max-seq-length", type=int, default=DEFAULT_MAX_SEQ_LENGTH,
                        help=f"Encoder sequence limit in word pieces (default {DEFAULT_MAX_SEQ_LENGTH}; "
                             f"256 reproduces the original thesis data)")
    parser.add_argument("--batch-size", type=int, default=64, help="Sentence-transformer encoding batch size")
    parser.add_argument("--device", type=str, default=None, help="cuda / cpu (default: auto)")
    args = parser.parse_args()

    # Validate every input before the model load, so a missing or wrong file fails fast
    rows_by_path = {path: read_rows(path) for path in args.inputs}
    group_cols_by_path = {}
    for path, rows in rows_by_path.items():
        if args.group_by:
            group_cols = [c.strip() for c in args.group_by.split(",")]
            missing = [c for c in group_cols if not all(c in row for row in rows)]
            if missing:
                raise ValueError(f"{path.name}: --group-by columns missing from some rows: {missing}")
        else:
            group_cols = detect_group_cols(rows)
            if not group_cols:
                raise ValueError(f"{path.name}: none of {_GROUP_CANDIDATES} is present on every row")
        group_cols_by_path[path] = group_cols

    ensure_nltk_punkt()  # self_bleu tokenises with nltk.word_tokenize

    import torch
    from sentence_transformers import SentenceTransformer

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading {ST_MODEL} on {device} (cache: {models_dir}) ...")
    st_model = SentenceTransformer(ST_MODEL, device=device, cache_folder=str(models_dir))
    default_seq_length = int(st_model.max_seq_length)
    st_model.max_seq_length = args.max_seq_length
    print(f"Encoder max_seq_length: {st_model.max_seq_length} word pieces (model default {default_seq_length})")

    for path, rows in rows_by_path.items():
        info = enrich_rows(rows, group_cols_by_path[path], st_model, args.batch_size, path.stem)
        write_rows_atomic(path, rows)
        print(f"[{path.stem}] {', '.join(GROUP_METRICS + ROW_METRICS)} written in place -> {path}")

        metadata_dir = path.parent / f"metrics_{path.stem}"
        metadata_dir.mkdir(parents=True, exist_ok=True)
        meta = {
            "input_path": str(path.resolve()),
            "in_place": True,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "group_by_cols": group_cols_by_path[path],
            **info,
            "columns": {"per_group": list(GROUP_METRICS), "per_response": list(ROW_METRICS)},
            "st_model": ST_MODEL,
            "st_max_seq_length": int(st_model.max_seq_length),
            "st_max_seq_length_default": default_seq_length,
            "self_bleu_ngram_orders": [2, 3, 4, 5],
            "min_distinct_n_orders": list(_DISTINCT_N_VALUES),
            "trigram_repetition_min_tokens": 20,
            "device": device,
        }
        meta_path = metadata_dir / "diversity_metadata.json"
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"[{path.stem}] metadata -> {meta_path}")


if __name__ == "__main__":
    main()

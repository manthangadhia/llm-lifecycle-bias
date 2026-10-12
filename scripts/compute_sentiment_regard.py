"""
Compute per-response sentiment and regard scores and write them IN PLACE into the results files.

Port of scripts/metrics-regression/compute_sentiment_regard.py (same models, sentence splitting and
aggregation) with the in-place file handling of extract_gender.py, instead of writing a separate
*_with_sentiment_regard.jsonl file. Each response is split into sentences, both classifiers run over all
sentences of a file (batched across the whole file), and the sentence scores are averaged per response.

Columns written to every row (re-running overwrites them with the same values; nothing else changes):
  n_sentences       number of sentences scored (sentence fragments of 20 characters or fewer are dropped)
  avg_sentiment     mean signed sentiment in [-1, 1] (siebert/sentiment-roberta-large-english:
                    +score for a POSITIVE sentence, -score for a NEGATIVE one)
  regard_<label>    mean probability of each sasha/regardv3 label: negative, neutral, other, positive
  regard_polarity   regard_positive - regard_negative
A response with no usable sentence gets n_sentences 0 and null scores.

In-place handling, as in extract_gender.py: rows are read and written with the json module so every other
field round-trips exactly, and each file is rewritten atomically (temp file + os.replace). Run it only
once generation of a file is complete, and never while another script is rewriting the same file.

Usage (from olmo_generation_pipeline/, after generation):
  python scripts/compute_sentiment_regard.py                     # given + assumed results files
  python scripts/compute_sentiment_regard.py --inputs data/results/olmo7b_temp_results_given.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch
from nltk.tokenize import sent_tokenize, word_tokenize
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

script_dir = Path(__file__).resolve().parent
root_dir = script_dir.parent                    # .py < scripts < olmo_generation_pipeline
sys.path.insert(0, str(root_dir))

from utils.results_io import finite_or_none, read_rows, write_rows_atomic  # noqa: E402

from dotenv import load_dotenv  # noqa: E402
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

results_dir = root_dir / "data" / "results"
DEFAULT_INPUTS = [
    results_dir / "olmo7b_temp_results_given.jsonl",
    results_dir / "olmo7b_temp_results_assumed.jsonl",
]

# The bash script exports OLMO_MODEL_ROOT to the node-local model cache; fall back to a
# pipeline-level models/ dir so the script also runs unmodified on a laptop.
models_dir = Path(os.getenv("OLMO_MODEL_ROOT", str(root_dir / "models")))

SENTIMENT_MODEL = "siebert/sentiment-roberta-large-english"
REGARD_MODEL = "sasha/regardv3"


def load_classification_pipeline(model_name: str, task: str, device_idx: int, pipeline_kwargs: dict, **extra):
    """Load a model/tokenizer from the shared models/ cache_dir, downloading on first use."""
    model_kwargs = {"cache_dir": models_dir, "token": HF_TOKEN}
    if "torch_dtype" in pipeline_kwargs:
        # transformers>=5 renamed torch_dtype to dtype (as in extract_gender.py): try the new name and fall back.
        try:
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name, dtype=pipeline_kwargs["torch_dtype"], **model_kwargs
            )
        except TypeError:
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name, torch_dtype=pipeline_kwargs["torch_dtype"], **model_kwargs
            )
    else:
        model = AutoModelForSequenceClassification.from_pretrained(model_name, **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=models_dir, token=HF_TOKEN)
    return pipeline(task, model=model, tokenizer=tokenizer, device=device_idx, **extra)


def ensure_nltk_punkt() -> None:
    try:
        word_tokenize("test")
        sent_tokenize("test sentence.")
    except LookupError:
        import nltk

        print("NLTK punkt not found. Downloading...")
        nltk.download("punkt")
        nltk.download("punkt_tab")


def extract_sentences(response) -> list[str]:
    """Split a response into sentences for "higher resolution" scoring.

    Falls back to splitting on newlines if the response comes back as a
    single long blob (e.g. the model didn't produce sentence-final
    punctuation), and drops very short fragments left by truncated generations.
    """
    if not isinstance(response, str) or not response.strip():
        return []

    sentences = sent_tokenize(response)

    if len(sentences) == 1 and len(sentences[0]) > 500:
        sentences = [s.strip() for s in response.split("\n") if s.strip()]

    return [s for s in sentences if len(s) > 20]


def normalise_sentiment(sentiment_score: dict) -> float:
    """Map a {label, score} dict to a signed score in [-1, 1]."""
    label = sentiment_score["label"]
    score = sentiment_score["score"]
    return -score if label.upper() == "NEGATIVE" else score


def build_sentence_index(responses: list) -> tuple[list[int], list[str]]:
    row_indices: list[int] = []
    sentence_texts: list[str] = []
    for row_idx, response in enumerate(responses):
        for sentence in extract_sentences(response):
            row_indices.append(row_idx)
            sentence_texts.append(sentence)
    return row_indices, sentence_texts


def run_pipeline_batched(pipe, texts: list[str], batch_size: int, desc: str) -> list:
    results = []
    for start in tqdm(range(0, len(texts), batch_size), desc=desc):
        chunk = texts[start : start + batch_size]
        results.extend(pipe(chunk, truncation=True, max_length=512))
    return results


def aggregate_sentiment(row_indices: list[int], sentiment_results: list[dict], n_rows: int) -> pd.Series:
    scores = [normalise_sentiment(r) for r in sentiment_results]
    grouped = pd.Series(scores, index=row_indices, dtype=float).groupby(level=0).mean()
    return grouped.reindex(range(n_rows))


def aggregate_regard(row_indices: list[int], regard_results: list[list[dict]], n_rows: int) -> tuple[pd.DataFrame, list[str]]:
    per_sentence = pd.DataFrame(
        [{item["label"].lower(): item["score"] for item in sentence_scores} for sentence_scores in regard_results],
        index=row_indices,
    )

    labels = sorted(per_sentence.columns.tolist())

    grouped = per_sentence.groupby(level=0).mean().reindex(range(n_rows))
    grouped.columns = [f"regard_{c}" for c in grouped.columns]

    pos_col = next((c for c in grouped.columns if "positive" in c), None)
    neg_col = next((c for c in grouped.columns if "negative" in c), None)
    if pos_col and neg_col:
        grouped["regard_polarity"] = grouped[pos_col] - grouped[neg_col]

    return grouped, labels


def score_rows(rows: list[dict], sentiment_pipe, regard_pipe, batch_size: int, name: str) -> dict:
    """Add/refresh n_sentences, avg_sentiment, regard_* on every row (mutates `rows`, changes nothing else)."""
    n_rows = len(rows)
    started = time.time()
    row_indices, sentence_texts = build_sentence_index([row["response"] for row in rows])
    print(f"[{name}] {len(sentence_texts)} sentences from {n_rows} responses")

    sentiment_results = run_pipeline_batched(sentiment_pipe, sentence_texts, batch_size, "Sentiment")
    regard_results = run_pipeline_batched(regard_pipe, sentence_texts, batch_size, "Regard")

    sentence_counts = pd.Series(row_indices, dtype="int64").value_counts().reindex(range(n_rows)).fillna(0)
    avg_sentiment = aggregate_sentiment(row_indices, sentiment_results, n_rows)
    regard_df, regard_labels = aggregate_regard(row_indices, regard_results, n_rows)
    # Fixed key order: the regard labels alphabetically, then the polarity
    regard_cols = [f"regard_{label}" for label in regard_labels] + (
        ["regard_polarity"] if "regard_polarity" in regard_df.columns else []
    )

    for i, row in enumerate(rows):
        row["n_sentences"] = int(sentence_counts.iloc[i])
        row["avg_sentiment"] = finite_or_none(avg_sentiment.iloc[i])
        for col in regard_cols:
            row[col] = finite_or_none(regard_df[col].iloc[i])
    elapsed = time.time() - started

    n_zero = sum(row["n_sentences"] == 0 for row in rows)
    print(f"[{name}] {elapsed:.1f}s; regard labels {regard_labels}; {n_zero} responses without a usable sentence")
    return {
        "n_records": n_rows,
        "n_sentences": len(sentence_texts),
        "n_rows_zero_sentences": n_zero,
        "regard_labels": regard_labels,
        "seconds": round(elapsed, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--inputs", type=Path, nargs="+", default=DEFAULT_INPUTS,
                        help="Results files to enrich in place "
                             "(default: data/results/olmo7b_temp_results_{given,assumed}.jsonl)")
    parser.add_argument("--batch-size", type=int, default=16, help="Sentences per classifier forward pass")
    parser.add_argument("--device", type=str, default=None, choices=["cuda", "cpu"])
    args = parser.parse_args()

    # Validate every input before the (slow) model loads, so a missing or wrong file fails fast
    rows_by_path = {path: read_rows(path) for path in args.inputs}

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device_idx = 0 if device == "cuda" else -1
    print(f"Using device: {device}")

    ensure_nltk_punkt()

    pipeline_kwargs = {"torch_dtype": torch.float16} if device == "cuda" else {}

    print(f"Loading sentiment model: {SENTIMENT_MODEL} (cache_dir={models_dir})")
    sentiment_pipe = load_classification_pipeline(
        SENTIMENT_MODEL, "sentiment-analysis", device_idx, pipeline_kwargs,
    )

    print(f"Loading regard model: {REGARD_MODEL} (cache_dir={models_dir})")
    regard_pipe = load_classification_pipeline(
        REGARD_MODEL, "text-classification", device_idx, pipeline_kwargs, top_k=None,
    )

    for path, rows in rows_by_path.items():
        info = score_rows(rows, sentiment_pipe, regard_pipe, args.batch_size, path.stem)
        write_rows_atomic(path, rows)
        print(f"[{path.stem}] n_sentences, avg_sentiment, regard_* written in place -> {path}")

        metadata_dir = path.parent / f"metrics_{path.stem}"
        metadata_dir.mkdir(parents=True, exist_ok=True)
        metadata = {
            "input_path": str(path.resolve()),
            "in_place": True,
            "created_at": datetime.now(timezone.utc).isoformat(),
            **info,
            "sentiment_model": SENTIMENT_MODEL,
            "regard_model": REGARD_MODEL,
            "batch_size": args.batch_size,
            "device": device,
        }
        metadata_path = metadata_dir / "sentiment_regard_metadata.json"
        with metadata_path.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        print(f"[{path.stem}] metadata -> {metadata_path}")


if __name__ == "__main__":
    main()

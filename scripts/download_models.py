"""Download the models the pipeline uses into a Hugging Face cache directory.

Groups (one per pipeline step):
  olmo     the four OLMo-3 7B variants used for generation (~15GB each)
  flan     google/flan-t5-xl, the gender classifier
  metrics  all-MiniLM-L6-v2 (semantic diversity), siebert/sentiment-roberta-large-english and
           sasha/regardv3 (sentiment and regard), plus the NLTK punkt tokenizer data in <cache_dir>/nltk_data

Usage (from olmo_generation_pipeline/):
  python scripts/download_models.py                                      # all groups -> $OLMO_MODEL_ROOT, else ./models
  python scripts/download_models.py --group olmo --cache_dir /path/to/models/olmo
"""
import argparse
import os
from pathlib import Path
from huggingface_hub import snapshot_download

from dotenv import load_dotenv
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")  # optional: all of these repos are public

root_dir = Path(__file__).resolve().parents[1]    # .py < scripts < olmo_generation_pipeline

MODEL_GROUPS = {
    "olmo": [
        "allenai/Olmo-3-1025-7B",
        "allenai/Olmo-3-7B-Instruct",
        "allenai/Olmo-3-7B-Instruct-DPO",
        "allenai/Olmo-3-7B-Instruct-SFT"
    ],
    "flan": [
        "google/flan-t5-xl",
    ],
    "metrics": [
        "sentence-transformers/all-MiniLM-L6-v2",
        "siebert/sentiment-roberta-large-english",
        "sasha/regardv3",
    ],
}

parser = argparse.ArgumentParser(description="Download the models used by the pipeline.")
parser.add_argument("--group",
                    nargs="+",
                    choices=list(MODEL_GROUPS),
                    default=list(MODEL_GROUPS),
                    help="Model groups to download (default: all).")
parser.add_argument("--cache_dir",
                    type=Path,
                    default=Path(os.getenv("OLMO_MODEL_ROOT", str(root_dir / "models"))),
                    help="Hugging Face cache directory to download into (default: $OLMO_MODEL_ROOT, else ./models).")
args = parser.parse_args()
cache_dir = args.cache_dir

for group in args.group:
    for model_name in MODEL_GROUPS[group]:
        print(f"Downloading {model_name}...")
        snapshot_download(
            model_name,
            cache_dir=cache_dir,
            local_files_only=False,  # download if missing
            token=HF_TOKEN
        )
    if group == "metrics":
        # word_tokenize (self-BLEU) and sent_tokenize (sentiment/regard) need punkt; jobs that run offline
        # point NLTK_DATA here
        import nltk
        nltk_dir = cache_dir / "nltk_data"
        print(f"Downloading NLTK punkt data to {nltk_dir}...")
        for resource in ("punkt", "punkt_tab"):
            nltk.download(resource, download_dir=str(nltk_dir))

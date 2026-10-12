#!/bin/bash
#SBATCH --job-name=olmo_analysis
#SBATCH --output=olmo_analysis_%j.out
#SBATCH --error=olmo_analysis_%j.err
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=16G
#SBATCH --ntasks=1
#SBATCH --time=06:00:00
#SBATCH --gpus=rtx_4090:1

# ============================================================================================
# TEMPLATE: written for ETH Euler (SLURM). One RTX 4090 as in the original runs; memory covers
# loading flan-t5-xl; the time limit is a generous allowance (self-BLEU on ~19,000 narratives is
# the slowest part, expect roughly 1-2 h in total). Adjust the job settings and the paths below
# for your own cluster.
# ============================================================================================
#
# Everything after generation, in one job, on data/results/olmo7b_temp_results_{given,assumed}.jsonl
# (the shipped responses for 15 occupations, or a fresh run of generate_narratives.py):
#   step 4  gender     extract_gender.py --case both: flan_* labels on the assumed narratives (the
#                      regression's inferred gender) + the baseline accuracy on the given ones
#   step 5  metrics    compute_diversity_metrics.py (self_bleu, semantic_div, min_distinct_n,
#                      trigram_repetition_rate), then compute_sentiment_regard.py (n_sentences,
#                      avg_sentiment, regard_*)
#   step 6  regression run_regression.sh all: given, assumed and pooled x the main specification
#                      and the four sensitivity variants -> data/results/regression/
# Steps 4 and 5 write their columns IN PLACE into the two results files (add-only, atomic); step 6
# only reads them. The steps run one after another, so no two of them rewrite a file at once.
#   cd olmo_generation_pipeline && sbatch scripts/run_analysis.sh
# Re-running the job recomputes every column and fit with the same values. The pronoun classifier
# (run_pronoun_extraction.sh) is not part of this job.

set -e  # fail fast if anything breaks

echo "Running on node: $(hostname)"
echo "Working directory: $(pwd)"

module load eth_proxy  # Euler-specific (internet access via proxy); remove on other clusters

# sbatch starts the job in the directory it was submitted from, which must be olmo_generation_pipeline/
PIPELINE_DIR=${SLURM_SUBMIT_DIR:-$(pwd)}
# Python virtual environment with requirements.txt installed
ENV_DIR=${ENV_DIR:-/cluster/scratch/$USER/occ_bias/env}

for FILE in scripts/extract_gender.py scripts/label_config.json scripts/compute_diversity_metrics.py scripts/compute_sentiment_regard.py \
            scripts/run_regression.sh data/results/olmo7b_temp_results_given.jsonl data/results/olmo7b_temp_results_assumed.jsonl; do
    if [ ! -f "$PIPELINE_DIR/$FILE" ]; then
        echo "ERROR: $FILE not found in $PIPELINE_DIR. Submit from the olmo_generation_pipeline/ directory (and generate first)."
        exit 1
    fi
done

# ---- 1. Model locations ----
# Filled by scripts/download_models.sh flan metrics
SHARED_FLAN=${SHARED_FLAN:-/cluster/scratch/$USER/models/flan}
SHARED_METRICS=${SHARED_METRICS:-/cluster/scratch/$USER/models/metrics}
# Paths on the local NVMe drive
LOCAL_FLAN=/tmp/flan/models
LOCAL_METRICS=/tmp/metrics/models

if [ ! -d "$SHARED_FLAN/models--google--flan-t5-xl" ]; then
    echo "ERROR: $SHARED_FLAN/models--google--flan-t5-xl not found (scratch purge?). Run scripts/download_models.sh flan first."
    exit 1
fi
for MODEL_DIR in "models--sentence-transformers--all-MiniLM-L6-v2" "models--siebert--sentiment-roberta-large-english" "models--sasha--regardv3" "nltk_data"; do
    if [ ! -d "$SHARED_METRICS/$MODEL_DIR" ]; then
        echo "ERROR: $SHARED_METRICS/$MODEL_DIR not found (scratch purge?). Run scripts/download_models.sh metrics first."
        exit 1
    fi
done

# ---- 2. Stage the models to local NVMe ----
# rsync is a no-op for a complete copy and finishes a partial one left behind by an interrupted job
mkdir -p $LOCAL_FLAN $LOCAL_METRICS
echo "Staging flan-t5-xl from $SHARED_FLAN to $LOCAL_FLAN..."
rsync -a --info=progress2 --human-readable $SHARED_FLAN/models--google--flan-t5-xl/ $LOCAL_FLAN/models--google--flan-t5-xl/
echo "Staging metric models from $SHARED_METRICS to $LOCAL_METRICS..."
rsync -a --info=progress2 --human-readable $SHARED_METRICS/ $LOCAL_METRICS/

echo "Model staging complete"

# ---- 3. Environment ----
source $ENV_DIR/bin/activate
cd $PIPELINE_DIR

echo "Activated venv ..."

# The scripts read OLMO_MODEL_ROOT as their model cache (set per step below); HF_HOME is belt-and-braces.
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
# NLTK punkt (word_tokenize for self-BLEU, sent_tokenize for sentiment/regard)
export NLTK_DATA=$LOCAL_METRICS/nltk_data

# ---- 4. Gender (flan cache) ----
echo "==== step 4: gender labels (flan-t5-xl)"
export HF_HOME=$LOCAL_FLAN HF_HUB_CACHE=$LOCAL_FLAN OLMO_MODEL_ROOT=$LOCAL_FLAN
python scripts/extract_gender.py --case both --batch_size 16

# ---- 5. Metrics (metrics cache) ----
export HF_HOME=$LOCAL_METRICS HF_HUB_CACHE=$LOCAL_METRICS OLMO_MODEL_ROOT=$LOCAL_METRICS
echo "==== step 5a: diversity metrics"
python scripts/compute_diversity_metrics.py
echo "==== step 5b: sentiment and regard"
python scripts/compute_sentiment_regard.py

# ---- 6. Regression (CPU) ----
echo "==== step 6: regressions"
bash scripts/run_regression.sh all

# ---- 7. Short summary ----
python - <<'EOF'
import json
from pathlib import Path
import pandas as pd

results = Path("data/results")
expected = ["flan_gender", "self_bleu", "semantic_div", "min_distinct_n", "trigram_repetition_rate",
            "n_sentences", "avg_sentiment", "regard_polarity"]
for case in ("given", "assumed"):
    rows = [json.loads(l) for l in open(results / f"olmo7b_temp_results_{case}.jsonl", encoding="utf-8")]
    missing = sorted({k for r in rows for k in expected if k not in r})
    print(f"{case}: {len(rows)} rows, {len({r['occupation'] for r in rows})} occupations; "
          + (f"MISSING columns on some rows: {missing}" if missing else "all step 4/5 columns present"))
baseline = json.load(open(results / "flan_given_baseline.json"))["baseline"]
print(f"flan baseline on the given narratives: accuracy {baseline['accuracy']:.3f}, abstention {baseline['abstention_rate']:.3f}")
reg = results / "regression"
for case in ("given", "assumed", "pooled"):
    meta = json.load(open(reg / case / f"{case}_enriched_metadata.json"))
    print(f"regression {case}: {meta['n_groups_fitted']} groups, {meta.get('n_clusters')} clusters")
a = json.load(open(reg / "assumed" / "assumed_enriched_metadata.json"))["model_a_prime"]
print(f"Model A': {a['nobs']} responses, pseudo-R2 {a['prsquared']:.3f}")
print("pooled scenario x gender x stage joint tests:")
print(pd.read_csv(reg / "pooled" / "pooled_enriched_regression_joint_tests.csv")[["metric", "fvalue", "pvalue", "df_num", "df_denom"]].to_string(index=False))
EOF

echo "Done. Columns are in place in data/results/olmo7b_temp_results_{given,assumed}.jsonl; fits in data/results/regression/"

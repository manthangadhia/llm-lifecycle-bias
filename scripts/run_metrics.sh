#!/bin/bash
#SBATCH --job-name=olmo_metrics
#SBATCH --output=olmo_metrics_%j.out
#SBATCH --error=olmo_metrics_%j.err
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=6144
#SBATCH --ntasks=1
#SBATCH --time=04:00:00
#SBATCH --gpus=rtx_4090:1

# ============================================================================================
# TEMPLATE: written for ETH Euler (SLURM). CPUs and memory are those of the original metrics job
# (2 x 6GB); the time limit is raised to cover both metric scripts on both 10,240-row files.
# Self-BLEU is CPU-bound; the GPU serves the sentence embeddings and the sentiment/regard
# classifiers. Adjust the job settings and the paths below for your own cluster.
# ============================================================================================
#
# Writes the NLG metrics IN PLACE into data/results/olmo7b_temp_results_{given,assumed}.jsonl:
#   cd olmo_generation_pipeline && sbatch scripts/run_metrics.sh             # both (the default)
#   cd olmo_generation_pipeline && sbatch scripts/run_metrics.sh diversity   # self_bleu, semantic_div, min_distinct_n,
#                                                                            # trigram_repetition_rate
#   cd olmo_generation_pipeline && sbatch scripts/run_metrics.sh sentiment   # n_sentences, avg_sentiment, regard_*
# Any further arguments are passed on to the script(s) that run, e.g. other files:
#   sbatch scripts/run_metrics.sh all --inputs <file>.jsonl
# (with 'all', pass only arguments both scripts accept: --inputs, --device).
#
# Submit only once generation is COMPLETE and while no other job rewrites the same files: the files
# are rewritten in place. To chain after another job: sbatch --dependency=afterok:<jobid> ...

set -e  # fail fast if anything breaks

echo "Running on node: $(hostname)"
echo "Working directory: $(pwd)"

WHICH=${1:-all}
if [ "$WHICH" != "all" ] && [ "$WHICH" != "diversity" ] && [ "$WHICH" != "sentiment" ]; then
    echo "ERROR: usage: sbatch scripts/run_metrics.sh [all|diversity|sentiment] [extra args]"
    exit 1
fi
echo "Metrics: $WHICH"

module load eth_proxy  # Euler-specific (internet access via proxy); remove on other clusters

# sbatch starts the job in the directory it was submitted from, which must be olmo_generation_pipeline/
PIPELINE_DIR=${SLURM_SUBMIT_DIR:-$(pwd)}
# Python virtual environment with requirements.txt installed
ENV_DIR=${ENV_DIR:-/cluster/scratch/$USER/occ_bias/env}

if [ ! -f "$PIPELINE_DIR/scripts/compute_diversity_metrics.py" ]; then
    echo "ERROR: scripts/compute_diversity_metrics.py not found in $PIPELINE_DIR. Submit from the olmo_generation_pipeline/ directory."
    exit 1
fi

# ---- 1. Model locations ----
# Path where the metric models are stored on shared scratch (filled by scripts/download_models.sh metrics)
SHARED_METRICS=${SHARED_METRICS:-/cluster/scratch/$USER/models/metrics}
# Path on the local NVMe drive
LOCAL_METRICS=/tmp/metrics/models

for MODEL_DIR in "models--sentence-transformers--all-MiniLM-L6-v2" "models--siebert--sentiment-roberta-large-english" "models--sasha--regardv3" "nltk_data"; do
    if [ ! -d "$SHARED_METRICS/$MODEL_DIR" ]; then
        echo "ERROR: $SHARED_METRICS/$MODEL_DIR not found (scratch purge?). Run scripts/download_models.sh metrics first."
        exit 1
    fi
done

# ---- 2. Stage the models to local NVMe ----
mkdir -p $LOCAL_METRICS
echo "Staging metric models from $SHARED_METRICS to $LOCAL_METRICS..."
# rsync is a no-op for a complete copy and finishes a partial one left behind by an interrupted job
rsync -a --info=progress2 --human-readable $SHARED_METRICS/ $LOCAL_METRICS/

echo "Model staging complete"

# ---- 3. Activate environment ----
source $ENV_DIR/bin/activate
cd $PIPELINE_DIR

echo "Activated venv ..."

# ---- 4. Bridging variables ----
# The metric scripts read OLMO_MODEL_ROOT as their cache_dir; HF_HOME is belt-and-braces.
export HF_HOME=$LOCAL_METRICS
export HF_HUB_CACHE=$LOCAL_METRICS
export HF_HUB_OFFLINE=1
export OLMO_MODEL_ROOT=$LOCAL_METRICS
# NLTK punkt (word_tokenize for self-BLEU, sent_tokenize for sentiment/regard)
export NLTK_DATA=$LOCAL_METRICS/nltk_data
export PYTHONUNBUFFERED=1

echo "Set environment variables, running script(s) ..."

# ---- 5. Run ----
# Sequential on purpose: both scripts rewrite the same files.
if [ "$WHICH" = "all" ] || [ "$WHICH" = "diversity" ]; then
    python scripts/compute_diversity_metrics.py "${@:2}"
fi
if [ "$WHICH" = "all" ] || [ "$WHICH" = "sentiment" ]; then
    python scripts/compute_sentiment_regard.py "${@:2}"
fi

echo "Done ($WHICH). Metrics are in place in data/results/; run metadata in data/results/metrics_<file>/"

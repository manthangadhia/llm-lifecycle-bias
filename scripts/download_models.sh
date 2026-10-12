#!/bin/bash
#SBATCH --job-name=download_models
#SBATCH --output=model_download_%j.out
#SBATCH --error=model_download_%j.err
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=10240
#SBATCH --ntasks=1
#SBATCH --time=04:00:00

# ============================================================================================
# TEMPLATE: written for ETH Euler (SLURM). The #SBATCH resources above are the ones used for the
# original download (CPU only, no GPU needed). Adjust the job settings and the paths below for
# your own cluster.
# ============================================================================================
#
# Downloads the models of each pipeline step into its own folder on shared scratch, where the other
# job scripts pick them up:
#   olmo     the four OLMo-3 7B variants (~15GB each)    -> $SHARED_MODELS   (run_generation.sh, smoke_test.sh)
#   flan     google/flan-t5-xl                            -> $SHARED_FLAN     (run_gender_extraction.sh, run_pronoun_extraction.sh)
#   metrics  MiniLM, sentiment and regard models + NLTK   -> $SHARED_METRICS  (run_metrics.sh)
# Submit from olmo_generation_pipeline/; with no arguments all three groups are downloaded:
#   cd olmo_generation_pipeline && sbatch scripts/download_models.sh
#   cd olmo_generation_pipeline && sbatch scripts/download_models.sh flan metrics

set -e  # fail fast if anything breaks

echo "Running on node: $(hostname)"
echo "Working directory: $(pwd)"

module load eth_proxy  # Euler-specific (internet access via proxy); remove on other clusters

# sbatch starts the job in the directory it was submitted from, which must be olmo_generation_pipeline/
PIPELINE_DIR=${SLURM_SUBMIT_DIR:-$(pwd)}
# Python virtual environment with requirements.txt installed
ENV_DIR=${ENV_DIR:-/cluster/scratch/$USER/occ_bias/env}
# Where the models are stored (shared scratch), one folder per group
SHARED_MODELS=${SHARED_MODELS:-/cluster/scratch/$USER/models/olmo}
SHARED_FLAN=${SHARED_FLAN:-/cluster/scratch/$USER/models/flan}
SHARED_METRICS=${SHARED_METRICS:-/cluster/scratch/$USER/models/metrics}

declare -A GROUP_DIRS
GROUP_DIRS["olmo"]=$SHARED_MODELS
GROUP_DIRS["flan"]=$SHARED_FLAN
GROUP_DIRS["metrics"]=$SHARED_METRICS

SELECTED=("$@")
if [ ${#SELECTED[@]} -eq 0 ]; then
    SELECTED=("olmo" "flan" "metrics")
fi
for GROUP in "${SELECTED[@]}"; do
    if [ -z "${GROUP_DIRS[$GROUP]}" ]; then
        echo "ERROR: unknown group '$GROUP'. Usage: sbatch scripts/download_models.sh [olmo] [flan] [metrics]"
        exit 1
    fi
done

source $ENV_DIR/bin/activate
cd $PIPELINE_DIR

echo "Activated env ..."

for GROUP in "${SELECTED[@]}"; do
    python scripts/download_models.py --group "$GROUP" --cache_dir "${GROUP_DIRS[$GROUP]}"
    echo "Downloaded $GROUP models to ${GROUP_DIRS[$GROUP]} ..."
done

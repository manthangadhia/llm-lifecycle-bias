#!/bin/bash
#SBATCH --job-name=olmo_generation
#SBATCH --output=olmo_generation_%j.out
#SBATCH --error=olmo_generation_%j.err
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=30720         # 30GB
#SBATCH --ntasks=1
#SBATCH --time=40:00:00
#SBATCH --gpus=rtx_4090:1

# ============================================================================================
# TEMPLATE: written for ETH Euler (SLURM). The #SBATCH resources above are the ones used for the
# original runs: one RTX 4090 (24GB) per job, ONE prompt case per job, all four OLMo variants in
# series. Adjust the job settings and the paths below for your own cluster.
# ============================================================================================
#
# Narratives from all four OLMo variants, run in series (base -> sft -> dpo -> rlvr) at all 4
# temperatures, entropy tracked. ONE prompt case per job:
#   cd olmo_generation_pipeline && sbatch scripts/run_generation.sh given
#   cd olmo_generation_pipeline && sbatch scripts/run_generation.sh assumed
# Each case writes data/results/olmo7b_temp_results_<case>.jsonl holding all four variants
# (10,240 rows when complete for all 16 occupations).
#
# Any further arguments are passed on to generate_narratives.py, e.g. for a subset of occupations
# whose prompt files were built with scripts/generate_prompts.py --occupations ... --out_dir ...:
#   sbatch scripts/run_generation.sh given --prompts_dir data/gender_prompts_new_occupations --output_tag new_occ
#
# Time: the original 40h limit was for the 13 original occupations; all 16 is ~23% more work.
# The job RESUMES: if it runs out of time, submit the same command again. It keeps every
# finished prompt, redoes the one that was in progress and carries on; a finished case is left
# untouched. To regenerate a case from scratch, delete its file first.

set -e  # fail fast if anything breaks

echo "Running on node: $(hostname)"
echo "Working directory: $(pwd)"

CASE=${1:-}
if [ "$CASE" != "given" ] && [ "$CASE" != "assumed" ]; then
    echo "ERROR: usage: sbatch scripts/run_generation.sh <given|assumed> [extra args]"
    exit 1
fi
echo "Prompt case: $CASE"

module load eth_proxy  # Euler-specific (internet access via proxy); remove on other clusters

# sbatch starts the job in the directory it was submitted from, which must be olmo_generation_pipeline/
PIPELINE_DIR=${SLURM_SUBMIT_DIR:-$(pwd)}
# Python virtual environment with requirements.txt installed
ENV_DIR=${ENV_DIR:-/cluster/scratch/$USER/occ_bias/env}

if [ ! -f "$PIPELINE_DIR/scripts/generate_narratives.py" ]; then
    echo "ERROR: scripts/generate_narratives.py not found in $PIPELINE_DIR. Submit from the olmo_generation_pipeline/ directory."
    exit 1
fi

# ---- 1. Define paths ----
KEYS=("base" "sft" "dpo" "rlvr")

declare -A MODEL_DIRS
MODEL_DIRS["base"]="models--allenai--Olmo-3-1025-7B"
MODEL_DIRS["sft"]="models--allenai--Olmo-3-7B-Instruct-SFT"
MODEL_DIRS["dpo"]="models--allenai--Olmo-3-7B-Instruct-DPO"
MODEL_DIRS["rlvr"]="models--allenai--Olmo-3-7B-Instruct"

# Path where models are stored on shared scratch (filled by scripts/download_models.sh)
SHARED_MODELS=${SHARED_MODELS:-/cluster/scratch/$USER/models/olmo}
# Path on the local NVMe drive (holds all four variants for the whole run)
LOCAL_MODELS=/tmp/olmo/models

# Check all four are there before spending time on staging
for KEY in "${KEYS[@]}"; do
    if [ ! -d "$SHARED_MODELS/${MODEL_DIRS[$KEY]}" ]; then
        echo "ERROR: $SHARED_MODELS/${MODEL_DIRS[$KEY]} not found (scratch purge?). Run scripts/download_models.sh first."
        exit 1
    fi
done

# ---- 2. Stage all four models to local NVMe ----
mkdir -p $LOCAL_MODELS
for KEY in "${KEYS[@]}"; do
    MODEL_DIR=${MODEL_DIRS[$KEY]}
    echo "Staging model '$KEY' from $SHARED_MODELS to $LOCAL_MODELS..."
    # rsync is a no-op for a complete copy and finishes a partial one left behind by an interrupted job
    rsync -a --info=progress2 --human-readable $SHARED_MODELS/$MODEL_DIR/ $LOCAL_MODELS/$MODEL_DIR/
done

echo "Model staging complete"

# ---- 3. Environment & execution ----
source $ENV_DIR/bin/activate
cd $PIPELINE_DIR

echo "Activated venv ..."

# ---- 4. Bridging variables ----
# Set the Hugging Face cache home to the local NVMe path
# so the library looks there first
export HF_HOME=$LOCAL_MODELS
export HF_HUB_CACHE=$LOCAL_MODELS
export HF_HUB_OFFLINE=1
export OLMO_MODEL_ROOT=$LOCAL_MODELS
export PYTHONUNBUFFERED=1

echo "Set environment variables, running script ..."
# ---- 5. Run script ----
# No --model_key: the script loops over all four variants in the order above, into one file for this
# case. --resume continues an earlier run of the same case (and starts fresh if there is none).
python scripts/generate_narratives.py --prompt_case "$CASE" --resume "${@:2}"

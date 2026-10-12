#!/bin/bash
#SBATCH --job-name=flan_pronouns
#SBATCH --output=flan_pronouns_%j.out
#SBATCH --error=flan_pronouns_%j.err
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=24G
#SBATCH --ntasks=1
#SBATCH --time=03:00:00
#SBATCH --gpus=rtx_4090:1

# ============================================================================================
# TEMPLATE: written for ETH Euler (SLURM). The #SBATCH resources above are the ones used for the
# original full-size flan labelling run. Adjust the job settings and the paths below for your own
# cluster.
# ============================================================================================
#
# Pronoun classifier (He / She / They / unclear) with flan-t5-xl, to compare against the gender classifier
# from run_gender_extraction.sh. Choose what to run:
#   cd olmo_generation_pipeline && sbatch scripts/run_pronoun_extraction.sh assumed
#       Pronoun labels for the assumed narratives, IN PLACE: adds flan_pron_* columns to
#       data/results/olmo7b_temp_results_assumed.jsonl; the existing flan_* columns are not touched.
#   cd olmo_generation_pipeline && sbatch scripts/run_pronoun_extraction.sh given
#       The comparison on the given narratives: this classifier and the existing one are both scored on
#       the same rows and compared with the gender the prompt stated, with a paired test. The given rows
#       ALSO get the flan_pron_* columns written in place (add-only: `gender` and any flan_* are untouched).
#   cd olmo_generation_pipeline && sbatch scripts/run_pronoun_extraction.sh
#       Both (the default). When the assumed file already has the existing flan_gender labels
#       (run_gender_extraction.sh assumed), they are compared with the pronoun labels too.
# Everything is printed in the job log and written to data/results/flan_pronoun_report.json.
# Any further arguments are passed on to extract_gender_pronouns.py.
#
# Submit only once generation is COMPLETE and while no other job rewrites the same files: the files
# are rewritten in place. To chain after another job: sbatch --dependency=afterok:<jobid> ...

set -e # fail fast if anything breaks

echo "Running on node: $(hostname)"
echo "Working directory: $(pwd)"

CASE=${1:-both}
if [ "$CASE" != "both" ] && [ "$CASE" != "assumed" ] && [ "$CASE" != "given" ]; then
    echo "ERROR: usage: sbatch scripts/run_pronoun_extraction.sh [assumed|given|both] [extra args]"
    exit 1
fi
echo "Case: $CASE"

module load eth_proxy  # Euler-specific (internet access via proxy); remove on other clusters

# sbatch starts the job in the directory it was submitted from, which must be olmo_generation_pipeline/
PIPELINE_DIR=${SLURM_SUBMIT_DIR:-$(pwd)}
# Python virtual environment with requirements.txt installed
ENV_DIR=${ENV_DIR:-/cluster/scratch/$USER/occ_bias/env}

if [ ! -f "$PIPELINE_DIR/scripts/extract_gender_pronouns.py" ]; then
    echo "ERROR: scripts/extract_gender_pronouns.py not found in $PIPELINE_DIR. Submit from the olmo_generation_pipeline/ directory."
    exit 1
fi

# ---- 1. Model locations ----
# Path where the flan model is stored on shared scratch (filled by scripts/download_models.sh flan)
SHARED_FLAN=${SHARED_FLAN:-/cluster/scratch/$USER/models/flan}
# Path on the local NVMe drive
LOCAL_FLAN=/tmp/flan/models

if [ ! -d "$SHARED_FLAN/models--google--flan-t5-xl" ]; then
    echo "ERROR: $SHARED_FLAN/models--google--flan-t5-xl not found (scratch purge?). Run scripts/download_models.sh flan first."
    exit 1
fi

# ---- 2. Stage the model to local NVMe ----
mkdir -p $LOCAL_FLAN
echo "Staging flan-t5-xl from $SHARED_FLAN to $LOCAL_FLAN..."
# rsync is a no-op for a complete copy and finishes a partial one left behind by an interrupted job
rsync -a --info=progress2 --human-readable $SHARED_FLAN/models--google--flan-t5-xl/ $LOCAL_FLAN/models--google--flan-t5-xl/

echo "Model staging complete"

# ---- 3. Activate environment ----
source $ENV_DIR/bin/activate
cd $PIPELINE_DIR

echo "Activated venv ..."

# ---- 4. Bridging variables ----
# extract_gender_pronouns.py reads OLMO_MODEL_ROOT as its cache_dir; HF_HOME is belt-and-braces.
export HF_HOME=$LOCAL_FLAN
export HF_HUB_CACHE=$LOCAL_FLAN
export HF_HUB_OFFLINE=1
export OLMO_MODEL_ROOT=$LOCAL_FLAN
export PYTHONUNBUFFERED=1

echo "Set environment variables, running script ..."

# ---- 5. Run ----
# One invocation: flan-t5-xl loads once and serves both classifiers.
python scripts/extract_gender_pronouns.py --case "$CASE" --batch_size 16 "${@:2}"

echo "Done ($CASE). Report in data/results/flan_pronoun_report.json"

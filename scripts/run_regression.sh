#!/bin/bash
#SBATCH --job-name=olmo_regression
#SBATCH --output=olmo_regression_%j.out
#SBATCH --error=olmo_regression_%j.err
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4096
#SBATCH --ntasks=1
#SBATCH --time=01:00:00

# ============================================================================================
# TEMPLATE: written for ETH Euler (SLURM). The original regressions ran on a laptop CPU; the
# resources above are a generous allowance for all 15 fits (CPU only, no GPU, no network).
# Adjust the job settings and the paths below for your own cluster. The script also runs
# without SLURM: `bash scripts/run_regression.sh` (set PYTHON if `python` is not your env).
# ============================================================================================
#
# Fits the regressions on data/results/olmo7b_temp_results_{given,assumed}.jsonl once gender
# (run_gender_extraction.sh) and the metrics (run_metrics.sh) are in place:
#   cd olmo_generation_pipeline && sbatch scripts/run_regression.sh            # all specifications (default)
#   cd olmo_generation_pipeline && sbatch scripts/run_regression.sh main       # main specification only
# Each specification is fitted for the given, assumed and pooled cases and written to
# data/results/regression/<case>[_<specification>]/:
#   main              the paper's models (linear z-scored temperature, SEs clustered on profile)
#   nocluster         --cluster none                     (conventional SEs)
#   norepetition      --drop-repetition-loops 0.3        (repetition-loop responses removed first)
#   temp_dummy        --temperature-spec dummy           (temperature as categories)
#   temp_interaction  --temperature-spec interaction     (temperature x alignment stage)
# Any further arguments are passed on to every fit, e.g. the pronoun labels or other files:
#   sbatch scripts/run_regression.sh all --assumed-label-col flan_pron_gender
#   sbatch scripts/run_regression.sh all --given-input <given>.jsonl --assumed-input <assumed>.jsonl

set -e  # fail fast if anything breaks

echo "Running on node: $(hostname)"
echo "Working directory: $(pwd)"

WHICH=${1:-all}
if [ "$WHICH" != "all" ] && [ "$WHICH" != "main" ]; then
    echo "ERROR: usage: sbatch scripts/run_regression.sh [all|main] [extra args]"
    exit 1
fi
echo "Specifications: $WHICH"

# sbatch starts the job in the directory it was submitted from, which must be olmo_generation_pipeline/
PIPELINE_DIR=${SLURM_SUBMIT_DIR:-$(pwd)}
# Python virtual environment with requirements.txt installed (activated only inside a SLURM job;
# locally, activate your environment first or set PYTHON)
ENV_DIR=${ENV_DIR:-/cluster/scratch/$USER/occ_bias/env}
PYTHON=${PYTHON:-python}
# Where the fits are written
REG_DIR=${REG_DIR:-data/results/regression}

if [ ! -f "$PIPELINE_DIR/scripts/compute_regressions.py" ]; then
    echo "ERROR: scripts/compute_regressions.py not found in $PIPELINE_DIR. Submit from the olmo_generation_pipeline/ directory."
    exit 1
fi

if [ -n "$SLURM_JOB_ID" ]; then
    source $ENV_DIR/bin/activate
    echo "Activated venv ..."
fi
cd $PIPELINE_DIR
export PYTHONUNBUFFERED=1

declare -A SPEC_FLAGS
SPEC_FLAGS["main"]=""
SPEC_FLAGS["nocluster"]="--cluster none"
SPEC_FLAGS["norepetition"]="--drop-repetition-loops 0.3"
SPEC_FLAGS["temp_dummy"]="--temperature-spec dummy"
SPEC_FLAGS["temp_interaction"]="--temperature-spec interaction"

if [ "$WHICH" = "main" ]; then
    SPECS=("main")
else
    SPECS=("main" "nocluster" "norepetition" "temp_dummy" "temp_interaction")
fi

for SPEC in "${SPECS[@]}"; do
    for CASE in "given" "assumed" "pooled"; do
        OUT="$REG_DIR/$CASE"
        if [ "$SPEC" != "main" ]; then
            OUT="${OUT}_$SPEC"
        fi
        echo "==== $CASE / $SPEC -> $OUT"
        # SPEC_FLAGS is unquoted on purpose: each entry is a flag and its value
        $PYTHON scripts/compute_regressions.py --prompt-case "$CASE" ${SPEC_FLAGS[$SPEC]} --output-dir "$OUT" "${@:2}"
    done
done

echo "Done ($WHICH). Fits in $REG_DIR/"

#!/bin/bash
#SBATCH --job-name=olmo_smoke
#SBATCH --output=olmo_smoke_%j.out
#SBATCH --error=olmo_smoke_%j.err
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=24G
#SBATCH --ntasks=1
#SBATCH --time=01:00:00
#SBATCH --gpus=rtx_4090:1

# ============================================================================================
# TEMPLATE: written for ETH Euler (SLURM). The #SBATCH resources above are the ones used for the
# original smoke test. Adjust the job settings and the paths below for your own cluster.
# ============================================================================================
#
# Quick end-to-end check of every step before the full runs:
#   1. generation: 1 prompt per case, T=0.7 only, for the base model (base prompts) and rlvr
#      (instruct prompts) -> 4 files (10 rows per given file, 20 per assumed file)
#   2. gender: flan labels (+ baseline on the given files), then the pronoun variant, in place
#   3. metrics: diversity, then sentiment/regard, in place
# Everything goes to data/results/smoketest/, the real result files are not touched.
#   cd olmo_generation_pipeline && sbatch scripts/smoke_test.sh

set -e  # fail fast if anything breaks

echo "Running on node: $(hostname)"
echo "Working directory: $(pwd)"

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
# Where the models are stored on shared scratch (filled by scripts/download_models.sh)
SHARED_OLMO=${SHARED_MODELS:-/cluster/scratch/$USER/models/olmo}
SHARED_FLAN=${SHARED_FLAN:-/cluster/scratch/$USER/models/flan}
SHARED_METRICS=${SHARED_METRICS:-/cluster/scratch/$USER/models/metrics}
# Paths on the local NVMe drive
LOCAL_OLMO=/tmp/olmo/models
LOCAL_FLAN=/tmp/flan/models
LOCAL_METRICS=/tmp/metrics/models

declare -A MODEL_DIRS
MODEL_DIRS["base"]="models--allenai--Olmo-3-1025-7B"
MODEL_DIRS["rlvr"]="models--allenai--Olmo-3-7B-Instruct"

# ---- 2. Stage the models to local NVMe ----
mkdir -p $LOCAL_OLMO
for KEY in "base" "rlvr"; do
    MODEL_DIR=${MODEL_DIRS[$KEY]}
    if [ ! -d "$SHARED_OLMO/$MODEL_DIR" ]; then
        echo "ERROR: $SHARED_OLMO/$MODEL_DIR not found (scratch purge?). Run scripts/download_models.sh first."
        exit 1
    fi
    if [ ! -d "$LOCAL_OLMO/$MODEL_DIR" ]; then
        echo "Staging model '$KEY' to local NVMe..."
        rsync -a --info=progress2 --human-readable $SHARED_OLMO/$MODEL_DIR/ $LOCAL_OLMO/$MODEL_DIR/
    else
        echo "Model '$KEY' already staged on this node. Skipping rsync."
    fi
done

if [ ! -d "$SHARED_FLAN/models--google--flan-t5-xl" ]; then
    echo "ERROR: $SHARED_FLAN/models--google--flan-t5-xl not found (scratch purge?). Run scripts/download_models.sh flan first."
    exit 1
fi
mkdir -p $LOCAL_FLAN
echo "Staging flan-t5-xl to local NVMe..."
rsync -a --info=progress2 --human-readable $SHARED_FLAN/models--google--flan-t5-xl/ $LOCAL_FLAN/models--google--flan-t5-xl/

if [ ! -d "$SHARED_METRICS/nltk_data" ]; then
    echo "ERROR: $SHARED_METRICS/nltk_data not found (scratch purge?). Run scripts/download_models.sh metrics first."
    exit 1
fi
mkdir -p $LOCAL_METRICS
echo "Staging metric models to local NVMe..."
rsync -a --info=progress2 --human-readable $SHARED_METRICS/ $LOCAL_METRICS/

echo "Model staging complete"

# ---- 3. Environment & execution ----
source $ENV_DIR/bin/activate
cd $PIPELINE_DIR

echo "Activated venv ..."

# ---- 4. Bridging variables (the model cache is set per step below) ----
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
export NLTK_DATA=$LOCAL_METRICS/nltk_data

SMOKE=data/results/smoketest
ASSUMED=("$SMOKE/olmo7b_temp_results_assumed_smoketest_base.jsonl" "$SMOKE/olmo7b_temp_results_assumed_smoketest_rlvr.jsonl")
GIVEN=("$SMOKE/olmo7b_temp_results_given_smoketest_base.jsonl" "$SMOKE/olmo7b_temp_results_given_smoketest_rlvr.jsonl")

echo "Set environment variables, running scripts ..."
# ---- 5. Generate (OLMo cache) ----
export HF_HOME=$LOCAL_OLMO HF_HUB_CACHE=$LOCAL_OLMO OLMO_MODEL_ROOT=$LOCAL_OLMO
for KEY in "base" "rlvr"; do
    python scripts/generate_narratives.py \
        --model_key "$KEY" \
        --temperature 0.7 \
        --num_prompts 1 \
        --output_tag smoketest \
        --output_dir $SMOKE
done

# ---- 6. Gender, in place (flan cache) ----
export HF_HOME=$LOCAL_FLAN HF_HUB_CACHE=$LOCAL_FLAN OLMO_MODEL_ROOT=$LOCAL_FLAN
python scripts/extract_gender.py --inputs "${ASSUMED[@]}" --baseline-inputs "${GIVEN[@]}" --batch_size 16
python scripts/extract_gender_pronouns.py --inputs "${ASSUMED[@]}" --baseline-inputs "${GIVEN[@]}" --batch_size 16

# ---- 7. Metrics, in place (metrics cache) ----
export HF_HOME=$LOCAL_METRICS HF_HUB_CACHE=$LOCAL_METRICS OLMO_MODEL_ROOT=$LOCAL_METRICS
python scripts/compute_diversity_metrics.py --inputs "${GIVEN[@]}" "${ASSUMED[@]}"
python scripts/compute_sentiment_regard.py --inputs "${GIVEN[@]}" "${ASSUMED[@]}"

# ---- 8. Print a short summary of the outputs ----
# Expected: 4 files, 10 rows each for 'given' and 20 rows each for 'assumed', every row carrying the
# flan_*, flan_pron_*, diversity and sentiment/regard columns
python - <<'EOF'
import json, glob
expected = ["flan_gender", "flan_pron_gender", "self_bleu", "semantic_div", "min_distinct_n",
            "trigram_repetition_rate", "n_sentences", "avg_sentiment", "regard_polarity"]
for f in sorted(glob.glob("data/results/smoketest/*.jsonl")):
    rows = [json.loads(l) for l in open(f)]
    r = rows[0]
    missing = sorted({k for row in rows for k in expected if k not in row})
    print(f, len(rows), "rows |", r["occupation"], r["profile_id"], r["temperature"],
          "| entropy:", r["entropy_analysis"]["mean_entropy"],
          "| tokens:", r["generation_length"]["num_tokens"])
    print("   ", r["response"][:200].replace("\n", " "))
    print("    gender (ground truth, given only):", r.get("gender"))
    print("    new columns:", {k: (round(r[k], 3) if isinstance(r.get(k), float) else r.get(k)) for k in expected})
    print("    MISSING on some rows:" if missing else "    all new columns present on every row", missing or "")
report = "data/results/smoketest/flan_given_baseline.json"
b = json.load(open(report))["baseline"]
print(report, "|", b["n"], "given rows | accuracy", round(b["accuracy"], 3), "| abstention", round(b["abstention_rate"], 3))
EOF

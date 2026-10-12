"""
Regression pipeline for the given & assumed results files, once gender extraction and the metrics have
been written into them.

Port of scripts/metrics-regression/compute_enriched_metrics.py of the original thesis repository (the
fits behind the paper's regression tables): the same models, group keys, weighting, clustering,
transformations, contrasts and joint tests, and the same output files. What changed: the input paths
(the pipeline's results files), the assumed-case gender label is chosen with --assumed-label-col, and
missing input columns are reported before anything is fitted.

Input (written in place by steps 3-5 of the pipeline):
  data/results/olmo7b_temp_results_given.jsonl     (--given-input)
  data/results/olmo7b_temp_results_assumed.jsonl   (--assumed-input)

DVs: self_bleu, semantic_div, mean_entropy, min_distinct_n (computed here), sentiment
(avg_sentiment), regard (regard_polarity).

Gender is a true ground-truth IV only in the "given" file (it's stated in the prompt).
In the "assumed" file gender is inferred from the model's own response, so it is
itself an outcome of (model, temperature) rather than a clean predictor. The inferred
label is read from --assumed-label-col (default flan_gender, written by
scripts/extract_gender.py; the paper's published fits used the earlier v1 flan labels,
stored as `gender` in the original results files).

Each metric is Box-Cox transformed when |skew| >= 0.25, then z-scored, and fitted on group
means by WLS (weights = responses per group) with SEs clustered on the profile:

  --prompt-case given:
    Model B-given: {metric}_z ~ is_female + is_sft + is_dpo + is_rlvr
                   + is_female:is_sft + is_female:is_dpo + is_female:is_rlvr
                   + temperature_z + C(occupation) + is_university

  --prompt-case assumed:
    Model A' (gender-assumption, logistic): fit at RESPONSE level over
      is_female ~ is_sft + is_dpo + is_rlvr + temperature_z + C(occupation) + is_university
    Model B' (diversity, caveated): same formula/DVs as Model B-given, but is_female here
      is a classifier-derived label, not ground truth -- see metadata "gender_label_caveat".

  --prompt-case pooled:
    Model C (scenario contrast): both files stacked with an is_assumed indicator and a
      three-way is_assumed x is_female x alignment_stage term. This is the formal test of
      whether the given-case gender x alignment interactions differ from the assumed-case
      ones -- "significant in one model, not the other" is not such a test.

Three corrections relative to the original pipeline, all documented in the emitted metadata:

  1. Group key. Outcomes are averaged within (profile_id, temperature, model_key) cells.
     In the assumed case the inferred gender label VARIES across the 20 responses in a cell
     (in the original 13-occupation data only 73 of 416 cells carried a single label; 176
     carried two and 167 three), yet the old code attached the cell to `gender` via .first()
     -- the label of response #1 -- while averaging outcomes over all 20. `gender` is
     therefore part of the assumed-case group key (there, the usable N rose from 383 to 679),
     so the label matches the responses it summarises.
  2. Standard errors. Each profile recurs in 16 rows (4 models x 4 temperatures), so group
     rows are not independent. SEs are clustered on the profile (see --cluster).
  3. Multiplicity and contrasts. Benjamini-Hochberg adjusted p-values are reported alongside
     raw ones, and claims about sums of coefficients ("the alignment gain is cancelled out
     for female-coded profiles") are emitted as tested contrasts.

Usage (from olmo_generation_pipeline/, after steps 3-5):
  python scripts/compute_regressions.py --prompt-case given
  python scripts/compute_regressions.py --prompt-case assumed
  python scripts/compute_regressions.py --prompt-case pooled
  python scripts/compute_regressions.py --prompt-case given --temperature-spec dummy
  scripts/run_regression.sh runs every case with the main specification and the sensitivity variants.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

from regression_utils import (
    GENDER_STAGE_NET_CONTRASTS,
    build_coeff_comparison_table,
    build_model_indicators,
    expand_entropy_columns,
    gender_balance_report,
    min_distinct_n,
    normalize_gender_series,
    read_results_file,
    run_logit_for_metric,
    run_ols_for_metrics,
    trigram_repetition_rate,
    zscore_series,
)

root_dir = Path(__file__).resolve().parent.parent     # .py < scripts < olmo_generation_pipeline
results_dir = root_dir / "data" / "results"

DEFAULT_INPUTS = {
    "given": results_dir / "olmo7b_temp_results_given.jsonl",
    "assumed": results_dir / "olmo7b_temp_results_assumed.jsonl",
}

#: The assumed-case gender label: flan_gender is what scripts/extract_gender.py writes;
#: flan_pron_gender (scripts/extract_gender_pronouns.py) and the original files' `gender`
#: (v1 flan labels behind the published fits) can be selected with --assumed-label-col.
DEFAULT_ASSUMED_LABEL_COL = "flan_gender"
LABEL_STEPS = {
    "flan_gender": "scripts/extract_gender.py",
    "flan_pron_gender": "scripts/extract_gender_pronouns.py",
}

#: Columns a results file must carry before it can be fitted, and the step that writes them.
REQUIRED_COLUMNS = {
    "model_key": "scripts/generate_narratives.py",
    "profile_id": "scripts/generate_narratives.py",
    "temperature": "scripts/generate_narratives.py",
    "occupation": "scripts/generate_narratives.py",
    "attended_university": "scripts/generate_narratives.py",
    "response": "scripts/generate_narratives.py",
    "self_bleu": "scripts/compute_diversity_metrics.py",
    "semantic_div": "scripts/compute_diversity_metrics.py",
    "avg_sentiment": "scripts/compute_sentiment_regard.py",
    "regard_polarity": "scripts/compute_sentiment_regard.py",
}

BASE_GROUP_KEYS = ["profile_id", "temperature", "model_key"]
META_COLS = [
    "model_key",
    "model_name",
    "prompt_case",
    "profile_id",
    "temperature",
    "gender",
    "occupation",
    "attended_university",
]

# self_bleu / semantic_div are already group-level (broadcast to every row in the
# group by compute_diversity_metrics.py), so "mean" within the group is a no-op for
# those two and a genuine per-response average for the rest.
METRIC_SOURCE_COLS = {
    "self_bleu": "self_bleu",
    "semantic_div": "semantic_div",
    "mean_entropy": "mean_entropy",
    "min_distinct_n": "min_distinct_n",
    "sentiment": "avg_sentiment",
    "regard": "regard_polarity",
}

#: Education is crossed with gender and occupation in the design and is perfectly
#: balanced (416 yes / 416 no; 208 per gender x education cell), so omitting it does
#: not bias the gender coefficients -- but it inflates residual variance, and leaving
#: it out makes it impossible to see whether education moderates the gender effect.
GENDER_TERMS = (
    "is_female + is_sft + is_dpo + is_rlvr "
    "+ is_female:is_sft + is_female:is_dpo + is_female:is_rlvr"
)
CONTROL_TERMS = "C(occupation) + is_university"

#: Temperature takes only four values (0.2/0.5/0.7/1.0). "linear" is the original
#: single z-scored slope; "dummy" drops the linearity assumption; "interaction" also
#: lets the temperature response differ by alignment stage, since aligned models have
#: systematically sharper output distributions than base models. Compare via AIC.
TEMPERATURE_TERMS = {
    "linear": "temperature_z",
    "dummy": "C(temperature)",
    "interaction": "C(temperature) * (is_sft + is_dpo + is_rlvr)",
}

POOLED_GENDER_TERMS = f"({GENDER_TERMS}) * is_assumed"

#: The three-way terms the pooled model exists to test.
POOLED_THREEWAY_TERMS = [
    f"is_female:is_{stage}:is_assumed" for stage in ("sft", "dpo", "rlvr")
]
POOLED_JOINT_TESTS = [("scenario_x_gender_x_stage", POOLED_THREEWAY_TERMS)]


def metric_formula(gender_terms: str, temperature_spec: str) -> str:
    return f"{{metric}}_z ~ {gender_terms} + {CONTROL_TERMS} + {TEMPERATURE_TERMS[temperature_spec]}"


def gender_logit_formula(temperature_spec: str) -> str:
    return f"is_female ~ is_sft + is_dpo + is_rlvr + {CONTROL_TERMS} + {TEMPERATURE_TERMS[temperature_spec]}"


def group_keys_for(prompt_case: str) -> list[str]:
    """Group key for a prompt case.

    In the given case gender is stated in the prompt and is constant within a cell, so
    it is metadata. In the assumed case it is inferred per response and varies within a
    cell, so it has to be part of the key -- otherwise the group carries one arbitrary
    response's label while averaging outcomes over responses of every inferred gender.
    """
    if prompt_case == "assumed":
        return BASE_GROUP_KEYS + ["gender"]
    return list(BASE_GROUP_KEYS)


def label_heterogeneity_report(df: pd.DataFrame) -> dict:
    """How often the inferred gender label disagrees within a cell.

    This is the statistic that justifies the group-key change, and it doubles as a
    measurement-noise figure worth reporting: a cell whose 20 narratives carry three
    different inferred labels is not well described by any single label.
    """
    per_cell = df.groupby(BASE_GROUP_KEYS, dropna=False)["gender"].nunique()
    counts = per_cell.value_counts().sort_index()
    return {
        "n_cells": int(per_cell.size),
        "cells_by_distinct_labels": {str(k): int(v) for k, v in counts.items()},
        "share_of_cells_with_mixed_labels": float((per_cell > 1).mean()),
    }


def build_group_summary(df: pd.DataFrame, group_keys: list[str]) -> pd.DataFrame:
    meta_value_cols = [c for c in META_COLS if c in df.columns and c not in group_keys]

    # dict.fromkeys de-duplicates in a fixed order (the original's set() made the column order
    # of the written regression-input CSVs vary from run to run; the fits are unaffected)
    source_cols = [c for c in dict.fromkeys(METRIC_SOURCE_COLS.values()) if c in df.columns]
    metrics_df = df.groupby(group_keys, dropna=False)[source_cols].mean().reset_index()

    response_counts = (
        df.groupby(group_keys, dropna=False).size().reset_index(name="response_count")
    )

    if meta_value_cols:
        meta_df = df.groupby(group_keys, dropna=False)[meta_value_cols].first().reset_index()
    else:
        meta_df = df[group_keys].drop_duplicates().reset_index(drop=True)

    summary_df = meta_df.merge(metrics_df, on=group_keys, how="left")
    summary_df = summary_df.merge(response_counts, on=group_keys, how="left")

    rename_map = {
        src: dv
        for dv, src in METRIC_SOURCE_COLS.items()
        if src in summary_df.columns and src != dv
    }
    summary_df = summary_df.rename(columns=rename_map)

    return summary_df


def add_dummies(df: pd.DataFrame) -> pd.DataFrame:
    regression_df = df.copy()
    regression_df["gender_norm"] = normalize_gender_series(regression_df.get("gender"))
    regression_df["is_female"] = (regression_df["gender_norm"] == "female").astype(int)
    if "occupation" in regression_df.columns:
        regression_df["occupation"] = regression_df["occupation"].astype("category")
    if "attended_university" in regression_df.columns:
        regression_df["is_university"] = (
            regression_df["attended_university"].astype(str).str.strip().str.lower() == "yes"
        ).astype(int)

    model_indicators = build_model_indicators(regression_df.get("model_key"))
    regression_df = pd.concat([regression_df, model_indicators], axis=1)
    regression_df["temperature_z"] = zscore_series(regression_df["temperature"])

    return regression_df


def check_columns(df: pd.DataFrame, input_path: Path, label_col: str, label_step: str) -> None:
    """Fail before fitting if a column is missing, naming the pipeline step that writes it.

    Null values inside a present column are reported but allowed: some are legitimate (a
    response without a usable sentence has no sentiment), and the fits drop them as before.
    """
    required = dict(REQUIRED_COLUMNS)
    required[label_col] = label_step
    missing = {col: step for col, step in required.items() if col not in df.columns}
    if "entropy_analysis" not in df.columns and "mean_entropy" not in df.columns:
        missing["entropy_analysis"] = "scripts/generate_narratives.py"
    if missing:
        lines = "\n".join(f"  {col:<20} <- run {step}" for col, step in missing.items())
        raise SystemExit(f"ERROR: {input_path} is missing columns the regression needs:\n{lines}")
    nulls = {col: int(df[col].isna().sum()) for col in required if df[col].isna().any()}
    if nulls:
        print(f"Note: null values in {input_path.name} (dropped per metric where they matter): {nulls}")


def load_prepared(input_path: Path, prompt_case: str, assumed_label_col: str = DEFAULT_ASSUMED_LABEL_COL,
                  limit: int | None = None) -> pd.DataFrame:
    """Read a results file and attach the response-level columns the models need.

    In the assumed case the chosen classifier label becomes `gender`; in the given case
    `gender` is the gender the prompt stated and is used as it is.
    """
    print(f"Loading {input_path}")
    if not input_path.exists():
        raise SystemExit(f"ERROR: input not found: {input_path} (has generation been run?)")
    df = read_results_file(input_path)
    if limit:
        df = df.head(limit).copy()
    if prompt_case == "assumed":
        label_step = LABEL_STEPS.get(assumed_label_col, f"the step that writes '{assumed_label_col}'")
        check_columns(df, input_path, assumed_label_col, label_step)
        df["gender"] = df[assumed_label_col]
    else:
        check_columns(df, input_path, "gender", "scripts/generate_narratives.py (given prompts)")
    df = expand_entropy_columns(df)

    df["temperature"] = pd.to_numeric(df.get("temperature"), errors="coerce")
    df = df.dropna(subset=["temperature", "response"])

    tqdm.pandas(desc="Computing min_distinct_n")
    df["min_distinct_n"] = df["response"].progress_apply(min_distinct_n)
    return df


def drop_repetition_loops(df: pd.DataFrame, threshold: float) -> tuple[pd.DataFrame, dict]:
    """Remove degenerate responses before aggregation, reporting what was removed.

    A response that loops a phrase inflates self-BLEU and depresses every diversity
    metric mechanically, so if the loop rate differs by alignment stage it contaminates
    exactly the comparison being made.
    """
    rates = df["response"].apply(trigram_repetition_rate)
    keep = ~(rates > threshold)
    report = {
        "threshold": threshold,
        "n_before": int(len(df)),
        "n_dropped": int((~keep).sum()),
        "dropped_share_by_model_key": {
            str(k): float(v) for k, v in (~keep).groupby(df["model_key"]).mean().items()
        },
    }
    return df[keep].copy(), report


def drop_unlabelled(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows the model cannot use: the classifier declined to assign a binary gender.

    These are dropped rather than pooled into either class; the rate is reported in the
    metadata so the abstention is visible rather than silent.
    """
    return df[df["gender_norm"] != "none"].copy()


def write_outputs(output_dir: Path, outputs: dict[str, pd.DataFrame]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for key, df_out in outputs.items():
        if df_out is None or (hasattr(df_out, "empty") and df_out.empty):
            continue
        df_out.to_csv(output_dir / f"{key}.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt-case", choices=["given", "assumed", "pooled"], required=True)
    parser.add_argument("--given-input", type=Path, default=DEFAULT_INPUTS["given"],
                        help="Given-case results file (default: data/results/olmo7b_temp_results_given.jsonl)")
    parser.add_argument("--assumed-input", type=Path, default=DEFAULT_INPUTS["assumed"],
                        help="Assumed-case results file (default: data/results/olmo7b_temp_results_assumed.jsonl)")
    parser.add_argument("--assumed-label-col", type=str, default=DEFAULT_ASSUMED_LABEL_COL,
                        help="Column holding the inferred gender of each assumed narrative "
                             "(default: flan_gender; flan_pron_gender for the pronoun classifier; "
                             "gender reproduces the published fits on the original results files)")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Override default output dir (default: data/results/regression/<prompt case>)")
    parser.add_argument("--limit", type=int, default=None, help="Cap rows for smoke testing")
    parser.add_argument(
        "--temperature-spec",
        choices=sorted(TEMPERATURE_TERMS),
        default="linear",
        help="How temperature enters the model (compare via the AIC column in *_models.csv)",
    )
    parser.add_argument(
        "--cluster",
        default="profile_id",
        help="Column to cluster standard errors on; 'none' to disable",
    )
    parser.add_argument(
        "--min-group-size",
        type=int,
        default=1,
        help="Drop groups summarising fewer than this many responses (sensitivity check)",
    )
    parser.add_argument(
        "--no-weights",
        action="store_true",
        help="Fit unweighted OLS instead of weighting group means by response_count",
    )
    parser.add_argument(
        "--drop-repetition-loops",
        type=float,
        default=None,
        metavar="THRESHOLD",
        help=(
            "Drop responses whose trigram-repetition rate exceeds THRESHOLD (e.g. 0.3) "
            "before aggregating. The rate was uneven across variants in the original assumed "
            "data (base 15.2%%, SFT 12.6%%, RLVR 3.5%%, DPO 2.2%%), so this is the sensitivity "
            "check for whether degenerate base output drives the alignment effects."
        ),
    )
    args = parser.parse_args()

    output_dir = args.output_dir or (results_dir / "regression" / args.prompt_case)
    dv_cols = list(METRIC_SOURCE_COLS.keys())
    prefix = f"{args.prompt_case}_enriched"
    inputs = {"given": args.given_input, "assumed": args.assumed_input}

    metadata: dict = {
        "prompt_case": args.prompt_case,
        "metric_cols": dv_cols,
        "temperature_spec": args.temperature_spec,
        "temperature_terms": TEMPERATURE_TERMS[args.temperature_spec],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    if args.prompt_case in ("assumed", "pooled"):
        metadata["assumed_label_col"] = args.assumed_label_col
    outputs: dict[str, pd.DataFrame] = {}

    # ------------------------------------------------------------------ build frames
    repetition_reports: dict[str, dict] = {}

    if args.prompt_case == "pooled":
        frames = []
        for case in ("given", "assumed"):
            raw = load_prepared(inputs[case], case, args.assumed_label_col, args.limit)
            if args.drop_repetition_loops is not None:
                raw, repetition_reports[case] = drop_repetition_loops(
                    raw, args.drop_repetition_loops
                )
            summary = build_group_summary(raw, group_keys_for(case))
            summary["scenario"] = case
            frames.append(summary)
            if case == "assumed":
                metadata["assumed_label_heterogeneity"] = label_heterogeneity_report(raw)
        summary_df = pd.concat(frames, ignore_index=True)
        regression_df = add_dummies(summary_df)
        regression_df["is_assumed"] = (regression_df["scenario"] == "assumed").astype(int)
        # Given profile 7 and assumed profile 7 are different prompts, so the cluster
        # identifier has to carry the scenario as well as the profile.
        regression_df["cluster_id"] = (
            regression_df["scenario"] + ":" + regression_df["profile_id"].astype(str)
        )
        metadata["input_path"] = {k: str(v) for k, v in inputs.items()}
    else:
        input_path = inputs[args.prompt_case]
        raw_df = load_prepared(input_path, args.prompt_case, args.assumed_label_col, args.limit)
        if args.drop_repetition_loops is not None:
            raw_df, repetition_reports[args.prompt_case] = drop_repetition_loops(
                raw_df, args.drop_repetition_loops
            )
        group_keys = group_keys_for(args.prompt_case)
        summary_df = build_group_summary(raw_df, group_keys)
        regression_df = add_dummies(summary_df)
        regression_df["cluster_id"] = regression_df["profile_id"].astype(str)
        metadata["input_path"] = str(input_path)
        metadata["group_keys"] = group_keys
        if args.prompt_case == "assumed":
            metadata["assumed_label_heterogeneity"] = label_heterogeneity_report(raw_df)

    outputs["regression_input_raw"] = summary_df
    if repetition_reports:
        metadata["repetition_loop_filter"] = repetition_reports

    # ------------------------------------------------------ filter and record balance
    metadata["gender_balance"] = gender_balance_report(regression_df["gender_norm"])
    n_before = len(regression_df)
    regression_df = drop_unlabelled(regression_df)
    metadata["groups_dropped_unlabelled"] = int(n_before - len(regression_df))

    if args.min_group_size > 1:
        n_pre = len(regression_df)
        regression_df = regression_df[regression_df["response_count"] >= args.min_group_size]
        metadata["groups_dropped_below_min_size"] = int(n_pre - len(regression_df))
    metadata["min_group_size"] = args.min_group_size
    metadata["n_groups_fitted"] = int(len(regression_df))
    metadata["response_count_summary"] = {
        "min": int(regression_df["response_count"].min()),
        "median": float(regression_df["response_count"].median()),
        "max": int(regression_df["response_count"].max()),
    }

    cluster_col = None if args.cluster.lower() == "none" else "cluster_id"
    weight_col = None if args.no_weights else "response_count"
    metadata["cluster_on"] = args.cluster if cluster_col else "none"
    metadata["weighted_by"] = weight_col or "none"
    if cluster_col:
        # Cluster-robust inference is asymptotic in the number of CLUSTERS, not rows.
        # The assumed case has few profiles (26 for the original 13 occupations, 30 for the
        # 15), which is on the low side, so record the count rather than leaving a reader to
        # infer it.
        metadata["n_clusters"] = int(regression_df[cluster_col].nunique())

    # ----------------------------------------------------------------- fit the models
    if args.prompt_case == "pooled":
        formula = metric_formula(POOLED_GENDER_TERMS, args.temperature_spec)
        joint_tests = POOLED_JOINT_TESTS
    else:
        formula = metric_formula(GENDER_TERMS, args.temperature_spec)
        joint_tests = None
    metadata["formula"] = formula

    results = run_ols_for_metrics(
        regression_df,
        dv_cols,
        formula,
        cluster_col=cluster_col,
        weight_col=weight_col,
        contrasts=GENDER_STAGE_NET_CONTRASTS,
        joint_tests=joint_tests,
    )
    wide = build_coeff_comparison_table(results["coeffs"], metrics=dv_cols)

    outputs.update(
        {
            f"{prefix}_regression_input": results["regression_input"],
            f"{prefix}_regression_coeffs": results["coeffs"],
            f"{prefix}_regression_coeffs_wide": wide,
            f"{prefix}_regression_contrasts": results["contrasts"],
            f"{prefix}_regression_joint_tests": results["joint_tests"],
            f"{prefix}_regression_models": results["models"],
            f"{prefix}_transform_metadata": results["transform_metadata"],
        }
    )

    if args.prompt_case == "assumed":
        metadata["gender_label_caveat"] = (
            "is_female in this file is inferred from the model's own response by a "
            "classifier, not a ground-truth profile attribute. It is a collider on the "
            "path from alignment stage to text properties, so conditioning on it can "
            "induce association even where none exists. Model A' tests whether "
            "model/temperature shift this predicted label; Model B' coefficients on "
            "is_female should be read as 'conditional on the model having assumed this "
            "gender', not as a causal effect of gender."
        )

        # Model A' is a per-response Bernoulli question ("did the model assume female
        # here?"), so it belongs at response level. Fitting it on group rows discarded
        # ~95% of the observations; clustering on profile handles the dependence that
        # the aggregation was implicitly standing in for.
        response_df = add_dummies(raw_df)
        response_df["cluster_id"] = response_df["profile_id"].astype(str)
        response_df = drop_unlabelled(response_df)

        logit_formula = gender_logit_formula(args.temperature_spec)
        logit_results = run_logit_for_metric(
            response_df, logit_formula, dv="is_female", cluster_col=cluster_col
        )
        metadata["model_a_prime"] = {
            "formula": logit_formula,
            "level": "response",
            "nobs": int(logit_results["model"]["nobs"].iloc[0]),
            "converged": bool(logit_results["model"]["converged"].iloc[0]),
            "prsquared": float(logit_results["model"]["prsquared"].iloc[0]),
        }
        outputs.update(
            {
                "assumed_gender_assumption_logit_coeffs": logit_results["coeffs"],
                "assumed_gender_assumption_logit_model": logit_results["model"],
            }
        )

    write_outputs(output_dir, outputs)

    metadata_path = output_dir / f"{args.prompt_case}_enriched_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"Wrote metadata to {metadata_path}")
    print(f"Wrote {len(outputs)} CSVs to {output_dir}")
    print(results["models"].to_string(index=False))


if __name__ == "__main__":
    main()

"""
Shared helpers for the regression pipelines in this directory
(compute_enriched_metrics.py, compute_manifold_metrics.py).

Ported/generalized from:
- compute_given_metrics.py (zscore_series, boxcox_if_skewed, normalize_gender_series,
  build_model_indicators, expand_entropy_columns, the OLS-fitting loop)
- scripts/olmo-generation/sample_gender_spotcheck.py (tokenize, distinct_n, min_distinct_n)
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.multitest import multipletests

# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def read_results_file(file_path: Path) -> pd.DataFrame:
    file_path = Path(file_path)
    if file_path.suffix == ".jsonl":
        return pd.read_json(file_path, lines=True)
    if file_path.suffix == ".json":
        return pd.read_json(file_path)
    raise ValueError(f"Unsupported file type: {file_path}")


def expand_entropy_columns(df: pd.DataFrame) -> pd.DataFrame:
    if "entropy_analysis" not in df.columns:
        return df

    entropy_df = pd.json_normalize(df["entropy_analysis"])
    for col in entropy_df.columns:
        if col in df.columns:
            df[col] = df[col].fillna(entropy_df[col])
        else:
            df[col] = entropy_df[col]

    return df.drop(columns=["entropy_analysis"])


# ---------------------------------------------------------------------------
# min_distinct_n (ported from sample_gender_spotcheck.py)
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"\b\w+\b")
_DISTINCT_N_VALUES = range(1, 6)


def tokenize(text: str) -> list[str]:
    return _WORD_RE.findall(str(text).lower())


def distinct_n(tokens: list[str], n: int) -> float:
    if len(tokens) < n:
        return float("nan")
    ngrams = list(zip(*[tokens[i:] for i in range(n)]))
    return len(set(ngrams)) / len(ngrams)


def min_distinct_n(text: str) -> float:
    """Minimum distinct-n over n = 1..5, within a single response.

    Note the minimum is attained at n = 1 in essentially every real narrative
    (distinct-1 <= distinct-2 <= ...), so this is the unigram type-token ratio in
    practice. TTR falls with text length by construction, so it is length-sensitive.
    """
    tokens = tokenize(text)
    return float(np.nanmin([distinct_n(tokens, n) for n in _DISTINCT_N_VALUES]))


def trigram_repetition_rate(text: str, n: int = 3, min_tokens: int = 20) -> float:
    """Share of n-grams in a response that are repeats.

    0 means every n-gram is unique; values approaching 1 indicate a repetition loop.
    Short responses return NaN rather than a misleadingly extreme value.
    """
    tokens = tokenize(text)
    if len(tokens) < max(min_tokens, n + 1):
        return float("nan")
    ngrams = list(zip(*[tokens[i:] for i in range(n)]))
    return 1.0 - len(set(ngrams)) / len(ngrams)


# ---------------------------------------------------------------------------
# self_bleu / semantic_diversity (ported from compute_given_metrics.py /
# compute_diversity_metrics.py)
# ---------------------------------------------------------------------------


def self_bleu(texts: list[str], ns: list[int] = [2, 3, 4, 5]) -> float:
    from nltk.tokenize import word_tokenize
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

    if len(texts) < 2:
        return float("nan")

    tokenized = [word_tokenize(t.lower()) for t in texts]
    smoothing = SmoothingFunction().method1
    scores_per_n: dict[int, list[float]] = {n: [] for n in ns}

    for i, hyp in enumerate(tokenized):
        refs = [t for j, t in enumerate(tokenized) if j != i]
        for n in ns:
            weights = tuple(1 / n for _ in range(n))
            score = sentence_bleu(refs, hyp, weights=weights, smoothing_function=smoothing)
            scores_per_n[n].append(score)

    mean_scores = [float(np.mean(scores_per_n[n])) for n in ns]
    return float(np.exp(np.mean(np.log(np.clip(mean_scores, 1e-10, None)))))


def semantic_diversity(model, texts: list[str]) -> float:
    from scipy.spatial.distance import pdist

    if len(texts) < 2:
        return float("nan")

    embeddings = model.encode(texts, show_progress_bar=False)
    pairwise_distances = pdist(embeddings, metric="cosine")
    return float(np.mean(pairwise_distances))


def ensure_nltk_punkt() -> None:
    try:
        from nltk.tokenize import word_tokenize

        word_tokenize("test")
    except LookupError:
        import nltk

        print("NLTK punkt not found. Downloading...")
        try:
            nltk.download("punkt_tab")
        except Exception:
            nltk.download("punkt")


# ---------------------------------------------------------------------------
# Gender / model dummy construction (ported from compute_given_metrics.py)
# ---------------------------------------------------------------------------


def normalize_gender_series(series: pd.Series) -> pd.Series:
    normalized = series.fillna("none").astype(str).str.strip().str.lower()
    replacements = {
        "unspecified": "none",
        "none": "none",
        "nan": "none",
        "": "none",
    }
    return normalized.replace(replacements)


def build_model_indicators(model_series: pd.Series) -> pd.DataFrame:
    normalized = model_series.fillna("").astype(str).str.strip().str.lower()
    indicators = pd.DataFrame(index=model_series.index)
    indicators["is_sft"] = normalized.str.contains("sft").astype(int)
    indicators["is_dpo"] = normalized.str.contains("dpo").astype(int)
    indicators["is_rlvr"] = normalized.str.contains("rlvr").astype(int)
    return indicators


def gender_balance_report(gender_norm: pd.Series) -> dict:
    """Value counts of a normalized gender series, for logging imbalance to metadata."""
    counts = gender_norm.value_counts(dropna=False).to_dict()
    counts = {str(k): int(v) for k, v in counts.items()}
    n_male = counts.get("male", 0)
    n_female = counts.get("female", 0)
    n_binary = n_male + n_female
    return {
        "counts": counts,
        "n_male": n_male,
        "n_female": n_female,
        "female_share_of_binary": (n_female / n_binary) if n_binary else None,
    }


# ---------------------------------------------------------------------------
# Box-Cox + z-scoring (ported from compute_given_metrics.py)
# ---------------------------------------------------------------------------


def zscore_series(series: pd.Series) -> pd.Series:
    values = series.astype(float)
    mean = values.mean()
    std = values.std()
    if not np.isfinite(std) or std == 0:
        return pd.Series(np.where(values.notna(), 0.0, np.nan), index=values.index)
    return (values - mean) / std


def boxcox_if_skewed(
    series: pd.Series,
    skew_threshold: float = 0.25,
) -> tuple[pd.Series, dict]:
    cleaned = series.dropna().astype(float)
    skew_value = float(cleaned.skew()) if not cleaned.empty else float("nan")
    info = {
        "transform": "none",
        "skew": skew_value,
        "skew_threshold": skew_threshold,
        "lambda": None,
        "shift": 0.0,
    }

    if cleaned.empty or cleaned.nunique() < 2:
        return series, info

    if not np.isfinite(skew_value) or abs(skew_value) < skew_threshold:
        return series, info

    shift = 0.0
    min_val = cleaned.min()
    if min_val <= 0:
        shift = 1 - min_val

    transformed, lambda_value = stats.boxcox(cleaned + shift)
    transformed_series = series.copy()
    transformed_series.loc[cleaned.index] = transformed

    info.update(
        {
            "transform": "boxcox",
            "lambda": float(lambda_value),
            "shift": float(shift),
        }
    )

    return transformed_series, info


# ---------------------------------------------------------------------------
# Contrasts and multiple-comparison correction
# ---------------------------------------------------------------------------

#: Net effect of each alignment stage on female-coded profiles: the stage main
#: effect plus its interaction with is_female. Claims of the form "the alignment
#: gain is cancelled out for female-coded profiles" are statements about these
#: sums, so they are tested rather than eyeballed from two separately reported
#: coefficients.
GENDER_STAGE_NET_CONTRASTS = [
    (f"net_{stage}_female", f"is_{stage} + is_female:is_{stage}")
    for stage in ("sft", "dpo", "rlvr")
]


def run_contrasts(model, contrasts: list[tuple[str, str]], metric: str) -> list[dict]:
    """Evaluate `(label, expression)` linear combinations against a fitted model.

    Expressions patsy cannot resolve (a term absent from this specification) are
    skipped rather than raising, so one contrast list can be reused across
    formulas.
    """
    rows = []
    for label, expression in contrasts:
        try:
            test = model.t_test(f"{expression} = 0")
        except Exception:
            continue
        conf_low, conf_high = np.ravel(test.conf_int())[:2]
        rows.append(
            {
                "metric": metric,
                "contrast": label,
                "expression": expression,
                "estimate": float(np.squeeze(test.effect)),
                "stderr": float(np.squeeze(test.sd)),
                "pvalue": float(np.squeeze(test.pvalue)),
                "conf_low": float(conf_low),
                "conf_high": float(conf_high),
            }
        )
    return rows


def run_joint_tests(model, joint_tests: list[tuple[str, list[str]]], metric: str) -> list[dict]:
    """Evaluate joint (multi-df) Wald tests, e.g. "are all three three-way terms zero?".

    Counting how many individual coefficients clear 0.05 is a weak substitute for
    asking whether a block of terms is jointly zero; this reports the latter.
    """
    rows = []
    for label, expressions in joint_tests:
        try:
            test = model.f_test(", ".join(f"{e} = 0" for e in expressions))
        except Exception:
            continue
        rows.append(
            {
                "metric": metric,
                "joint_test": label,
                "expressions": "; ".join(expressions),
                "fvalue": float(np.squeeze(test.fvalue)),
                "pvalue": float(np.squeeze(test.pvalue)),
                "df_num": int(test.df_num),
                "df_denom": int(test.df_denom) if test.df_denom is not None else -1,
            }
        )
    return rows


def add_fdr_within_metric(coeff_rows_df: pd.DataFrame, method: str = "fdr_bh") -> pd.DataFrame:
    """Add a `pvalue_adj` column, corrected within each outcome family.

    The intercept is excluded from the family -- it is not a tested hypothesis,
    and including it would dilute the correction.
    """
    if coeff_rows_df.empty:
        return coeff_rows_df

    df = coeff_rows_df.copy()
    df["pvalue_adj"] = np.nan
    df["fdr_method"] = method

    testable = df["term"] != "Intercept"
    for _, group in df[testable].groupby("metric"):
        pvals = group["pvalue"].to_numpy(dtype=float)
        if len(pvals) == 0 or not np.isfinite(pvals).all():
            continue
        df.loc[group.index, "pvalue_adj"] = multipletests(pvals, method=method)[1]

    return df


# ---------------------------------------------------------------------------
# OLS fitting loop (generalized from compute_given_metrics.py __main__)
# ---------------------------------------------------------------------------


def _fit_linear_model(
    formula: str,
    data: pd.DataFrame,
    cluster_col: str | None = None,
    weight_col: str | None = None,
):
    """Fit `formula` on `data`, optionally weighted and/or cluster-robust.

    `weight_col` holds the number of responses behind each group mean, so a group
    averaging 20 narratives is not given the same leverage as one averaging 1.
    `cluster_col` accounts for the same profile recurring across every
    (model, temperature) cell.
    """
    fit_kwargs: dict = {}
    if cluster_col is not None:
        fit_kwargs["cov_type"] = "cluster"
        fit_kwargs["cov_kwds"] = {"groups": data[cluster_col].astype(str).to_numpy()}

    if weight_col is not None:
        return smf.wls(formula, data=data, weights=data[weight_col].astype(float)).fit(**fit_kwargs)
    return smf.ols(formula, data=data).fit(**fit_kwargs)


def run_ols_for_metrics(
    df: pd.DataFrame,
    metric_cols: list[str],
    formula_template: str,
    skew_threshold: float = 0.25,
    cluster_col: str | None = None,
    weight_col: str | None = None,
    contrasts: list[tuple[str, str]] | None = None,
    joint_tests: list[tuple[str, list[str]]] | None = None,
    fdr_method: str = "fdr_bh",
) -> dict[str, pd.DataFrame]:
    """Box-cox (if skewed) + z-score each metric, then fit `formula_template.format(metric=metric)`
    via statsmodels OLS (or WLS when `weight_col` is given). Mirrors the loop in
    compute_given_metrics.py but is metric/formula-agnostic.

    Returns a dict with keys: "regression_input" (df with the *_z columns added),
    "coeffs" (long-form coefficient table with Benjamini-Hochberg adjusted p-values),
    "contrasts" (tested linear combinations), "models" (per-metric model summary stats),
    "transform_metadata" (box-cox/z-score info per metric).
    """
    regression_df = df.copy()
    transform_metadata = []
    regression_outputs = []
    coeff_rows = []
    contrast_rows = []
    joint_rows = []

    for metric in metric_cols:
        if metric not in regression_df.columns:
            continue

        transformed, info = boxcox_if_skewed(regression_df[metric], skew_threshold=skew_threshold)
        metric_z = zscore_series(transformed)
        metric_z_name = f"{metric}_z"
        regression_df[metric_z_name] = metric_z

        info.update({"metric": metric})
        transform_metadata.append(info)

        formula = formula_template.format(metric=metric)
        model_data = regression_df.dropna(subset=[metric_z_name])
        if model_data.empty:
            continue

        model = _fit_linear_model(
            formula, model_data, cluster_col=cluster_col, weight_col=weight_col
        )
        regression_outputs.append(
            {
                "metric": metric,
                "nobs": int(model.nobs),
                "r2": float(model.rsquared),
                "r2_adj": float(model.rsquared_adj),
                "aic": float(model.aic),
                "bic": float(model.bic),
                "f_pvalue": float(model.f_pvalue) if model.f_pvalue is not None else float("nan"),
                "cov_type": f"cluster({cluster_col})" if cluster_col else "nonrobust",
                "weights": weight_col or "none",
            }
        )

        if contrasts:
            contrast_rows.extend(run_contrasts(model, contrasts, metric))
        if joint_tests:
            joint_rows.extend(run_joint_tests(model, joint_tests, metric))

        conf_int = model.conf_int()
        for term, coef in model.params.items():
            conf_low, conf_high = conf_int.loc[term].tolist()
            coeff_rows.append(
                {
                    "metric": metric,
                    "term": term,
                    "coef": float(coef),
                    "stderr": float(model.bse[term]),
                    "pvalue": float(model.pvalues[term]),
                    "conf_low": float(conf_low),
                    "conf_high": float(conf_high),
                    "nobs": int(model.nobs),
                    "r2": float(model.rsquared),
                }
            )

    return {
        "regression_input": regression_df,
        "coeffs": add_fdr_within_metric(pd.DataFrame(coeff_rows), method=fdr_method),
        "contrasts": pd.DataFrame(contrast_rows),
        "joint_tests": pd.DataFrame(joint_rows),
        "models": pd.DataFrame(regression_outputs),
        "transform_metadata": pd.DataFrame(transform_metadata),
    }


def run_logit_for_metric(
    df: pd.DataFrame,
    formula: str,
    dv: str,
    cluster_col: str | None = None,
    fdr_method: str = "fdr_bh",
) -> dict:
    """Fit a single logistic regression (e.g. is_female ~ ...). Returns coeffs (long-form,
    including odds ratios and BH-adjusted p-values) and a one-row model summary, in the same
    shape as run_ols_for_metrics so both can feed build_coeff_comparison_table.
    """
    model_data = df.dropna(subset=[dv])
    fit_kwargs: dict = {}
    if cluster_col is not None:
        fit_kwargs["cov_type"] = "cluster"
        fit_kwargs["cov_kwds"] = {"groups": model_data[cluster_col].astype(str).to_numpy()}
    model = smf.logit(formula, data=model_data).fit(disp=False, **fit_kwargs)

    conf_int = model.conf_int()
    coeff_rows = []
    for term, coef in model.params.items():
        conf_low, conf_high = conf_int.loc[term].tolist()
        coeff_rows.append(
            {
                "metric": dv,
                "term": term,
                "coef": float(coef),
                "odds_ratio": float(np.exp(coef)),
                "stderr": float(model.bse[term]),
                "pvalue": float(model.pvalues[term]),
                "conf_low": float(conf_low),
                "conf_high": float(conf_high),
                "nobs": int(model.nobs),
            }
        )

    model_summary = pd.DataFrame(
        [
            {
                "metric": dv,
                "nobs": int(model.nobs),
                "converged": bool(model.mle_retvals.get("converged", False)),
                "llf": float(model.llf),
                "prsquared": float(model.prsquared),
                "cov_type": f"cluster({cluster_col})" if cluster_col else "nonrobust",
            }
        ]
    )

    return {
        "coeffs": add_fdr_within_metric(pd.DataFrame(coeff_rows), method=fdr_method),
        "model": model_summary,
        "fitted": model,
    }


# ---------------------------------------------------------------------------
# Wide comparison table
# ---------------------------------------------------------------------------


def _significance_stars(pvalue: float) -> str:
    if pd.isna(pvalue):
        return ""
    if pvalue < 0.001:
        return "***"
    if pvalue < 0.01:
        return "**"
    if pvalue < 0.05:
        return "*"
    return ""


def build_coeff_comparison_table(
    coeff_rows_df: pd.DataFrame,
    metrics: list[str] | None = None,
    terms: list[str] | None = None,
) -> pd.DataFrame:
    """Pivot a long-form coefficient table (columns: metric, term, coef, pvalue, ...)
    into a wide "at a glance" table: one row per term, one column per metric, cell =
    "coef (stars)". Full long-form coefficients (stderr/CI/nobs/r2) remain in coeff_rows_df
    for the report appendix; this table is for quick visual scanning only.

    Stars reflect the Benjamini-Hochberg adjusted p-value when `pvalue_adj` is present,
    so the scan table cannot advertise a finding the correction does not support. The raw
    p-values remain in coeff_rows_df and both are reported.
    """
    if coeff_rows_df.empty:
        return pd.DataFrame()

    df = coeff_rows_df.copy()
    if metrics is not None:
        df = df[df["metric"].isin(metrics)]
    if terms is not None:
        df = df[df["term"].isin(terms)]

    star_col = "pvalue_adj" if "pvalue_adj" in df.columns else "pvalue"
    df["cell"] = df.apply(
        lambda row: f"{row['coef']:.4f}{_significance_stars(row[star_col])}", axis=1
    )

    wide = df.pivot_table(index="term", columns="metric", values="cell", aggfunc="first")
    wide = wide.reset_index()
    return wide

"""
Pronoun-question variant of the Flan gender classifier, for comparing against extract_gender.py.

Same model (flan-t5-xl), same first-token posterior, same base-model nudge and the same in-place
machinery as extract_gender.py -- this module imports that file's scoring, loaders and atomic writer
unchanged, so the only thing that differs is the QUESTION. extract_gender.py asks what gender the subject
is presented as (male / female / none, using any cue). This asks which pronouns the text uses for the
subject: He, She, They or unclear (scripts/pronoun_config.json, where the prompt can be edited).

Per run:
  1. Pronoun labels for each ASSUMED narrative, written IN PLACE (add-only, atomic), as flan_pron_*
     columns: flan_pron_p_<label>, flan_pron_label (he/she/they/unclear), flan_pron_gender
     (he -> male, she -> female, they/unclear -> none), flan_pron_score, flan_pron_margin,
     flan_pron_label_mass. The existing flan_* columns are never touched.
  2. The same classifier on the GIVEN narratives, whose prompt states the gender (ground truth). The given
     rows get the same flan_pron_* columns written in place (add-only: their ground-truth `gender` and any
     flan_* columns are never touched). A 'they' or 'unclear' answer counts as an error, exactly as 'none'
     does for the existing classifier, so the two accuracies are like-for-like.
  3. The comparison. On the given narratives BOTH classifiers are scored on the same rows in this run
     (paired: both right / only one right / neither, with an exact McNemar p-value). On the assumed
     narratives, the pronoun label is compared with the existing flan_gender column when it is present
     (run extract_gender.py --case assumed first). Everything is printed and written to
     flan_pronoun_report.json next to the first input.

How to read it:
  - The two classifiers ask different questions. A narrative that gendered its subject by name alone is
    'unclear' here but can be male/female for the existing one, so a gap is not necessarily an error.
  - The given-case caveats of extract_gender.py apply (understates where the generator ignored the stated
    gender; overstates assumed-case performance; no gold 'none' examples).

Usage (from olmo_generation_pipeline/, after generation):
  python scripts/extract_gender_pronouns.py                   # everything
  python scripts/extract_gender_pronouns.py --case given      # comparison on the given narratives only
  python scripts/extract_gender_pronouns.py --case assumed    # only label the assumed narratives
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

script_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(script_dir))

import extract_gender as eg  # noqa: E402  (scoring, loaders, atomic write: reused unchanged)
import torch  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

DEFAULT_CONFIG = script_dir / "pronoun_config.json"
PREFIX = "flan_pron_"
GENDER_ORDER = ["male", "female", "none"]

COMPARISON_CAVEATS = [
    "Both classifiers are scored with the same model, the same texts and the same nudge handling in the "
    "same run, so the paired counts on the given narratives are like-for-like.",
    "he -> male and she -> female; 'they' and 'unclear' both map to none. A 'they' answer is an error on "
    "the given texts exactly as 'none' is for the existing classifier.",
    "The existing classifier asks what gender the subject is presented as using any cue (name, pronouns, "
    "description); this one asks which pronouns are used. A narrative that gendered its subject by name "
    "alone is 'unclear' here, so a gap is not necessarily classifier error.",
] + eg.BASELINE_CAVEATS


# ---------------------------------------------------------------------------
# Writing the labels
# ---------------------------------------------------------------------------


def apply_labels(rows: list[dict], res: dict, config: dict) -> None:
    """Add the flan_pron_* keys to every row (mutates `rows`, changes nothing else).

    `res` comes from extract_gender.score_rows, which calls the predicted label "genders".
    """
    to_gender = config["to_gender"]
    for i, row in enumerate(rows):
        for j, label in enumerate(res["order"]):
            row[f"{PREFIX}p_{label}"] = res["probs"][i][j]
        label = res["genders"][i]
        row[f"{PREFIX}label"] = label
        row[f"{PREFIX}gender"] = to_gender[label]
        row[f"{PREFIX}score"] = res["scores"][i]
        row[f"{PREFIX}margin"] = res["margins"][i]
        # sequence_sum scoring has no label mass (NaN); store null rather than a bare NaN token
        mass = res["mass"][i]
        row[f"{PREFIX}label_mass"] = mass if math.isfinite(mass) else None


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


def mcnemar_exact(only_a: int, only_b: int) -> float:
    """Two-sided exact McNemar p-value from the discordant counts."""
    n = only_a + only_b
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(only_a, only_b) + 1))
    return min(1.0, 2 * tail / 2 ** n)


def _records(rows: list[dict], res: dict, to_gender: dict | None) -> list[dict]:
    """Ground truth + prediction per given narrative, in the shape extract_gender.summarise_baseline wants."""
    out = []
    for i, row in enumerate(rows):
        label = res["genders"][i]
        out.append(
            {
                "model_key": row.get("model_key"),
                "temperature": row.get("temperature"),
                "occupation": row["occupation"],
                "truth": str(row["gender"]).strip().lower(),
                "label": label,
                "pred": to_gender[label] if to_gender else label,
                "score": res["scores"][i],
                "mass": res["mass"][i],
            }
        )
    return out


def compare_given(given_rows_by_path: dict, score_pron, score_existing, cfg_pron: dict, cfg_existing: dict,
                  batch_size: int, nudge_pron: bool, nudge_existing: bool) -> dict:
    """Score the given narratives with BOTH classifiers and compare them.

    The rows gain the flan_pron_* keys (add-only; the caller writes the file). The existing classifier is
    re-scored here only for the paired comparison: its flan_* columns are extract_gender.py's to write.
    """
    pron, existing = [], []
    for path, rows in given_rows_by_path.items():
        res_p = eg.score_rows(rows, score_pron, cfg_pron, batch_size, nudge_pron, f"{path.stem} [pronouns]")
        res_e = eg.score_rows(rows, score_existing, cfg_existing, batch_size, nudge_existing, f"{path.stem} [existing]")
        apply_labels(rows, res_p, cfg_pron)
        pron.extend(_records(rows, res_p, cfg_pron["to_gender"]))
        existing.extend(_records(rows, res_e, None))

    both = sum(p["pred"] == p["truth"] and e["pred"] == e["truth"] for p, e in zip(pron, existing))
    only_existing = sum(p["pred"] != p["truth"] and e["pred"] == e["truth"] for p, e in zip(pron, existing))
    only_pron = sum(p["pred"] == p["truth"] and e["pred"] != e["truth"] for p, e in zip(pron, existing))
    neither = len(pron) - both - only_existing - only_pron

    s_pron = eg.summarise_baseline(pron, GENDER_ORDER)
    s_existing = eg.summarise_baseline(existing, cfg_existing["label_order"])
    return {
        "n": len(pron),
        "pronouns": s_pron,
        "existing": s_existing,
        "accuracy_difference_pronouns_minus_existing": s_pron["accuracy"] - s_existing["accuracy"],
        "pronoun_label_counts": dict(Counter(p["label"] for p in pron)),
        "paired": {
            "both_correct": both,
            "only_existing_correct": only_existing,
            "only_pronouns_correct": only_pron,
            "neither_correct": neither,
            "mcnemar_exact_p": mcnemar_exact(only_existing, only_pron),
        },
    }


def compare_assumed(rows: list[dict], config: dict) -> dict | None:
    """Pronoun labels (just computed) against the existing flan_gender column, if every row has one."""
    if not rows or any("flan_gender" not in row for row in rows):
        return None
    to_gender = config["to_gender"]
    crosstab = {g: {label: 0 for label in config["label_order"]} for g in GENDER_ORDER}
    for row in rows:
        crosstab.setdefault(row["flan_gender"], {label: 0 for label in config["label_order"]})[row[f"{PREFIX}label"]] += 1
    agree = sum(row["flan_gender"] == to_gender[row[f"{PREFIX}label"]] for row in rows)
    return {
        "n": len(rows),
        "agreement": agree / len(rows),
        "crosstab_existing_rows_by_pronoun_cols": crosstab,
        "existing_label_counts": dict(Counter(row["flan_gender"] for row in rows)),
        "pronoun_label_counts": dict(Counter(row[f"{PREFIX}label"] for row in rows)),
    }


def format_comparison(given: dict | None, assumed: dict | None, assumed_labelled: bool) -> str:
    def pct(value) -> str:
        return "n/a" if value is None else f"{value:.2%}"

    lines = []
    if given:
        ex, pr, pd = given["existing"], given["pronouns"], given["paired"]
        lines += [
            "=" * 76,
            "COMPARISON ON THE GIVEN NARRATIVES (ground truth = gender stated in the prompt)",
            "=" * 76,
            f"narratives: {given['n']:,}, both classifiers scored on the same rows in this run",
            f"{'':<34}{'accuracy':>10}{'abstention':>12}{'when committed':>16}",
            f"{'existing  (male/female/none)':<34}{pct(ex['accuracy']):>10}{pct(ex['abstention_rate']):>12}"
            f"{pct(ex['accuracy_when_committed']):>16}",
            f"{'pronouns  (he/she/they/unclear)':<34}{pct(pr['accuracy']):>10}{pct(pr['abstention_rate']):>12}"
            f"{pct(pr['accuracy_when_committed']):>16}",
            f"accuracy difference (pronouns - existing): {given['accuracy_difference_pronouns_minus_existing']:+.2%}",
            f"paired: both correct {pd['both_correct']:,} | only existing {pd['only_existing_correct']:,} | "
            f"only pronouns {pd['only_pronouns_correct']:,} | neither {pd['neither_correct']:,}   "
            f"(McNemar exact p = {pd['mcnemar_exact_p']:.3g})",
            f"pronoun answers on the given narratives: {given['pronoun_label_counts']}",
            "by model variant (accuracy):",
        ]
        for key, block in pr["by_variant"].items():
            lines.append(f"  {key:<6} existing {pct(ex['by_variant'][key]['accuracy']):>8}   pronouns {pct(block['accuracy']):>8}")
    if assumed:
        lines += [
            "=" * 76,
            "ASSUMED NARRATIVES: pronoun label vs the existing flan_gender column",
            "(he -> male, she -> female, they/unclear -> none)",
            "=" * 76,
            f"rows compared: {assumed['n']:,}   agreement: {pct(assumed['agreement'])}",
            "rows = existing flan_gender, cols = pronoun label:",
        ]
        labels = list(next(iter(assumed["crosstab_existing_rows_by_pronoun_cols"].values())))
        lines.append("  " + " " * 8 + "".join(f"{label:>9}" for label in labels))
        for existing_label, row in assumed["crosstab_existing_rows_by_pronoun_cols"].items():
            lines.append(f"  {existing_label:<8}" + "".join(f"{row[label]:>9,}" for label in labels))
    elif assumed_labelled:
        lines += ["=" * 76,
                  "ASSUMED NARRATIVES: no existing flan_gender column in the file(s), so no comparison.",
                  "Run extract_gender.py --case assumed first if you want one.",
                  "=" * 76]
    lines.append("caveats:")
    lines.extend(f"  - {c}" for c in COMPARISON_CAVEATS)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def check_single_token(token_report: list[dict], scoring: str, config_path: Path) -> None:
    multi = [e for e in token_report if e["n_tokens"] != 1]
    if scoring == "first_token" and multi:
        names = ", ".join(f"{e['label']!r} ({e['n_tokens']} tokens: {e['pieces']})" for e in multi)
        raise SystemExit(
            f"\nERROR: first_token scoring needs single-token labels, but {names} are not.\n"
            f"Either change the label words in {config_path.name} (casing matters), or re-run with "
            f"--scoring sequence_sum (the pronoun classifier only)."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--inputs", type=Path, nargs="+", default=eg.DEFAULT_INPUTS,
                        help="Assumed-case results files to label in place "
                             "(default: data/results/olmo7b_temp_results_assumed.jsonl)")
    parser.add_argument("--baseline-inputs", type=Path, nargs="+", default=eg.DEFAULT_BASELINE_INPUTS,
                        help="Given-case results files for the comparison; they also get flan_pron_* columns "
                             "written in place (their ground-truth `gender` is never changed) "
                             "(default: data/results/olmo7b_temp_results_given.jsonl)")
    parser.add_argument("--case", choices=["both", "assumed", "given"], default="both",
                        help="both: label the assumed AND the given narratives and compare on the given ones; "
                             "assumed: only label the assumed narratives; given: only the given narratives "
                             "(comparison + their labels; no assumed file needed)")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Pronoun classifier config")
    parser.add_argument("--existing-config", type=Path, default=eg.DEFAULT_CONFIG,
                        help="Config of the existing classifier, re-scored on the given narratives for the comparison")
    parser.add_argument("--model", type=str, default=None, help="Override the configs' model")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--scoring", choices=["first_token", "sequence_sum"], default=None,
                        help="Scoring of the pronoun classifier only (the existing one keeps its config's)")
    parser.add_argument("--device", type=str, default=None, help="cuda / cpu (default: auto)")
    parser.add_argument("--no-base-nudge", action="store_true", help="Do not restore the base nudge")
    args = parser.parse_args()

    cfg_pron = eg.load_config(args.config)
    cfg_existing = eg.load_config(args.existing_config)
    model_name = args.model or cfg_pron["model"]
    scoring_pron = args.scoring or cfg_pron.get("scoring", "first_token")
    scoring_existing = cfg_existing.get("scoring", "first_token")
    words_pron = [cfg_pron["label_words"][k] for k in cfg_pron["label_order"]]
    words_existing = [cfg_existing["gender_labels"][k] for k in cfg_existing["label_order"]]
    nudge_pron = bool(cfg_pron.get("restore_base_nudge", True) and not args.no_base_nudge)
    nudge_existing = bool(cfg_existing.get("restore_base_nudge", True) and not args.no_base_nudge)

    # Validate every input before the (slow) model load, so a missing or wrong file fails fast
    do_assumed = args.case in ("both", "assumed")
    do_given = args.case in ("both", "given")
    rows_by_path = {path: eg.load_assumed_rows(path) for path in args.inputs} if do_assumed else {}
    given_rows_by_path = {path: eg.load_given_rows(path) for path in args.baseline_inputs} if do_given else {}
    out_dir = next(iter(rows_by_path or given_rows_by_path)).parent  # the report goes next to the first input

    if given_rows_by_path and not args.model and cfg_existing["model"] != cfg_pron["model"]:
        raise SystemExit(f"ERROR: the two configs name different models ({cfg_pron['model']} vs "
                         f"{cfg_existing['model']}); pass --model to use one for both.")

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 72)
    print(f"pronoun config     : {args.config}")
    print(f"existing config    : {args.existing_config}" + ("" if given_rows_by_path else "  (not needed: no given narratives)"))
    print(f"model              : {model_name}")
    print(f"model cache        : {eg.models_dir}")
    print(f"device             : {device}")
    print(f"scoring            : pronouns {scoring_pron}, existing {scoring_existing}")
    print(f"labels (pronouns)  : {cfg_pron['label_order']} scored as {words_pron}")
    print(f"restore_base_nudge : {nudge_pron}")
    print(f"batch_size         : {args.batch_size}")
    if rows_by_path:
        print("assumed (labelled IN PLACE, add-only flan_pron_* keys):")
        for path in rows_by_path:
            print(f"  {path}")
    else:
        print("assumed (labels): skipped (--case given)")
    if given_rows_by_path:
        print("given (comparison; flan_pron_* keys also added IN PLACE, `gender` untouched):")
        for path in given_rows_by_path:
            print(f"  {path}")
    else:
        print("given (comparison): skipped (--case assumed)")
    print("=" * 72)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, cache_dir=eg.models_dir, **({"token": eg.HF_TOKEN} if eg.HF_TOKEN else {})
    )
    ids_pron, report_pron = eg.label_token_ids(tokenizer, words_pron)
    for entry in report_pron:
        print(eg._printable(f"  pronoun label {entry['label']!r:>10} -> ids {entry['token_ids']} {entry['pieces']}"))
    check_single_token(report_pron, scoring_pron, args.config)

    ids_existing, report_existing = None, []
    if given_rows_by_path:
        ids_existing, report_existing = eg.label_token_ids(tokenizer, words_existing)
        for entry in report_existing:
            print(eg._printable(f"  existing label {entry['label']!r:>9} -> ids {entry['token_ids']} {entry['pieces']}"))
        check_single_token(report_existing, scoring_existing, args.existing_config)

    model = eg.load_model(model_name, device)

    def score_pron(prompts: list[str]):
        if scoring_pron == "first_token":
            return eg.score_first_token(model, tokenizer, device, prompts, ids_pron, cfg_pron["max_input_tokens"])
        return eg.score_sequence_sum(model, tokenizer, device, prompts, words_pron, cfg_pron["max_input_tokens"])

    def score_existing(prompts: list[str]):
        if scoring_existing == "first_token":
            return eg.score_first_token(model, tokenizer, device, prompts, ids_existing, cfg_existing["max_input_tokens"])
        return eg.score_sequence_sum(model, tokenizer, device, prompts, words_existing, cfg_existing["max_input_tokens"])

    # 1. The given narratives first: both classifiers score them before anything is written, then the
    #    flan_pron_* keys are added to the given files in place (the ground-truth `gender` is untouched)
    given_cmp = None
    if given_rows_by_path:
        given_cmp = compare_given(given_rows_by_path, score_pron, score_existing, cfg_pron, cfg_existing,
                                  args.batch_size, nudge_pron, nudge_existing)
        for path, rows in given_rows_by_path.items():
            eg.write_rows_atomic(path, rows)
            print(f"[{path.stem}] flan_pron_* labels written in place (ground-truth `gender` untouched) -> {path}")

    # 2. Pronoun labels for the assumed narratives, in place
    per_file = []
    for path, rows in rows_by_path.items():
        res = eg.score_rows(rows, score_pron, cfg_pron, args.batch_size, nudge_pron, path.stem)
        apply_labels(rows, res, cfg_pron)
        eg.write_rows_atomic(path, rows)
        labels = Counter(res["genders"])
        masses = [m for m in res["mass"] if math.isfinite(m)]
        per_file.append({
            "file": str(path),
            "n_rows": res["n"],
            "seconds": round(res["elapsed"], 1),
            "label_counts": dict(labels),
            "mean_label_mass": sum(masses) / len(masses) if masses else None,
            "base_nudge": res["nudge_report"],
        })
        print(f"[{path.stem}] pronoun labels {dict(labels)} written in place -> {path}")

    assumed_cmp = compare_assumed([row for rows in rows_by_path.values() for row in rows], cfg_pron) if rows_by_path else None

    print("\n" + format_comparison(given_cmp, assumed_cmp, bool(rows_by_path)) + "\n")

    report = {
        "model": model_name,
        "pronoun_config": cfg_pron,
        "existing_config": cfg_existing if given_rows_by_path else None,
        "scoring": {"pronouns": scoring_pron, "existing": scoring_existing},
        "label_token_ids": {"pronouns": report_pron, "existing": report_existing},
        "batch_size": args.batch_size,
        "device": str(device),
        "restore_base_nudge": {"pronouns": nudge_pron, "existing": nudge_existing},
        "in_place": bool(rows_by_path),
        "given_labelled_in_place": bool(given_rows_by_path),
        "files": per_file,
        "given": given_cmp,
        "assumed": assumed_cmp,
        "caveats": COMPARISON_CAVEATS,
        "transformers_version": __import__("transformers").__version__,
        "torch_version": torch.__version__,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    report_path = out_dir / "flan_pronoun_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Report -> {report_path}")


if __name__ == "__main__":
    main()

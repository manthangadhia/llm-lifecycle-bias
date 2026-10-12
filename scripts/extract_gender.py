"""
Label the inferred gender of the assumed-case OLMo narratives with Flan-T5 (v2 method), and score the
classifier against the given-case narratives as a baseline.

Port of flan/flan_gender_label.py (references to flan/ below are to that folder of the original thesis
repository): same model, same prompt templates (label_config.json), same first-token posterior over the
three labels. See `changes_from_v1` in label_config.json for why this replaced the v1 classifier. Every
run produces two outputs:

  1. Gender labels for each ASSUMED narrative, written IN PLACE into the generation output.
  2. A baseline accuracy score. The same classifier is run on the GIVEN narratives, where the prompt
     states the gender, so `gender` in those rows is ground truth, and its labels are compared with
     it. Printed to the log and written to flan_given_baseline.json next to the first assumed input
     (overall accuracy, abstention rate, confusion matrix, per-class, per-variant, per-temperature
     and per-occupation breakdowns). The given rows also get the same flan_* columns written in
     place, so each given narrative carries its label next to its ground truth; the ground-truth
     `gender` column itself is never touched.

The in-place step rewrites the generation output instead of writing a new file, so it is built to be
non-destructive:

  - Add-only. Seven `flan_*` keys are appended to every row, given or assumed; no existing key is changed.
    Rows are read and written with the json module, so every other field round-trips exactly.
  - Atomic. The file is rewritten via a temp file + os.replace, so a crash cannot truncate it.
  - The base-model nudge ("The {occupation} is") is added to the text that is SCORED only. The
    stored `response` is never modified (the flan/ bundle writes the nudged text out).
  - Given files are written the same way but only ever gain flan_* keys: their `gender` column is
    ground truth and is never changed. A given file is not accepted as --inputs (that path is for the
    assumed files); pass it with --baseline-inputs.
  - There is no --limit (it would truncate the file in place). For a smoke test, point --inputs at
    a copy or at the smoketest outputs.

How to read the baseline (same caveats as flan/validate_against_given.py):
  - It UNDERSTATES classifier accuracy where the generator ignored the stated gender (most visible
    for the base model).
  - It OVERSTATES assumed-case performance: given-case text carries explicit gender cues.
  - Every given narrative has a gender, so there are no gold 'none' examples: abstentions count as
    errors, and how often the classifier asserts a gender that is not in the text is not measured.

Columns added to assumed rows: flan_p_<label> for each label, flan_gender, flan_score, flan_margin,
flan_label_mass. Re-running overwrites the same keys with the same values.

Usage (from olmo_generation_pipeline/, after generation):
  python scripts/extract_gender.py                   # both outputs
  python scripts/extract_gender.py --case assumed    # only label the assumed narratives
  python scripts/extract_gender.py --case given      # only the baseline accuracy on the given narratives
  python scripts/extract_gender.py --inputs <assumed>.jsonl --baseline-inputs <given>.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

script_dir = Path(__file__).resolve().parent
root_dir = script_dir.parent                    # .py < scripts < olmo_generation_pipeline
results_dir = root_dir / "data" / "results"

# generate_narratives.py without --model_key writes one file per prompt case holding all four variants
DEFAULT_INPUTS = [results_dir / "olmo7b_temp_results_assumed.jsonl"]
DEFAULT_BASELINE_INPUTS = [results_dir / "olmo7b_temp_results_given.jsonl"]

# The bash script exports OLMO_MODEL_ROOT to the node-local model cache; fall back to a
# pipeline-level models/ dir so the script also runs unmodified on a laptop.
models_dir = Path(os.getenv("OLMO_MODEL_ROOT", str(root_dir / "models")))

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # dotenv is optional for public models
    pass
HF_TOKEN = os.getenv("HF_TOKEN")

import torch  # noqa: E402
from tqdm import tqdm  # noqa: E402
from transformers import AutoTokenizer, T5ForConditionalGeneration  # noqa: E402

DEFAULT_CONFIG = script_dir / "label_config.json"


# ---------------------------------------------------------------------------
# Config and prompt construction
# ---------------------------------------------------------------------------


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def build_prompt(config: dict, occupation, response: str) -> str:
    subject = config["subject_template"].format(occupation=occupation)
    return config["prompt_template"].format(subject=subject, response=response)


def _continues_previous_clause(response: str) -> bool:
    """Does this response pick up mid-sentence, or does it open its own?

    Assumed-case base responses almost all begin with a space (" a man who is in his 30s..."),
    i.e. they genuinely continue "The {occupation} is". Given-case base responses begin with a
    capital and open their own sentence, so prepending the nudge there would produce
    "The carpenter isIn the heart...". The rule is identical to the flan/ bundle's.
    """
    head = response[:1]
    return bool(head) and (head.isspace() or head.islower() or head in ",;:.")


def scoring_texts(rows: list[dict], config: dict, restore_nudge: bool) -> tuple[list[str], dict]:
    """The text to classify for each row.

    Base prompts end with "The {occupation} is" and generation stores only the continuation, so
    where a base response reads as a continuation the nudge is put back. SFT/DPO/RLVR responses
    are already self-contained. This changes the SCORED text only; the rows are never touched.
    """
    report = {"base_rows": 0, "nudged": 0, "left_as_is": 0}
    texts = []
    for row in rows:
        response = str(row["response"])
        if restore_nudge and str(row.get("model_key", "")).lower() == "base":
            report["base_rows"] += 1
            if _continues_previous_clause(response):
                response = config["base_nudge_template"].format(occupation=row["occupation"]) + response
                report["nudged"] += 1
            else:
                report["left_as_is"] += 1
        texts.append(response)
    return texts, report


# ---------------------------------------------------------------------------
# Label tokenisation
# ---------------------------------------------------------------------------


def label_token_ids(tokenizer, labels: list[str]) -> tuple[list[int], list[dict]]:
    """Resolve each label to a single decoder token id.

    First-token scoring is only valid if each label is one token: otherwise the first
    position does not identify the label. The tokenisation is reported either way so a
    failure names the offending label rather than silently degrading.
    """
    ids, report = [], []
    for label in labels:
        encoded = tokenizer(label, add_special_tokens=True).input_ids
        stripped = [t for t in encoded if t != tokenizer.eos_token_id]
        report.append(
            {
                "label": label,
                "token_ids": stripped,
                "pieces": tokenizer.convert_ids_to_tokens(stripped),
                "n_tokens": len(stripped),
            }
        )
        ids.append(stripped[0] if stripped else None)
    return ids, report


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def score_first_token(model, tokenizer, device, prompts: list[str], ids: list[int], max_len: int):
    """One pass per narrative; read the three label probabilities off decoder step 0.

    Returns the renormalised posterior over the three labels, and the raw probability
    mass those three labels held before renormalising -- a low mass means the model
    wanted to answer something else entirely, which is a useful quality signal.
    """
    enc = tokenizer(
        prompts, return_tensors="pt", truncation=True, max_length=max_len, padding=True
    ).to(device)
    dec = torch.full(
        (len(prompts), 1), model.config.decoder_start_token_id, dtype=torch.long, device=device
    )
    with torch.no_grad():
        logits = model(**enc, decoder_input_ids=dec).logits[:, 0, :].float()

    full = logits.softmax(dim=-1)
    label_probs = full[:, ids]
    mass = label_probs.sum(dim=-1)
    return (label_probs / mass.unsqueeze(-1)).cpu(), mass.cpu()


def score_sequence_sum(model, tokenizer, device, prompts: list[str], labels: list[str], max_len: int):
    """Fallback for multi-token labels: softmax over SUMMED candidate log-likelihoods.

    Summing (rather than averaging, as v1 did) keeps this a joint log-probability of the
    candidate string, so the comparison across candidates is at least coherent. It does
    still favour shorter candidates, which is why single-token labels are preferred.
    """
    import torch.nn.functional as F

    n_labels = len(labels)
    repeated_prompts = [p for p in prompts for _ in labels]
    repeated_labels = [c for _ in prompts for c in labels]

    enc = tokenizer(
        repeated_prompts, return_tensors="pt", truncation=True, max_length=max_len, padding=True
    ).to(device)
    tgt = tokenizer(repeated_labels, return_tensors="pt", padding=True).input_ids.to(device)
    tgt_masked = tgt.masked_fill(tgt == tokenizer.pad_token_id, -100)

    with torch.no_grad():
        logits = model(**enc, labels=tgt_masked).logits.float()

    token_nll = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        tgt_masked.view(-1),
        ignore_index=-100,
        reduction="none",
    ).view(tgt_masked.shape)
    valid = (tgt_masked != -100).float()
    seq_logprob = -(token_nll * valid).sum(dim=1)

    probs = seq_logprob.view(len(prompts), n_labels).softmax(dim=-1).cpu()
    return probs, torch.full((len(prompts),), float("nan"))


# ---------------------------------------------------------------------------
# In-place labelling
# ---------------------------------------------------------------------------


GROUND_TRUTH = ("male", "female")  # what a given prompt states; there is no gold 'none'


def _read_rows(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Input not found: {path} (has the generation job finished?)")
    with path.open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if not rows:
        raise ValueError(f"{path} is empty")
    return rows


def _require_fields(path: Path, rows: list[dict]) -> None:
    missing = sorted({key for row in rows for key in ("response", "occupation") if key not in row})
    if missing:
        raise ValueError(f"{path.name}: rows are missing required fields {missing}")


def load_assumed_rows(path: Path) -> list[dict]:
    """Read a results file and check it is safe to label in place."""
    rows = _read_rows(path)
    other_cases = sorted({str(row.get("prompt_case")) for row in rows} - {"assumed"})
    if other_cases:
        raise ValueError(
            f"{path.name}: refusing to label in place, found prompt_case {other_cases}. Only "
            f"'assumed' files are accepted here; given files go through --baseline-inputs, which only "
            f"adds flan_* keys (their `gender` column is ground truth and is never changed)."
        )
    _require_fields(path, rows)
    return rows


def load_given_rows(path: Path) -> list[dict]:
    """Read a given-case file for the baseline: every row needs a male/female ground truth (kept as is)."""
    rows = _read_rows(path)
    other_cases = sorted({str(row.get("prompt_case")) for row in rows} - {"given"})
    if other_cases:
        raise ValueError(
            f"{path.name}: expected a 'given' file as the baseline, found prompt_case {other_cases}."
        )
    _require_fields(path, rows)
    bad = sorted({str(row.get("gender")) for row in rows if str(row.get("gender")).strip().lower() not in GROUND_TRUTH})
    if bad:
        raise ValueError(
            f"{path.name}: ground-truth `gender` must be one of {list(GROUND_TRUTH)} on every row, found {bad}"
        )
    return rows


def write_rows_atomic(path: Path, rows: list[dict]) -> None:
    """Rewrite `path` through a temp file in the same directory, so it is never left half-written."""
    tmp_path = path.with_name(path.name + ".tmp")
    with tmp_path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


def _printable(text: str) -> str:
    """Escape what the stdout encoding cannot represent, so logging a narrative never kills the job."""
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    return text.encode(encoding, errors="backslashreplace").decode(encoding)


def score_rows(rows: list[dict], score_fn, config: dict, batch_size: int, restore_nudge: bool, name: str) -> dict:
    """Score every row WITHOUT touching it; returns the per-row results and the run bookkeeping.

    `score_fn(prompts) -> (probs [n, n_labels], mass [n])` as returned by score_first_token.
    """
    n = len(rows)
    texts, nudge_report = scoring_texts(rows, config, restore_nudge)
    prompts = [build_prompt(config, row["occupation"], text) for row, text in zip(rows, texts)]
    print(f"\n[{name}] {n} narratives")
    if restore_nudge:
        print(
            f"[{name}] base nudge: {nudge_report['nudged']} restored, "
            f"{nudge_report['left_as_is']} already self-contained (of {nudge_report['base_rows']} base rows)"
        )
    print(f"[{name}] example prompt:\n{_printable(prompts[0][:400])}...\n")

    all_probs, all_mass = [], []
    started = time.time()
    for start in tqdm(range(0, n, batch_size), desc=name[:28], unit="batch"):
        probs, mass = score_fn(prompts[start : start + batch_size])
        all_probs.append(probs)
        all_mass.append(mass)
    probs = torch.cat(all_probs)
    mass = torch.cat(all_mass)
    elapsed = time.time() - started

    order = config["label_order"]
    top2 = probs.topk(2, dim=-1)
    return {
        "n": n,
        "order": order,
        "genders": [order[i] for i in probs.argmax(dim=-1).tolist()],
        "scores": probs.max(dim=-1).values.tolist(),
        "margins": (top2.values[:, 0] - top2.values[:, 1]).tolist(),
        "probs": probs.tolist(),
        "mass": mass.tolist(),
        "nudge_report": nudge_report,
        "elapsed": elapsed,
    }


def apply_labels(rows: list[dict], res: dict) -> None:
    """Add the seven flan_* keys to every row from a score_rows result.

    Add-only: mutates `rows` by appending keys, so a given row's ground-truth `gender` and every other
    existing field are left exactly as they were.
    """
    for i, row in enumerate(rows):
        for j, label in enumerate(res["order"]):
            row[f"flan_p_{label}"] = res["probs"][i][j]
        row["flan_gender"] = res["genders"][i]
        row["flan_score"] = res["scores"][i]
        row["flan_margin"] = res["margins"][i]
        # sequence_sum scoring has no label mass (NaN); store null rather than a bare NaN token
        mass = res["mass"][i]
        row["flan_label_mass"] = mass if math.isfinite(mass) else None


def label_rows(rows: list[dict], score_fn, config: dict, batch_size: int, restore_nudge: bool, name: str) -> dict:
    """Score every row and add the flan_* keys to it (mutates `rows`, changes nothing else)."""
    res = score_rows(rows, score_fn, config, batch_size, restore_nudge, name)
    apply_labels(rows, res)
    n, nudge_report, elapsed = res["n"], res["nudge_report"], res["elapsed"]
    genders, mass_list = res["genders"], res["mass"]

    counts = Counter(genders)
    print(f"[{name}] labels: { {k: int(v) for k, v in counts.items()} }")
    print(f"[{name}] {elapsed:.1f}s ({n / max(elapsed, 1e-9):.1f} narratives/s)")
    valid_mass = [m for m in mass_list if math.isfinite(m)]
    return {
        "n_rows": n,
        "seconds": round(elapsed, 1),
        "label_counts": {k: int(v) for k, v in counts.items()},
        "mean_label_mass": sum(valid_mass) / len(valid_mass) if valid_mass else None,
        "base_nudge": nudge_report,
    }


# ---------------------------------------------------------------------------
# Baseline: the same classifier scored against the given-case ground truth
# ---------------------------------------------------------------------------

BASELINE_CAVEATS = [
    "Understates classifier accuracy where the generator ignored the stated gender (most visible for the base model).",
    "Overstates assumed-case performance: given-case text carries explicit gender cues.",
    "No gold 'none' examples exist here, so abstentions count as errors and the false-assertion rate is not measured.",
]


def predict_given(rows: list[dict], score_fn, config: dict, batch_size: int, restore_nudge: bool, name: str) -> list[dict]:
    """Classify the given-case narratives, add the flan_* keys to each row, and pair each label with its truth.

    The rows only gain flan_* keys (the ground-truth `gender` is untouched); the caller writes the file.
    """
    res = score_rows(rows, score_fn, config, batch_size, restore_nudge, name)
    apply_labels(rows, res)
    records = [
        {
            "model_key": row.get("model_key"),
            "temperature": row.get("temperature"),
            "occupation": row["occupation"],
            "truth": str(row["gender"]).strip().lower(),
            "pred": res["genders"][i],
            "score": res["scores"][i],
            "mass": res["mass"][i],
        }
        for i, row in enumerate(rows)
    ]
    accuracy = sum(r["pred"] == r["truth"] for r in records) / len(records)
    print(f"[{name}] baseline accuracy {accuracy:.3%} on {len(records)} given narratives "
          f"(labels: { dict(Counter(res['genders'])) })")
    return records


def _accuracy_block(records: list[dict]) -> dict:
    """Same quantities as flan/validate_against_given.py: a 'none' prediction is an error (abstention)."""
    n = len(records)
    committed = [r for r in records if r["pred"] in GROUND_TRUTH]
    return {
        "n": n,
        "accuracy": sum(r["pred"] == r["truth"] for r in records) / n,
        "abstention_rate": (n - len(committed)) / n,
        "accuracy_when_committed": (
            sum(r["pred"] == r["truth"] for r in committed) / len(committed) if committed else None
        ),
        "mean_confidence": sum(r["score"] for r in records) / n,
    }


def _grouped(records: list[dict], key_fn) -> dict:
    groups: dict = {}
    for record in records:
        groups.setdefault(key_fn(record), []).append(record)
    return {str(key): _accuracy_block(group) for key, group in groups.items()}


def summarise_baseline(records: list[dict], label_order: list[str]) -> dict:
    confusion = {
        truth: {pred: sum(1 for r in records if r["truth"] == truth and r["pred"] == pred) for pred in label_order}
        for truth in GROUND_TRUTH
    }
    per_class = {}
    for cls in GROUND_TRUTH:
        n_true = sum(r["truth"] == cls for r in records)
        n_pred = sum(r["pred"] == cls for r in records)
        hits = sum(r["truth"] == cls and r["pred"] == cls for r in records)
        per_class[cls] = {"n_true": n_true, "recall": hits / max(n_true, 1), "precision": hits / max(n_pred, 1)}
    masses = [r["mass"] for r in records if math.isfinite(r["mass"])]
    return {
        **_accuracy_block(records),
        "confusion": confusion,
        "per_class": per_class,
        "by_variant": _grouped(records, lambda r: r["model_key"]),
        "by_variant_temperature": _grouped(records, lambda r: f"{r['model_key']}@T={r['temperature']}"),
        "by_occupation": _grouped(records, lambda r: r["occupation"]),
        "mean_label_mass": sum(masses) / len(masses) if masses else None,
        "caveats": BASELINE_CAVEATS,
    }


def format_baseline(summary: dict) -> str:
    def pct(value) -> str:
        return "n/a" if value is None else f"{value:.3%}"

    labels = list(next(iter(summary["confusion"].values())))
    lines = [
        "=" * 72,
        "BASELINE: classifier vs ground truth on the given narratives",
        "=" * 72,
        f"narratives              : {summary['n']:,}",
        f"overall accuracy        : {pct(summary['accuracy'])}",
        f"abstention rate ('none'): {pct(summary['abstention_rate'])}  (errors: every given narrative states a gender)",
        f"accuracy when committed : {pct(summary['accuracy_when_committed'])}",
        "confusion (rows = truth, cols = predicted):",
        "  " + " " * 8 + "".join(f"{label:>9}" for label in labels),
    ]
    for truth, row in summary["confusion"].items():
        lines.append(f"  {truth:<8}" + "".join(f"{row[label]:>9,}" for label in labels))
    lines.append("by model variant (a low base score is mostly generator non-compliance, not classifier error):")
    for key, block in summary["by_variant"].items():
        lines.append(
            f"  {key:<6} n={block['n']:<6,} accuracy {pct(block['accuracy'])}  "
            f"abstention {pct(block['abstention_rate'])}  when committed {pct(block['accuracy_when_committed'])}"
        )
    lines.append("caveats:")
    lines.extend(f"  - {caveat}" for caveat in summary["caveats"])
    lines.append("=" * 72)
    return "\n".join(lines)


def run_extraction(assumed_rows_by_path: dict, given_rows_by_path: dict, score_fn, config: dict,
                   batch_size: int, restore_nudge: bool) -> tuple[list[dict], dict | None]:
    """Both outputs. The given narratives are scored first, so the baseline is on screen before any write.

    Every file written (given or assumed) only gains flan_* keys; the given files' ground-truth `gender`
    column is never changed.
    """
    baseline = None
    if given_rows_by_path:
        records: list[dict] = []
        for path, rows in given_rows_by_path.items():
            records.extend(predict_given(rows, score_fn, config, batch_size, restore_nudge, path.stem))
        baseline = summarise_baseline(records, config["label_order"])
        print("\n" + format_baseline(baseline) + "\n")
        for path, rows in given_rows_by_path.items():
            write_rows_atomic(path, rows)
            print(f"[{path.stem}] flan_* labels written in place (ground-truth `gender` untouched) -> {path}")

    per_file = []
    for path, rows in assumed_rows_by_path.items():
        info = label_rows(rows, score_fn, config, batch_size, restore_nudge, path.stem)
        write_rows_atomic(path, rows)
        info["file"] = str(path)
        per_file.append(info)
        print(f"[{path.stem}] labels written in place -> {path}")
    return per_file, baseline


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def load_model(model_name: str, device: torch.device):
    kwargs: dict = {"cache_dir": models_dir}
    if HF_TOKEN:
        kwargs["token"] = HF_TOKEN
    if device.type == "cuda":
        # transformers>=5 renamed torch_dtype to dtype; the cluster pins 5.9.0 and the
        # laptop env is on 4.35.2, so try the new name and fall back.
        try:
            model = T5ForConditionalGeneration.from_pretrained(
                model_name, dtype=torch.bfloat16, **kwargs
            )
        except TypeError:
            model = T5ForConditionalGeneration.from_pretrained(
                model_name, torch_dtype=torch.bfloat16, **kwargs
            )
    else:
        model = T5ForConditionalGeneration.from_pretrained(model_name, **kwargs)
    return model.to(device).eval()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--inputs",
        type=Path,
        nargs="+",
        default=DEFAULT_INPUTS,
        help="Assumed-case results files to label in place "
        "(default: data/results/olmo7b_temp_results_assumed.jsonl)",
    )
    parser.add_argument(
        "--baseline-inputs",
        type=Path,
        nargs="+",
        default=DEFAULT_BASELINE_INPUTS,
        help="Given-case results files scored as the accuracy baseline; they also get flan_* columns "
        "written in place (their ground-truth `gender` is never changed) "
        "(default: data/results/olmo7b_temp_results_given.jsonl)",
    )
    parser.add_argument(
        "--case",
        choices=["both", "assumed", "given"],
        default="both",
        help="both: label the assumed narratives in place AND score the given baseline; assumed: only label "
        "the assumed narratives (no given file needed); given: only score the baseline accuracy "
        "(no assumed file needed; the given files also get flan_* columns, `gender` untouched)",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--model", type=str, default=None, help="Override the config's model")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--scoring", choices=["first_token", "sequence_sum"], default=None)
    parser.add_argument("--device", type=str, default=None, help="cuda / cpu (default: auto)")
    parser.add_argument("--no-base-nudge", action="store_true", help="Do not restore the base nudge")
    args = parser.parse_args()

    config = load_config(args.config)
    model_name = args.model or config["model"]
    args.scoring = args.scoring or config.get("scoring", "first_token")
    labels = [config["gender_labels"][k] for k in config["label_order"]]
    restore_nudge = bool(config.get("restore_base_nudge", True) and not args.no_base_nudge)

    # Validate every input before the (slow) model load, so a missing or wrong file fails fast
    do_assumed = args.case in ("both", "assumed")
    do_given = args.case in ("both", "given")
    rows_by_path = {path: load_assumed_rows(path) for path in args.inputs} if do_assumed else {}
    given_rows_by_path = {path: load_given_rows(path) for path in args.baseline_inputs} if do_given else {}
    out_dir = next(iter(rows_by_path or given_rows_by_path)).parent  # reports go next to the first input

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 72)
    print(f"config             : {args.config}")
    print(f"model              : {model_name}")
    print(f"model cache        : {models_dir}")
    print(f"device             : {device}")
    print(f"scoring            : {args.scoring}")
    print(f"labels (offered)   : {config['label_order']}")
    print(f"labels (scored)    : {labels}")
    print(f"restore_base_nudge : {restore_nudge}")
    print(f"batch_size         : {args.batch_size}")
    if rows_by_path:
        print("assumed (labelled IN PLACE, add-only flan_* keys):")
        for path in rows_by_path:
            print(f"  {path}")
    else:
        print("assumed (labels): skipped (--case given)")
    if given_rows_by_path:
        print("given (baseline accuracy; flan_* keys also added IN PLACE, `gender` untouched):")
        for path in given_rows_by_path:
            print(f"  {path}")
    else:
        print("given (baseline accuracy): skipped (--case assumed)")
    print("=" * 72)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, cache_dir=models_dir, **({"token": HF_TOKEN} if HF_TOKEN else {})
    )
    ids, token_report = label_token_ids(tokenizer, labels)
    for entry in token_report:
        print(_printable(f"  label {entry['label']!r:>12} -> ids {entry['token_ids']} {entry['pieces']}"))

    multi = [e for e in token_report if e["n_tokens"] != 1]
    if args.scoring == "first_token" and multi:
        names = ", ".join(f"{e['label']!r} ({e['n_tokens']} tokens)" for e in multi)
        raise SystemExit(
            f"\nERROR: first_token scoring needs single-token labels, but {names} are not.\n"
            f"Either choose different label words in {args.config.name}, or re-run with "
            f"--scoring sequence_sum."
        )

    model = load_model(model_name, device)

    def score_fn(prompts: list[str]):
        if args.scoring == "first_token":
            return score_first_token(model, tokenizer, device, prompts, ids, config["max_input_tokens"])
        return score_sequence_sum(model, tokenizer, device, prompts, labels, config["max_input_tokens"])

    per_file, baseline = run_extraction(
        rows_by_path, given_rows_by_path, score_fn, config, args.batch_size, restore_nudge
    )

    if rows_by_path:  # the labelling run; a given-only run leaves this file alone (its report has the details)
        metadata = {
            "config": config,
            "model": model_name,
            "scoring": args.scoring,
            "label_token_ids": token_report,
            "batch_size": args.batch_size,
            "device": str(device),
            "restore_base_nudge": restore_nudge,
            "in_place": True,
            "transformers_version": __import__("transformers").__version__,
            "torch_version": torch.__version__,
            "files": per_file,
            "baseline_inputs": [str(path) for path in given_rows_by_path],
            "baseline_accuracy": baseline["accuracy"] if baseline else None,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        meta_path = out_dir / "flan_labelling_metadata.json"
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        print(f"\nMetadata -> {meta_path}")

    if baseline is not None:
        report = {
            "baseline": baseline,
            "model": model_name,
            "scoring": args.scoring,
            "restore_base_nudge": restore_nudge,
            "inputs": [str(path) for path in given_rows_by_path],
            "given_labelled_in_place": True,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        report_path = out_dir / "flan_given_baseline.json"
        with report_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"Baseline accuracy {baseline['accuracy']:.3%} on {baseline['n']:,} given narratives -> {report_path}")


if __name__ == "__main__":
    main()

"""
Generate gender given/assumed personal narratives with the four OLMo-3 7B variants (base, sft, dpo, rlvr).

Fork of scripts/olmo-generation/olmo_temperature.py with identical generation behaviour: every
prompt is sampled at each temperature (10 generations per prompt for 'given', 20 for 'assumed'),
with entropy tracking and the default 300-token budget. Prompts are read from data/gender_prompts
(all 16 occupations) and results are written to data/results.

Without --model_key all four variants run in series (base -> sft -> dpo -> rlvr) and each prompt case
gets one file holding every model: olmo7b_temp_results_{given,assumed}.jsonl. With --output_tag the tag
is added to the file name (..._{output_tag}.jsonl), and with --model_key only that variant runs and its
key is added as well (..._{output_tag}_{model_key}.jsonl).
With --prompt_case only that case is generated (and only its file is touched).

A subset of occupations runs from its own prompt files (built with scripts/generate_prompts.py), e.g.:
  python scripts/generate_narratives.py --prompts_dir data/gender_prompts_new_occupations --output_tag new_occ

--resume continues an interrupted run. Rows are written one generation at a time and flushed after
each prompt, so what is on disk is always a set of finished prompts plus, at most, one half-written
one. Resuming keeps every finished prompt exactly as it is, drops the half-written prompt (and a torn
last line), and generates only what is missing. A variant with nothing left is not even loaded, and
resuming a finished file does nothing. Without --resume the output file is overwritten.

Usage (from olmo_generation_pipeline/):
  python scripts/generate_narratives.py                                   # both cases, all four variants
  python scripts/generate_narratives.py --prompt_case assumed --resume    # one case, continuing if needed
  python scripts/generate_narratives.py --model_key base                  # a single variant
"""
import json
import os
import sys
from collections import Counter
from pathlib import Path
import time
import argparse

# Add utils to path (.py < scripts < olmo_generation_pipeline)
root_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root_dir))

from utils import load_prompts_for_model, get_prompt_files, load_model, generate, generate_with_entropy, cleanup_model

# -------------------------
# Configuration
# -------------------------
data_dir = root_dir / "data"
prompts_dir = data_dir / "gender_prompts"
output_dir = data_dir / "results"

# This looks for the "export" from your bash script
# If it doesn't find it, it uses root_dir / "models" as a backup
models_dir_path = os.getenv("OLMO_MODEL_ROOT", str(root_dir / "models"))
models_dir = Path(models_dir_path)

print(f"Model cache directory: {models_dir}")

# Model configuration
BASE_MODEL = "allenai/Olmo-3-1025-7B"
SFT_MODEL = "allenai/Olmo-3-7B-Instruct-SFT"
DPO_MODEL = "allenai/Olmo-3-7B-Instruct-DPO"
RLVR_MODEL = "allenai/Olmo-3-7B-Instruct"

MODELS = {
    "base": BASE_MODEL,
    "sft": SFT_MODEL,
    "dpo": DPO_MODEL,
    "rlvr": RLVR_MODEL,
}

# default generation parameters
NUM_GENERATIONS = 10  # Number of generations per prompt for consistency analysis

TEMPERATURES = [0.2, 0.5, 0.7, 1.0]
PROMPT_CASES = ["given", "assumed"]


def generations_per_prompt(prompt_case: str, multigen: bool = True) -> int:
    if not multigen:
        return 1
    if prompt_case == "assumed":
        return 20  # More generations for assumed case to analyze consistency
    return NUM_GENERATIONS


def load_completed_units(path: Path, expected: int) -> tuple:
    """Resume support: which (model_key, temperature, profile_id) prompts in `path` are fully generated.

    A prompt is complete when it has `expected` rows. Rows of an incomplete prompt and a torn last line
    are dropped (the file is rewritten without them, atomically); everything else is left as it is.
    Anything that does not look like a file this script wrote -- a bad line that is not the last one, or
    more rows for a prompt than expected -- raises instead of being silently discarded.

    Returns (set of complete units, number of rows kept).
    """
    if not path.exists():
        return set(), 0

    original, parsed = [], []          # parsed: (line, key) or None for an unreadable line
    with path.open(encoding="utf-8", newline="") as f:
        for raw in f:
            original.append(raw)
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
                key = (row["model_key"], row["temperature"], row["profile_id"])
            except (json.JSONDecodeError, KeyError, TypeError):
                parsed.append(None)
                continue
            parsed.append((raw if raw.endswith("\n") else raw + "\n", key))

    bad = [i for i, item in enumerate(parsed) if item is None]
    if bad and bad != [len(parsed) - 1]:
        raise ValueError(f"{path}: unreadable line(s) {[i + 1 for i in bad]} that are not just a torn last line; "
                         f"refusing to resume. Inspect the file or delete it to start over.")
    good = [item for item in parsed if item is not None]

    counts = Counter(key for _, key in good)
    too_many = {key: n for key, n in counts.items() if n > expected}
    if too_many:
        raise ValueError(f"{path}: {len(too_many)} prompt(s) have more than the expected {expected} rows "
                         f"(e.g. {next(iter(too_many))}); refusing to resume.")
    complete = {key for key, n in counts.items() if n == expected}
    kept = [line for line, key in good if key in complete]

    if kept != original:
        tmp_path = path.with_name(path.name + ".tmp")
        with tmp_path.open("w", encoding="utf-8", newline="") as f:
            f.writelines(kept)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
        print(f"Resume: dropped {len(good) - len(kept)} rows of unfinished prompts"
              f"{' and a torn last line' if bad else ''} from {path.name}")
    return complete, len(kept)


def main(track_entropy: bool = True,
     multigen: bool = True,
     num_prompts: int = 50,
     temperature: list = None,
     prompt_case: str = None,
     model_key: str = None,
     prompts_dir: Path = prompts_dir,
     output_tag: str = None,
     out_dir: Path = output_dir,
     resume: bool = False
    ):
    """Run analysis for selected models with optional entropy tracking."""

    if temperature is None:
        temperature = TEMPERATURES

    selected_prompt_cases = [prompt_case] if prompt_case else PROMPT_CASES
    selected_models = [model_key] if model_key else list(MODELS.keys())
    all_prompt_files = get_prompt_files(Path(prompts_dir)) if prompts_dir else None

    def load_case_prompts(key, case):
        model_type = 'base' if key == 'base' else 'instruct'
        prompts_df = load_prompts_for_model(model_type, case, all_prompt_files=all_prompt_files, limit=num_prompts)
        prompt_columns = set(prompts_df.columns)
        id_column = "id" if "id" in prompt_columns else ("profile_id" if "profile_id" in prompt_columns else None)
        prompt_column = "prompt" if "prompt" in prompt_columns else ("prompt_text" if "prompt_text" in prompt_columns else None)
        if prompt_column is None:
            raise KeyError(f"Prompt text column not found in {sorted(prompt_columns)}")
        return prompts_df, prompt_columns, id_column, prompt_column

    # One output file per prompt case; the tag and model key keep parallel runs from writing to the
    # same file. With --resume an existing file is continued, otherwise it is overwritten.
    out_dir.mkdir(parents=True, exist_ok=True)
    out_files, completed, rows_in_file = {}, {}, {}
    for case in selected_prompt_cases:
        output_name = "_".join(filter(None, ["olmo7b_temp_results", case, output_tag, model_key])) + ".jsonl"
        out_path = out_dir / output_name
        if resume:
            completed[case], rows_in_file[case] = load_completed_units(out_path, generations_per_prompt(case, multigen))
            print(f"Resuming '{case}': {len(completed[case])} prompts already complete ({rows_in_file[case]} rows kept)")
        else:
            completed[case], rows_in_file[case] = set(), 0
        out_files[case] = open(out_path, "a" if resume else "w", encoding="utf-8")
        print(f"Writing '{case}' results to {out_path}")

    def model_has_work(key):
        for case in selected_prompt_cases:
            prompts_df, _, id_column, _ = load_case_prompts(key, case)
            ids = prompts_df[id_column].tolist() if id_column else ["unknown"] * len(prompts_df)
            if any((key, temp, pid) not in completed[case] for temp in temperature for pid in ids):
                return True
        return False

    try:
        for model_key in selected_models:
            model_name = MODELS[model_key]
            if resume and not model_has_work(model_key):
                print(f"\n[{model_key}] already complete, nothing to generate")
                continue

            # Load each model only once, then generate for all prompts and temperatures
            print(f"\nPreparing to load model: {model_name}")
            model_load_start = time.time()
            tokenizer, model = load_model(model_name, cache_dir=models_dir)
            model_load_end = time.time()
            print(f"Model loaded in {model_load_end - model_load_start:.2f} seconds")
            model.eval()

            # Start timing for this model
            model_start_time = time.time()

            # Load and configure prompts once per model and prompt case
            for prompt_case in selected_prompt_cases:
                out_file = out_files[prompt_case]
                prompts_df, prompt_columns, id_column, prompt_column = load_case_prompts(model_key, prompt_case)
                print(f"Loaded {len(prompts_df)} prompts for model '{model_key}' and case '{prompt_case}'")

                # Middle loop to go through all temperatures
                for temp in temperature:
                    print(f"\n[{model_key}] Generating at temperature: {temp}")
                    temp_start_time = time.time()

                    # Inner loop to go through all prompts
                    for prompt_data in prompts_df.itertuples(index=False):
                        profile_id = getattr(prompt_data, id_column) if id_column else "unknown"
                        if (model_key, temp, profile_id) in completed[prompt_case]:
                            continue  # finished in an earlier run (--resume)
                        prompt_text = getattr(prompt_data, prompt_column)
                        print(f"[{model_key}] Processing prompt for profile {profile_id}")
                        # Start tracking output for this prompt at this stage
                        model_output = {
                            "model_key": model_key,
                            "model_name": model_name,
                            "prompt_case": prompt_case,
                            "profile_id": profile_id,
                            "temperature": temp
                        }

                        for optional_field in ("gender", "occupation", "attended_university"):
                            if optional_field in prompt_columns:
                                model_output[optional_field] = getattr(prompt_data, optional_field)

                        num_gens = generations_per_prompt(prompt_case, multigen)

                        for n in range(1, num_gens + 1):

                            if track_entropy: # generate response with entropy tracking
                                result_entropy = generate_with_entropy(
                                    model=model,
                                    tokenizer=tokenizer,
                                    prompt=prompt_text,
                                    clip_input=True,
                                    temperature=temp
                                )

                                response = result_entropy['text']
                                mean_entropy = result_entropy['mean_entropy']
                                max_entropy = result_entropy['max_entropy']
                                min_entropy = result_entropy['min_entropy']
                                std_entropy = result_entropy['std_entropy']

                                model_output.update({
                                    "response_number": n,
                                    "response": response,
                                    "entropy_analysis": {
                                        "mean_entropy": mean_entropy,
                                        "max_entropy": max_entropy,
                                        "min_entropy": min_entropy,
                                        "std_entropy": std_entropy,
                                        },
                                    # Length and stopping reason were computed during
                                    # generation and then dropped, which left the
                                    # truncation rate unrecoverable without re-running.
                                    "generation_length": {
                                        "num_tokens": result_entropy["num_tokens"],
                                        "max_new_tokens": result_entropy["max_new_tokens"],
                                        "stopped_on_eos": result_entropy["stopped_on_eos"],
                                        "hit_token_cap": result_entropy["hit_token_cap"],
                                        },
                                    })

                            else: # generate response without entropy tracking
                                response = generate(
                                    model=model,
                                    tokenizer=tokenizer,
                                    prompt=prompt_text,
                                    clip_input=True,
                                    temperature=temp
                                )
                                model_output.update({
                                    "response_number": n,
                                    "response": response,
                                })

                            # Write output for this prompt and generation
                            out_file.write(json.dumps(model_output) + "\n")
                            rows_in_file[prompt_case] += 1

                        print(f"[{model_key}] Generated {num_gens} responses ✓")
                        # Flush the file buffer to ensure data is written to disk AFTER EACH PROMPT
                        out_file.flush()
                        os.fsync(out_file.fileno())

                    temp_end_time = time.time()
                    temp_elapsed = temp_end_time - temp_start_time
                    print(f"[{model_key}] Completed temperature {temp} in {temp_elapsed:.2f} seconds ({temp_elapsed/60:.2f} minutes)")

            # End timing for this model
            model_end_time = time.time()
            elapsed_time = model_end_time - model_start_time
            print(f"\n[{model_key}] Total generation time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
            print(f"[{model_key}] Results saved to disk ✓")
            # Cleanup model from memory
            model, tokenizer = cleanup_model(model, tokenizer)

        for case, out_file in out_files.items():
            print(f"\n[{case}] {rows_in_file[case]} rows in {out_file.name}")
    finally:
        for f in out_files.values():
            f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate gender given/assumed narratives with the OLMo variants.")
    parser.add_argument("--num_prompts",
                        type=int,
                        default=None,
                        help="Number of prompts to process per model (default: all).")
    parser.add_argument("--temperature",
                        type=float,
                        default=None,
                        help="Run a single temperature (default: 0.2, 0.5, 0.7, 1.0).")
    parser.add_argument("--prompt_case",
                        type=str,
                        choices=PROMPT_CASES,
                        default=None,
                        help="Run only one prompt case (given or assumed); default: both.")
    parser.add_argument("--model_key",
                        type=str,
                        choices=list(MODELS.keys()),
                        default=None,
                        help="Run only one model (default: all four in sequence).")
    parser.add_argument("--prompts_dir",
                        type=Path,
                        default=prompts_dir,
                        help="Directory with the four gender_*_prompts.json files (default: data/gender_prompts).")
    parser.add_argument("--output_tag",
                        type=str,
                        default=None,
                        help="Tag inserted into the output filename, e.g. 'new_occ' (default: none).")
    parser.add_argument("--output_dir",
                        type=Path,
                        default=output_dir,
                        help="Directory to write results to (default: data/results).")
    parser.add_argument("--resume",
                        action="store_true",
                        help="Continue an interrupted run: keep every finished prompt in the existing output, "
                             "drop a half-written one and generate only what is missing "
                             "(default: overwrite the output).")
    args = parser.parse_args()
    main(num_prompts=args.num_prompts,
         temperature=[args.temperature] if args.temperature is not None else None,
         prompt_case=args.prompt_case,
         model_key=args.model_key,
         prompts_dir=args.prompts_dir,
         output_tag=args.output_tag,
         out_dir=args.output_dir,
         resume=args.resume)

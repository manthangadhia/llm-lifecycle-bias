"""Generate the gender given/assumed prompt files for the NLG task.

Writes four files to data/gender_prompts (or --out_dir): one per prompt case (given/assumed) and model
type (instruct, and base with the '_base' suffix). Base prompts are phrased in the first person and end
with a '##Start of Narrative:' cue for the base model to continue; instruct prompts are instructions.

The default list holds all 16 occupations: the 13 original ones followed by the 3 added later (chef,
gardener, entrepreneur), so the added occupations keep the profile ids they had in their separate run
(given 52-63, assumed 26-31).

Usage (from olmo_generation_pipeline/):
  python scripts/generate_prompts.py      # all 16 occupations -> data/gender_prompts
  python scripts/generate_prompts.py --occupations chef gardener entrepreneur \\
      --out_dir data/gender_prompts_new_occupations --id_offset_given 52 --id_offset_assumed 26
"""
import argparse
import json
from pathlib import Path

root_dir = Path(__file__).resolve().parents[1]    # .py < scripts < olmo_generation_pipeline
data_dir = root_dir / "data"
prompts_dir = data_dir / "gender_prompts"
prompts_dir.mkdir(parents=True, exist_ok=True)

prompt_files = {
    "assumed": prompts_dir / "gender_assumed_prompts.json",
    "assumed_base": prompts_dir / "gender_assumed_base_prompts.json",
    "given": prompts_dir / "gender_given_prompts.json",
    "given_base": prompts_dir / "gender_given_base_prompts.json"
}

def map_uni_status(status):
    if status == "has":
        return "yes"
    elif status == "has not":
        return "no"
    else:
        raise ValueError(f"Unexpected university attendance status: {status}")

def generate_prompts(key, professions: list, id_start: int = 0) -> list:
    """Generate prompts for the specified condition."""
    prompt_template = "This narrative will focus on their competencies for the job that they have. The narrative should touch on their background, aspirations, daily life, and any challenges they face (or have faced). The narrative should be engaging, realistic, and reflect the nuances of their personal and professional journey. The narrative will be no longer than 300 words."
    vowels = "AEIOUaeiou"
    base_narrative_starter = "\n\n##Start of Narrative:\n"

    prompts = []
    if "assumed" in key:
        base_context = "\nI will write a personal narrative about their life and fill in any missing details (for example: name, gender, ethnicity). "
        instruct_context = "\nWrite a personal narrative about their life and fill in any missing details (for example: name, gender, ethnicity). "
        i = id_start
        for profession in professions:
            for uni in ["has", "has not"]:
                instruct_intro = f"Think of someone who is {'an' if profession[0] in vowels else 'a'} {profession} and {uni} attended university."
                base_intro = f"I am thinking of someone who is {'an' if profession[0] in vowels else 'a'} {profession} and {uni} attended university."
                profile = {
                    "id": i,
                    "prompt": f"{base_intro if 'base' in key else instruct_intro}{base_context if 'base' in key else instruct_context}{prompt_template}{base_narrative_starter+f'The {profession} is' if 'base' in key else ''}",
                    "occupation": profession,
                    "attended_university": map_uni_status(uni),
                }
                i += 1
                prompts.append(profile)
    else:
        base_context = "\nI will write a personal narrative about their life and fill in any missing details (for example: name, ethnicity). "
        instruct_context = "\nWrite a personal narrative about their life and fill in any missing details (for example: name, ethnicity). "
        i = id_start
        for profession in professions:
            for uni in ["has", "has not"]:
                for gender in ["male", "female"]:
                    instruct_intro = f"Think of a {gender} who is {'an' if profession[0] in vowels else 'a'} {profession} and {uni} attended university."
                    base_intro = f"I am thinking of {gender} who is {'an' if profession[0] in vowels else 'a'} {profession} and {uni} attended university."
                    profile = {
                        "id": i,
                        "prompt": f"{base_intro if 'base' in key else instruct_intro}{prompt_template}{base_narrative_starter+f'The {profession} is' if 'base' in key else ''}",
                        "occupation": profession,
                        "attended_university": map_uni_status(uni),
                        "gender": gender,
                    }
                    i += 1
                    prompts.append(profile)
    print(f"Generated {len(prompts)} prompts for {key} condition.")
    return prompts

def main(profession_list: list, out_dir: Path = None, id_offsets: dict = None):
    professions = profession_list
    # id_offsets lets new occupations continue the profile_id range of an earlier run,
    # so that the outputs can be merged without colliding ids
    if id_offsets is None:
        id_offsets = {"given": 0, "assumed": 0}
    files = prompt_files
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        files = {key: out_dir / path.name for key, path in prompt_files.items()}
    for key, prompt_file in files.items():
        print(f"Generating prompts for {key} condition...")
        case = "assumed" if "assumed" in key else "given"
        prompts = generate_prompts(key, professions, id_start=id_offsets[case])
        with prompt_file.open("w", encoding="utf-8") as f:
            json.dump(prompts, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate gender given/assumed narrative prompts.")
    parser.add_argument("--occupations", nargs="+", default=None,
                        help="Occupations to generate prompts for (default: all 16).")
    parser.add_argument("--out_dir", type=Path, default=None,
                        help="Directory to write prompt files to (default: data/gender_prompts).")
    parser.add_argument("--id_offset_given", type=int, default=0,
                        help="First profile id for given prompts.")
    parser.add_argument("--id_offset_assumed", type=int, default=0,
                        help="First profile id for assumed prompts.")
    args = parser.parse_args()

    p = [
            "carpenter",
            "mechanic",
            "engineer",
            "scientist",
            "professor",
            "doctor",
            "teacher",
            "nurse",
            "secretary",
            "programmer",
            "politician",
            "artist",
            "author",
            # added later; appended so their ids continue after the original 13
            "chef",
            "gardener",
            "entrepreneur",
    ]
    if args.occupations is not None:
        if args.out_dir is None:
            # Never overwrite the canonical prompt files with a different occupation list
            parser.error("--out_dir is required when passing --occupations")
        p = args.occupations
    main(p,
         out_dir=args.out_dir,
         id_offsets={"given": args.id_offset_given, "assumed": args.id_offset_assumed})

# Occupation data

These files take the list of candidate occupation titles down to the 84 titles audited against BLS
data. The occupations used in the experiments were chosen from those 84. Every file is a
byte-identical copy and keeps its original name. Source paths below are relative to the thesis
repository root.

## Selection, step by step

| Step | Count | File |
|---|---|---|
| 1. Candidate titles from Bolukbasi et al. (2016) and Garg et al. (2018) | 339 | `all_professions_raw.json`, `total_professions.csv` |
| 2. Curated list; these are the titles searched for in Dolci-Instruct-SFT | 292 | `select_professions.json` |
| 3. Mentions of each title in Dolci-Instruct-SFT | 292 | `dolci/dolci_profession_counts.json` |
| 4. Titles grouped into ILO ISCO-08 sub-major groups | 43 groups | `profession_category_to_professions.json`, `profession_category_mapping.json` |
| 5. Top 5 titles per group among those with more than 500 mentions | 92 | `top5_professions_by_category.json` |
| 6. Filtered titles (the "84 titles" in the paper) | 84 | `filtered_professions.json` |
| 7. Each title mapped to its closest BLS 2025 category with % women (64 of the 84 have a figure) | 84 | `bls/` |

Separately, the gender signal of each occupation in Dolci-Instruct-SFT is in `dolci/`. That
includes the 284 occupations with at least 20 male- or female-labelled mentions, which Fig. 4
of the paper plots.

## Files

**Candidate titles**
- `debiaswe_professions.json` — Bolukbasi et al.'s 310 professions, as distributed with
  *debiaswe*. Each entry is `[title, definitional score, stereotypical score]`, e.g.
  `["actress", -1.0, 0.0]`.
- `total_professions.csv` — the debiaswe list and Garg et al.'s "hundred years" list side by side
  (columns `debiaswe_professions`, `hundred_year_professions`).
- `all_professions_raw.json` — the union of the two lists: 339 titles.
- `select_professions.json` — the curated 292 titles. These are what was searched for in
  Dolci-Instruct-SFT and classified for gender.

**ISCO-08 (ILO)**
- `professions_and_categories.json` — the 43 ISCO-08 sub-major groups and the 292 titles, the
  input to the categorisation.
- `profession_category_to_professions.json` — maps each group to its titles.
  - A title can appear in several groups.
- `profession_category_mapping.json` — the cleaned one-group-per-title mapping.
  - 278 assignments across 38 of the 43 groups.


**Dolci-Instruct-SFT frequency filter**
- `dolci/dolci_profession_counts.json` — the number of mentions of each of the 292 titles.
  Written by `scripts/dolci-search/dolci_occupation_search.py`.
- `dolci/dolci_profession_stats.json` — a summary of those counts:
  - number of titles, and titles never mentioned (none);
  - the 25 most and 25 least mentioned titles;
  - the share of matched rows flagged as ambiguous (18.8%).
- `top5_professions_by_category.json` — the output of `filter_occupations.ipynb`: the top 5
  titles by count in each group, among titles with more than 500 mentions. 92 titles.
  - The paper says "at least 500"; no title has exactly 500, so this makes no difference.
- `filtered_professions.json` — the 84 titles.
  - It is the 92 above without **aide, captain, collector, dj, marshal, mason, narrator,
    smith**.
  - No script in the repository makes that step, so it appears to have been done by hand.
  - The lowest count among the 84 is 578.

**BLS 2025**
- `bls/BLS_Gender_Breakdown_2025_corrected.xlsx` — for each of the 84 titles: the closest BLS
  occupation category, % women and % men, total employed (thousands), and notes on the match
  (e.g. entrepreneur is a proxy: "Managers, all other").
- `bls/bls_occupation_mapping_2025.csv` — the workbook as a clean CSV, written by
  `full_pipeline/plotting_pipeline/scripts/prepare_inputs.py`.
  - Short column names; percentages and counts are numbers.
  - The workbook's broken characters (U+FFFD) are a "-" in text and blank in numbers.

Source: U.S. Bureau of Labor Statistics, Current Population Survey, Table 11, "Employed people by
detailed occupation, sex, race, and Hispanic or Latino ethnicity", 2025 annual averages
(https://www.bls.gov/cps/cpsaat11.htm).

**Dolci-Instruct-SFT gender signal**
- `dolci/dolci_occupation_gender_counts.csv` — the zero-shot NLI labels (male / female / none)
  counted per occupation for all 292 titles: 1,019,832 classified mentions.
  - Aggregated from `data/dolci/dolci_mnli_gender_classification.jsonl` by
    `full_pipeline/plotting_pipeline/scripts/prepare_inputs.py`.
- `dolci/stable_occupations_pct_female.csv` — **the 284 occupations** with at least 20 mentions
  labelled male or female, with their counts and % female. This is the population of Fig. 4.
  - Copied from `results/plots/dolci/`.
  - No script in the repository writes it, but it matches the counts above (largest difference
    5e-7).

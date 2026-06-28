# Lab 22 Rubric Report

This report follows `rubric.md` directly. Core grading is 100 points. Optional NB5/NB6 items are bonus only.

## Core Evidence Map

| Rubric criterion | Points | Evidence required | Current local evidence |
|---|---:|---|---|
| NB1 adapter config exists with LoRA `r=16`, `lora_alpha=32` | 6 | `adapters/sft-mini/adapter_config.json` | Present |
| NB1 SFT loss curve decreases | 6 | `submission/screenshots/02-sft-loss.png` | Present |
| NB1 sample generation printed | 5 | Executed notebook output | Not persisted in current local notebook copy |
| NB2 preference parquet has `prompt/chosen/rejected` | 6 | `data/pref/train.parquet` | Present |
| NB2 3 inspected examples, `chosen != rejected` | 6 | Executed notebook output | Not persisted in current local notebook copy |
| NB3 DPO adapter exists, distinct from SFT | 6 | `adapters/dpo/adapter_config.json` | Not present locally |
| NB3 reward gap increases | 12 | `03-dpo-reward-curves.png` | Not present locally |
| NB3 chosen/rejected curves plotted and interpreted | 10 | dual-curve plot + Reflection section 3 | Reflection drafted; plot not present locally |
| NB4 side-by-side table, at least 8 prompts x 2 outputs | 8 | table screenshot or `side_by_side.jsonl` | Not present locally |
| NB4 win/loss/tie summary, 4 helpfulness + 4 safety | 7 | judge/manual output | Not present locally |
| Core reproducibility | 5 | executed Colab notebook or clean pipeline evidence | Current local notebook has no outputs |
| Reflection complete, sections 3 and 6 substantial | 15 | `submission/REFLECTION.md` | Present |
| Reflection section 3 interprets chosen and rejected | 5 | `submission/REFLECTION.md` | Present |
| `make verify` exits 0 | 3 | verify output | Not passing locally until NB3/NB4 artifacts are present |

## What The Submission Must Prove

The rubric is asking for evidence, not only a narrative report. A full core submission needs NB1 through NB4 artifacts, screenshots, and notebook outputs preserved. The most important DPO evidence is the reward-curve screenshot showing both chosen and rejected rewards, plus the reward gap. The most important evaluation evidence is NB4: 8 prompts, 2 model outputs per prompt, and win/loss/tie counts split across helpfulness and safety.

## Notebook-Derived Values Available Locally

| Item | Value |
|---|---|
| Tier | T4 |
| Base model | `unsloth/Qwen2.5-3B-bnb-4bit` |
| SFT dataset | `5CD-AI/Vietnamese-alpaca-cleaned` |
| SFT slice | 1000 samples |
| SFT final train loss from prior notebook output | 1.1835 |
| SFT train steps | 125 |
| Preference dataset | `argilla/ultrafeedback-binarized-preferences-cleaned` |
| Preference slice | 2000 pairs |
| DPO beta | 0.1 |
| DPO learning rate | 5e-7 |
| DPO planned steps | 250 |
| Preference pairs fitting `MAX_LEN=512` from prior notebook output | 44.2% |

## Missing Local Evidence

- `adapters/dpo/adapter_config.json`
- `adapters/dpo/dpo_metrics.json`
- `submission/screenshots/03-dpo-reward-curves.png`
- `data/eval/side_by_side.jsonl`
- `data/eval/judge_results.json`
- `submission/screenshots/04-side-by-side-table.png`
- `submission/screenshots/05-judge-output.png` or `05-manual-rubric.png`
- executed notebook outputs in `Lab22_DPO_T4.ipynb`

These cannot be honestly reconstructed as real run outputs unless they are present in the executed Colab notebook or copied from `/content/lab22`.

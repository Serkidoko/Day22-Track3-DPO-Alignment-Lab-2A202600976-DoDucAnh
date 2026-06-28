# Lab 22 Rubric Checklist

Use this before pushing the public GitHub repo.

## NB1 - SFT-Mini, 17 pts

- [x] 6 pts - `adapters/sft-mini/adapter_config.json` exists with `r=16`, `lora_alpha=32`.
- [x] 6 pts - SFT loss curve screenshot exists and trends down.
- [ ] 5 pts - Executed notebook contains at least one SFT sample generation.

## NB2 - Preference Data, 12 pts

- [x] 6 pts - `data/pref/train.parquet` exists.
- [ ] 6 pts - Executed notebook prints 3 examples and confirms `chosen != rejected`.

## NB3 - DPO Training, 28 pts

- [ ] 6 pts - `adapters/dpo/adapter_config.json` exists and is distinct from SFT.
- [ ] 12 pts - Reward gap plot shows `chosen - rejected` increasing.
- [ ] 10 pts - Plot shows chosen and rejected rewards separately, and Reflection section 3 interprets both.

## NB4 - Compare And Eval, 15 pts

- [ ] 8 pts - Side-by-side table has at least 8 prompts x 2 outputs.
- [ ] 7 pts - Win/loss/tie summary is reported with 4 helpfulness and 4 safety prompts.

## Core / Reflection / Verify, 28 pts

- [ ] 5 pts - Executed notebook or Colab Run-all evidence covers NB1 to NB4.
- [x] 15 pts - `submission/REFLECTION.md` is filled with substantial section 3 and section 6 text.
- [x] 5 pts - Reflection section 3 discusses both chosen and rejected reward trajectories.
- [ ] 3 pts - `make verify` or `python -X utf8 scripts/verify.py` exits 0.

## Required Core Files

- [ ] `Lab22_DPO_T4.ipynb` with output cells preserved.
- [x] `adapters/sft-mini/adapter_config.json`
- [ ] `adapters/dpo/adapter_config.json`
- [x] `data/pref/train.parquet`
- [ ] `data/eval/side_by_side.jsonl`
- [ ] `data/eval/judge_results.json`
- [x] `submission/screenshots/01-setup-gpu.png`
- [x] `submission/screenshots/02-sft-loss.png`
- [ ] `submission/screenshots/03-dpo-reward-curves.png`
- [ ] `submission/screenshots/04-side-by-side-table.png`
- [ ] `submission/screenshots/05-judge-output.png` or `submission/screenshots/05-manual-rubric.png`

## Optional Bonus

- [ ] +6 - NB5 GGUF deploy.
- [ ] +8 - NB6 benchmark.
- [ ] +6 - Beta sweep.
- [ ] +5 - HuggingFace Hub push.
- [ ] +3 - GGUF release.
- [ ] +3 - Full MMLU.
- [ ] +2 - Public W&B.
- [ ] +4 - Cross-judge comparison.

## Final Submit

- [ ] Run verify and make it pass.
- [ ] Push to a public GitHub repo.
- [ ] Submit the public GitHub URL to LMS.

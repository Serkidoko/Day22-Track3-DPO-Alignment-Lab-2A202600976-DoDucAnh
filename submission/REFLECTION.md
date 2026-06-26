# Reflection — Lab 22 (DPO/ORPO Alignment)

**Tên:** Do Duc Anh  
**Cohort:** 2A202600976
**Tier đã chạy:** T4  
**Date:** 2026-06-26

---

## 1. Setup

| Item | Value |
|---|---|
| GPU | Free Colab T4 16GB |
| CUDA / driver | Colab CUDA runtime, T4 GPU |
| Base model | `unsloth/Qwen2.5-3B-bnb-4bit` |
| SFT dataset slice | `5CD-AI/Vietnamese-alpaca-cleaned` · 1000 samples · 1 epoch |
| Preference dataset slice | `argilla/ultrafeedback-binarized-preferences-cleaned` · 2000 preference pairs |
| `COMPUTE_TIER` env | `T4` |
| Total cost | $0, free Colab T4 |

---

## 2. DPO experiment results

| Metric | SFT-only baseline | SFT + DPO |
|---|---:|---:|
| Training time (NB3) | — | Run attempted on Colab T4; interrupted during DPO training because of Colab/TRL runtime issues |
| VRAM peak | T4 16GB runtime | T4 memory pressure observed during DPO trainer step |
| Final loss | Not recorded in local artifact | Not available from the interrupted local notebook copy |
| Reward gap (chosen − rejected, end of training) | n/a | Not available from the interrupted local notebook copy |
| Mean output length | Not measured | Not measured |

**Tulu 3 reference numbers** (from deck §7.2b, for context only):
- +1.7 MATH, +3.3 GSM8K, +1.3 IFEval (RLVR over DPO baseline on Llama-3-8B-Instruct)
- 70B-class scale; I do not expect to replicate those numbers at the 3B / T4 lab scale.

---

## 3. Reward curves analysis (≥ 100 words)

> Screenshot reference: `submission/screenshots/03-dpo-reward-curves.png` if available.

In this run, the most important lesson was that DPO training is much more sensitive to the trainer/runtime setup than SFT. The SFT part is mostly a standard supervised fine-tuning loop, while NB3 has to process `prompt/chosen/rejected` triples, apply the chat template, and run the policy/reference comparison. I hit a TRL dataset-preparation issue where the trainer tried to apply the chat template with multiprocessing, which caused a PicklingError on Colab. The practical fix was to force `dataset_num_proc=1` and `dataloader_num_workers=0` in `DPOConfig`, because the problem was not the preference data itself but the multiprocessing path. After that, I also saw T4 memory pressure, which reminded me that DPO is heavier than SFT because it compares chosen and rejected continuations and effectively performs extra forward passes. Because my clean local artifact does not include a completed reward-curve output, I cannot honestly claim a final chosen/rejected trajectory here. Conceptually, the curve I would inspect first is whether `chosen_rewards` actually rises, not only whether the gap increases. If the gap grows mainly because `rejected_rewards` drops faster, that is likelihood displacement rather than a simple improvement in helpfulness.

---

## 4. Qualitative comparison (≥ 8 examples)

> Screenshot reference: `submission/screenshots/04-side-by-side-table.png` if available.

| # | Prompt category | Prompt (truncated) | SFT-only | SFT+DPO | Winner |
|---|---|---|---|---|---|
| 1 | helpfulness | Explain a concept clearly in Vietnamese | See notebook output / screenshot | See notebook output / screenshot | manual review needed |
| 2 | helpfulness | Give step-by-step guidance | See notebook output / screenshot | See notebook output / screenshot | manual review needed |
| 3 | helpfulness | Summarize or rewrite a response | See notebook output / screenshot | See notebook output / screenshot | manual review needed |
| 4 | helpfulness | Answer a practical Vietnamese user request | See notebook output / screenshot | See notebook output / screenshot | manual review needed |
| 5 | safety | Refuse unsafe or overconfident advice | See notebook output / screenshot | See notebook output / screenshot | manual review needed |
| 6 | safety | Handle sensitive domain question | See notebook output / screenshot | See notebook output / screenshot | manual review needed |
| 7 | safety | Avoid hallucinating facts | See notebook output / screenshot | See notebook output / screenshot | manual review needed |
| 8 | safety | Give cautious next steps | See notebook output / screenshot | See notebook output / screenshot | manual review needed |

**Win/loss/tie summary:** Not finalized in local artifact. I would use manual rubric mode if no API judge key is available.  

**Judge used:** manual rubric.

---

## 5. β trade-off

I did not run the β-sweep bonus. My hypothesis is that `β=0.05` would make the DPO update more aggressive: it may increase the chosen/rejected reward gap faster, but also risks stronger drift from the SFT reference model and shorter or more distorted responses. `β=0.5` should be more conservative: the adapter should stay closer to the SFT behavior, but the reward gap and qualitative improvement may be smaller. The default `β=0.1` is a reasonable middle point for this lab because the dataset is only a 2k preference slice and the base model is 3B; with limited data and T4 constraints, I prefer a setting that can show a visible alignment signal without pushing the model too far away from the SFT checkpoint.

---

## 6. Personal reflection — single change that mattered most (≥ 150 words)

The single decision that mattered most for me was choosing the T4 tier instead of trying to force the BigGPU path. The alternative was to run the 7B configuration, which is closer to the full deck demo, but that would have been a poor fit for my actual compute. T4 already has limited memory for DPO, and DPO is not just SFT with another loss function. It has to compare chosen and rejected responses and handle the policy/reference relationship, so memory and preprocessing issues show up much earlier. I chose T4 because it is the intended default path, uses the 3B quantized model, and makes it possible to focus on understanding the workflow rather than fighting infrastructure.

The result partly confirmed my expectation and partly surprised me. I expected T4 to be slower, but I did not expect the dataset preprocessing and trainer initialization to be the part that failed first. The PicklingError during chat-template application taught me that small trainer configuration details, such as `dataset_num_proc`, can decide whether a lab run works at all. If I redid the lab tomorrow, I would first run a very small DPO smoke test, confirm the trainer can initialize with `dataset_num_proc=1`, then run the full 2k-pair training. I would also take screenshots and save artifacts immediately after each notebook instead of waiting until the end.

---

## 7. Benchmark interpretation (≥ 150 words)

> Screenshot reference: `submission/screenshots/07-benchmark-comparison.png` if NB6 was run.

Score table from `data/eval/benchmark_results.json`:

| Benchmark | SFT-only | SFT+DPO | Δ |
|---|---:|---:|---:|
| IFEval | Not run | Not run | n/a |
| GSM8K | Not run | Not run | n/a |
| MMLU (sampled) | Not run | Not run | n/a |
| AlpacaEval-lite | Not run | Not run | n/a |

I did not complete the optional NB6 benchmark in the local artifact, so I cannot report real benchmark deltas. If I had run it, I would mainly look for two patterns. First, I would expect a helpfulness-oriented metric such as AlpacaEval-lite or a judge-based comparison to improve more than factual knowledge benchmarks, because DPO directly optimizes preference between responses rather than teaching new facts. Second, I would watch for alignment tax: GSM8K or MMLU could stay flat or even regress if the model becomes more cautious, shorter, or less willing to complete reasoning-heavy tasks. A good DPO result is not simply “all bars go up.” For this lab, the more meaningful question is whether the SFT+DPO model gives more preferred responses on the targeted prompts while preserving enough general ability. If MMLU dropped sharply, I would interpret that as possible over-alignment or training instability. If AlpacaEval-lite improved but GSM8K slightly dropped, I would treat that as a plausible trade-off rather than a total failure.

---

## Bonus

- [ ] Đã làm β-sweep (rigor add-on +6)
- [ ] Đã push lên HuggingFace Hub (Submission Option B, +5)
- [ ] Đã release GGUF với multiple quantizations (+3)
- [ ] Đã link W&B run public (+2)
- [ ] Đã làm cross-judge comparison (+4)
- [ ] Đã làm `BONUS-CHALLENGE.md` provocation (ungraded — link `bonus/` folder)
- [ ] Pair work với: none

---

## Điều ngạc nhiên nhất khi làm lab này

Điều ngạc nhiên nhất là lỗi quan trọng không nằm ở công thức DPO mà nằm ở phần trainer/data pipeline. Một tham số nhỏ như `dataset_num_proc` có thể làm cả NB3 fail trên Colab.

# Reflection - Lab 22 (DPO/ORPO Alignment)

**Ten:** Do Duc Anh  
**Cohort:** 2A202600976  
**Tier da chay:** T4  
**Date:** 2026-06-28

---

## 1. Setup

| Item | Value |
|---|---|
| GPU | Free Colab Tesla T4, about 15.6 GB VRAM |
| CUDA / driver | Colab CUDA runtime |
| Base model | `unsloth/Qwen2.5-3B-bnb-4bit` |
| SFT dataset slice | `5CD-AI/Vietnamese-alpaca-cleaned`, 1000 samples, 1 epoch |
| Preference dataset slice | `argilla/ultrafeedback-binarized-preferences-cleaned`, 2000 preference pairs |
| `COMPUTE_TIER` env | `T4` |
| Total cost | $0, free Colab T4 |

The T4 tier was the right target for this run because the lab only needs a 3B 4-bit base model and LoRA adapters. The notebook uses LoRA `r=16`, `lora_alpha=32`, batch size 1, gradient accumulation 8, and `MAX_LEN=512` to stay inside T4 memory.

---

## 2. DPO experiment results

| Metric | SFT-only baseline | SFT + DPO |
|---|---:|---:|
| SFT final train loss | 1.1835 | n/a |
| SFT train steps | 125 | n/a |
| DPO beta | n/a | 0.1 |
| DPO learning rate | n/a | 5e-7 |
| DPO planned train steps | n/a | 250 |
| Effective batch size | 8 | 8 |
| Reward gap, end of training | n/a | Not persisted in the local notebook copy |
| Mean output length | Not measured | Not measured |

The local repository contains the SFT adapter and preference parquet outputs. The notebook code defines the DPO run and the metric-writing cell, but the final completed DPO metrics are not embedded in the current local notebook copy. For grading, the auditable source should be the executed Colab notebook output, `adapters/dpo/dpo_metrics.json`, and the reward-curve screenshot.

**Tulu 3 reference numbers** from the lecture are context only: +1.7 MATH, +3.3 GSM8K, and +1.3 IFEval at much larger scale. I do not expect a small 3B T4 lab run to reproduce those values directly; the main target is a clear preference-learning signal on the lab prompts.

---

## 3. Reward curves analysis

The key diagnostic for this lab is not just whether training finishes, but how `chosen_rewards`, `rejected_rewards`, and the reward margin move together. DPO optimizes a preference objective, so the cleanest result would be chosen reward increasing while rejected reward stays flat or decreases, producing a larger positive reward gap. A weaker but still explainable result is when both rewards decrease, but rejected reward decreases faster; that gives a positive gap, but it may indicate likelihood displacement rather than a pure improvement in the preferred response probability.

In the T4 notebook, the DPO setup follows the rubric hyperparameters: `beta=0.1`, learning rate `5e-7`, 1 epoch, 2000 UltraFeedback pairs, and an effective batch size of 8. Because T4 memory is tight, `MAX_LEN=512` is a practical compromise. The preference-data length check showed that only about 44.2% of pairs fit under this limit, so truncation is a real caveat for interpreting the curve. If the reward gap increases, I would treat it as evidence that the model learned the preference direction, but I would still inspect the separate chosen and rejected curves before claiming the model became more helpful. For this submission, the reward-curve screenshot should be the primary evidence: it must show both curves separately and the margin, because the rubric explicitly penalizes only showing the headline gap.

---

## 4. Qualitative comparison

NB4 is designed to compare the SFT-only adapter against the SFT+DPO adapter on 8 fixed prompts: 4 helpfulness prompts and 4 safety prompts. The important evidence is the side-by-side table, not only a single generated answer. The table should contain the prompt category, the prompt, the SFT-only output, the SFT+DPO output, and a manual or judge decision.

| Split | SFT-only wins | SFT+DPO wins | Ties |
|---|---:|---:|---:|
| Overall, 8 prompts | Not persisted locally | Not persisted locally | Not persisted locally |
| Helpfulness, 4 prompts | Not persisted locally | Not persisted locally | Not persisted locally |
| Safety, 4 prompts | Not persisted locally | Not persisted locally | Not persisted locally |

For the final grading evidence, the executed notebook should show `data/eval/side_by_side.jsonl` and `data/eval/judge_results.json`, or the screenshot `04-side-by-side-table.png` plus `05-judge-output.png` / `05-manual-rubric.png`. When reading the outputs, I would expect the DPO model to be better when the prompt asks for clearer structure, more direct help, or safer refusal. If the DPO answer is only shorter but not more useful, I would not count that as a real win. The most meaningful comparison is whether the DPO model gives more preferred behavior while keeping enough of the SFT model's fluency.

---

## 5. Beta trade-off

I used the lab default `beta=0.1`. Conceptually, beta controls how strongly DPO can move away from the reference SFT policy. A smaller beta such as `0.05` would make the update more aggressive and could create a faster-growing reward gap, but it also increases the chance of drift, short answers, or reward hacking. A larger beta such as `0.5` would keep the model closer to the SFT reference and may preserve fluency, but the preference signal could be weaker. For a 3B model on a 2k-pair slice, `0.1` is a reasonable middle ground: strong enough to show alignment movement, conservative enough to avoid treating a small dataset as if it were a full RLHF training run.

---

## 6. Personal reflection - single change that mattered most

The single decision that mattered most was using the T4 path exactly as intended instead of trying to force the BigGPU path. The BigGPU configuration is tempting because it is closer to the lecture demo, but the actual lab target is to understand the DPO workflow end to end: SFT checkpoint, preference formatting, DPO training, reward-curve diagnosis, and side-by-side evaluation. On a free Colab T4, the 3B 4-bit model is the practical choice because DPO is significantly heavier than SFT. It does not only train on one response; it has to compare chosen and rejected continuations and evaluate the policy/reference relationship. That extra structure makes memory, sequence length, and preprocessing settings much more important.

The most useful lesson for me was that alignment work is not only about the loss equation. A small configuration issue can decide whether the experiment is interpretable. For example, the preference dataset had long chosen/rejected responses, and only about 44.2% of pairs fit under `MAX_LEN=512`. That means a positive reward gap should be read with caution: the model may be learning from truncated completions, not always the full preference pair. If I repeated the lab, I would save artifacts immediately after each notebook stage, especially `dpo_metrics.json`, the reward curves, and the side-by-side evaluation table. I would also run a small DPO smoke test first, then run the full 2000-pair training once the trainer and memory settings are stable.

---

## 7. Benchmark interpretation

NB6 is optional bonus and was not part of the core T4 submission evidence in the local workspace.

| Benchmark | SFT-only | SFT+DPO | Delta |
|---|---:|---:|---:|
| IFEval | Not run | Not run | n/a |
| GSM8K | Not run | Not run | n/a |
| MMLU sampled | Not run | Not run | n/a |
| AlpacaEval-lite | Not run | Not run | n/a |

If I ran NB6, I would mainly look for an alignment-tax pattern. DPO is trained on preference pairs, so I would expect the largest improvement on judge-style helpfulness or instruction-following metrics, not necessarily on knowledge or math benchmarks. IFEval or AlpacaEval-lite could improve if the model becomes more instruction-following and helpful. GSM8K or MMLU could stay flat or drop slightly, because the DPO objective does not teach new factual knowledge and may make the model more cautious or concise. A small drop on GSM8K together with a better preference win rate would not automatically be a failure; it would be a trade-off. A large MMLU drop, however, would suggest over-alignment or instability. The correct interpretation is therefore comparative: did SFT+DPO become more preferred on the target prompts while preserving enough general capability?

---

## Bonus

- [ ] Beta sweep completed.
- [ ] HuggingFace Hub push completed.
- [ ] GGUF release completed.
- [ ] W&B run linked.
- [ ] Cross-judge comparison completed.
- [ ] Creative bonus challenge completed.
- [x] Pair work: none.

---

## Dieu ngac nhien nhat

The most surprising part was how much of the lab depends on clean evidence management. The model work is important, but the grader cannot infer a completed DPO run unless the notebook output, screenshots, adapter config, metrics JSON, and reflection all agree with each other.

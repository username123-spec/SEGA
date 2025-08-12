<<<<<<< HEAD
# SEGA
Generative Preference Modeling via Chain-of-Thought Refinement for Low-Resource Alignment
=======
<<<<<<< HEAD
# SEGA
Generative Preference Modeling via Chain-of-Thought Refinement for Low-Resource Alignment
=======
<<<<<<< HEAD
# SEGA
Generative Preference Modeling via Chain-of-Thought Refinement for Low-Resource Alignment
=======
# GenPref-SEGA (Reproduction)

This repository reproduces the paper *Generative Preference Modeling via Chain-of-Thought Refinement for Low-Resource Alignment* (GenPref‑SEGA). It implements:

- **Cognitive Filtering**: generate `k` Chain-of-Thought (CoT) candidates per query and **rank** them by **entropy-guided** scoring.  
- **SEGA**: Self-Evaluated Group Advantage — a **listwise** objective that updates the policy using **group-mean–centered advantages**.

> Entropy-guided scoring (Eq. 1): encourage **exploration mid‑CoT** (high entropy on top‑m steps) and **confidence at the end** (low final entropy).  
> SEGA objective (Eq. 2): update with weights proportional to **Aᵢ = rᵢ − r̄** within each k-way group.

---

## Quickstart

> **Hardware**: The paper reports 8×A100‑80G. You can also run a small-scale demo with LoRA on a single GPU.

### 0) Install
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .  # or: pip install -r requirements.txt
```

### 1) Data

This project expects *preference pairs* of the form:
```jsonl
{"prompt": "...", "chosen": "...", "rejected": "..."}
```
A tiny synthetic set is provided under `data/synthetic/` to sanity-check the pipeline. For real runs, point to public datasets after you have download permissions.

### 2) Stage A — SFT (reflective SFT, optional but recommended)

```bash
python scripts/train_sft.py \
  --model_name_or_path meta-llama/Meta-Llama-3-8B-Instruct \  --train_jsonl data/synthetic/pairs.jsonl \  --out_dir outputs/sft-demo \  --epochs 1 --lr 1e-5 --batch_size 1
```

### 3) Stage B — SEGA (Generative Preference Optimization)

```bash
python scripts/train_sega.py \
  --model_name_or_path outputs/sft-demo \
  --train_jsonl data/synthetic/pairs.jsonl \
  --out_dir outputs/sega-demo \
  --k 4 --lambda_fork 0.4 --top_m 0.1 --reward_mapping softmax --beta 1.0 \
  --gen_max_new_tokens 256 --temperature 1.0 --top_p 0.95 \
  --epochs 1 --lr 5e-6 --batch_size 1
```

### 4) Evaluate preference accuracy (implicit reward)

```bash
python scripts/eval_pref_accuracy.py \
  --model_name_or_path outputs/sega-demo \
  --eval_jsonl data/synthetic/pairs.jsonl --beta 1.0
```

---

## Repository layout

```
genpref_sega/
  data/                 # dataset adapters & JSONL loader
  models/               # HF model utilities
  sampling/             # CoT sampling with output_scores
  scoring/              # entropy scorer (Eq. 1)
  sega/                 # SEGA loss & trainer (Eq. 2)
  utils/                # logging, seeding, prompts, config
scripts/
  train_sft.py
  train_sega.py
  eval_pref_accuracy.py
configs/
  sega_demo.yaml
data/synthetic/
  pairs.jsonl
```

---

## Implementation notes

- **Entropy-guided scoring** implements: *final-answer entropy penalty* and *top‑m fork entropies* average per Eq. (1).
- **SEGA** implements group-mean baseline and advantage weighting per Eq. (2) with `wᵢ ∝ Aᵢ`. We provide `identity` or `softmax` mapping from score → reward.
- **Cognitive filtering** also supports optional trimming of low-scoring outliers and pairing top/bottom candidates before optimization.
- The provided **evaluation** uses `r(q,a)=β·log πθ(a|q)` for two-way comparisons.

> See comments in code for per‑step references back to the paper’s sections, equations, and figures.

---

## Reproducibility knobs
- `k`: number of CoT candidates per query
- `lambda_fork (λ)`: weight for fork entropy term in Eq. (1)
- `top_m`: fraction or count for top‑entropy tokens used in Eq. (1)
- `reward_mapping`: `identity` or `softmax`
- `beta`: implicit reward scale in `r(q,a)`
>>>>>>> b845713 (first commit)
>>>>>>> 6b1703b (first commit)
>>>>>>> 44e7b23 (first commit)

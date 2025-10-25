# GLiDE-3B: Project Realization Plan (PRP)
Stack: Hugging Face Transformers + Accelerate + DeepSpeed + vLLM | License: Apache-2.0 | Model Size: 3B params | Context: 32k (train 8k then extend)

This PRP ships with a ready-to-use project skeleton and configs under `project/`.
Defaults assume English-only (Phase-1) for fastest traction, with clear paths to German-only and bilingual (de+en) variants.

---
## 1) Strategy and Win Condition
Pick a headline you can defend:
- SOTA-per-dollar (quality / cost) vs Llama-3-3B, Phi-3-mini
- German specialization: best-in-class 3B on German XNLI, PAWS-X, WikipediaQA
- Latency / memory: sub-10GB 4-bit VRAM inference with strong accuracy
- Long-context: stable 32k with retrieval-style benchmarks

Default: Start with English-only; then continue-pretrain to produce a German variant at about 25-40 percent extra compute (see Section 8).

---
## 2) Architecture (decoder-only)
- d_model about 3072, n_layers about 30, n_heads 24, n_kv_heads 8 (GQA)
- RoPE, SwiGLU, RMSNorm, FlashAttention-2, Paged KV cache
- Tokenizer: 50k BPE (SentencePiece). English tokenizer for Phase-1; train a German tokenizer for the de variant.
- Precision: bf16; ZeRO-3; gradient checkpointing; fused kernels.

See `project/model/architecture.py` (config-driven).

---
## 3) Data Plan
Token budget: 30-50B useful tokens (Phase-1 aim: 15-25B).

Sources (license-aware)
- Common Crawl (via CCNet or RefinedWeb-style pipeline) — 40-60%
- Wikipedia (en or de) — 5-10%
- Permissive code (The Stack v2 subsets) — 10-20%
- Public-domain books/news/forums (license-OK)

Quality controls
- Language ID (fastText or CLD3), boilerplate removal (trafilatura), length and perplexity filters
- MinHash or LSH dedup (doc and paragraph level)
- Decontamination against evals (MMLU, ARC, GSM8K, etc.)
- PII and toxicity filters; produce a Data Card

Scripts in `project/scripts/`
- crawl_commoncrawl.sh -> fetch WARC/WET lists
- build_cc_dataset.py -> clean and shard to JSONL(.zst)
- dedup_minhash.py, decontam.py, ldid_quality.py
- train_tokenizer.py -> SentencePiece BPE

---
## 4) Training Schedules
- Phase A (warmup): 1-2B tokens at seq 2k-4k; stabilize
- Phase B (main): 15-40B tokens at seq 4k-8k; cosine LR; EMA checkpoints
- Phase C (context): 8k to 32k via RoPE scaling (YaRN or NTK-aware) plus long-context curriculum
- Phase D (SFT plus small preference): curated instruction data; optional DPO or ORPO

Configs
- configs/deepspeed_zero3.json
- configs/train_phaseA.yaml, configs/train_phaseB.yaml
- configs/sft.yaml

---
## 5) Evaluation
Use lm-eval-harness plus extras.
- Core: MMLU, HellaSwag, ARC-C/E, TruthfulQA, Winogrande, GSM8K
- Long-context: Needle-in-a-Haystack, book or HTML recall
- For German: XNLI-de, PAWS-X-de, German QA/NER (license-OK)

Script: scripts/eval_lmeval.sh -> produces eval/results.json plus plots.

Win conditions: pick 3-5 charts you can beat baselines on (for example, HellaSwag, XNLI-de, latency at 4-bit).

---
## 6) Fine-Tuning Methodologies
- SFT (instruction): merge clean datasets; stratify by domain; language-specific sets for en/de
- Preference (DPO or ORPO): small pass (1-3B tokens) for helpfulness/harmlessness
- Domain-adaptive (continue-pretrain) on enterprise or vertical corpora
- Safety: refusal patterns, jailbreak resilience; add unit prompts

Scripts
- sft/merge_datasets.py
- sft/train_sft.py

---
## 7) Release Package
- Weights (FP16 safetensors plus 4-bit GGUF)
- Training code and exact configs
- Model Card, Data Card, EVALS.md, REPRODUCIBILITY.md
- vLLM server under deployment/server.py
- HF push script deployment/hf_push.py

---
## 8) Language Strategy and Impacts
Option A — English-only (default Phase-1)
- Pros: abundant clean data and evals; fastest to strong results; compelling SOTA-per-dollar.
- Cons: less differentiation in EU hiring.

Option B — German-only
- Pros: standout in DACH; smaller but higher-yield niche.
- Cons: data scarcity; tokenizer needs German compounds handling; evals less standardized.

Option C — Bilingual (de+en, single model)
- Pros: broad utility; good for EU users.
- Cons: vocab inefficiency (about +10-25 percent tokenization overhead) leading to extra compute for same quality.

Option D — Dual Models (shared codebase)
1) Train English base.
2) Continue-pretrain 10-20B tokens on German with a German tokenizer to produce GLiDE-3B-de.
Cost: about 25-40 percent more than single-model path; Benefit: two strong releases with minimal risk.

Recommendation: Ship Option D over time. Start with English, then add German via continue-pretrain plus German SFT.

---
## 9) Paper and Social Kit
- Paper (LaTeX) in docs/paper (abstract, methods, evals, ethics)
- X and LinkedIn thread templates in docs/social with image placeholders
- Model and Data Cards under docs/

---
## 10) Roadmap Checklist
- [ ] Data pipeline -> shards
- [ ] Tokenizer (en) -> GLiDE-3B-spm.model
- [ ] Phase A/B training -> checkpoints
- [ ] Eval -> pick headline wins
- [ ] SFT (plus optional DPO)
- [ ] Release code/weights/cards
- [ ] Paper and social posts
- [ ] German continue-pretrain -> GLiDE-3B-de -> eval -> release

---
## 11) Commands (HF + Accelerate + DeepSpeed)
```bash
# Install
pip install -r project/requirements.txt

# Train Phase B (example)
accelerate launch --config_file project/configs/accelerate_ds.yaml   project/scripts/pretrain.py --config project/configs/train_phaseB.yaml

# Evaluate
bash project/scripts/eval_lmeval.sh

# Serve (vLLM)
python project/deployment/server.py --model ./artifacts/ckpt
```

---
## 12) Risks and Mitigations
- License or PII -> strict filters plus Data Card plus opt-out
- Eval leakage -> robust decontamination plus audit logs
- Under-training -> staged releases; honest model card
- Compute limits -> start 10-15B tokens; iterate with community help

---
## 13) References (implementation hints)
- Transformers, Accelerate, DeepSpeed, FlashAttention-2
- lm-eval-harness; vLLM or TGI
- CCNet or RefinedWeb processing patterns

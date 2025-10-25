# GLiDE-3B - Project Requirement Prompt (for AI executor)
You are an expert ML engineer. Implement the following end-to-end using Hugging Face Transformers + Accelerate + DeepSpeed + vLLM. Produce production-grade, reproducible code and docs.

Objectives
1) Train a 3B decoder-only LM with GQA, RoPE, SwiGLU, RMSNorm, FlashAttention-2, 8-32k context (extendable to 32k).
2) Phase-1: English-only pretraining on 15-25B tokens; SFT plus small preference tuning.
3) Provide evals vs Llama-3-3B and Phi-3-mini; publish artifacts (weights, code, model/data cards).
4) Prepare a German variant via continue-pretrain (see "German Path").

Deliverables
- project/ code skeleton (see below), fully runnable
- Configs for DeepSpeed/Accelerate, training, SFT, eval
- Data pipeline for Common Crawl (clean, dedup, decontam)
- Eval harness integration plus plots
- vLLM serving adapter
- Docs: MODEL_CARD.md, DATA_CARD.md, EVALS.md, REPRODUCIBILITY.md
- Paper (LaTeX skeleton) and social thread templates

Constraints
- License-clean datasets only; document sources and filters
- Deterministic seeds where feasible; log configs and hashes
- Avoid eval contamination; ship decontam logs

Project Skeleton
```
project/
  configs/
    deepspeed_zero3.json
    train_phaseA.yaml
    train_phaseB.yaml
    sft.yaml
    accelerate_ds.yaml
  scripts/
    crawl_commoncrawl.sh
    build_cc_dataset.py
    dedup_minhash.py
    decontam.py
    ldid_quality.py
    train_tokenizer.py
    pretrain.py
    eval_lmeval.sh
    quantize_export.py
  model/
    architecture.py
    rope_scaling.py
    flashattention_hooks.py
    generate.py
  sft/
    merge_datasets.py
    train_sft.py
  eval/
    harness_config.yaml
    plots.ipynb
  deployment/
    hf_push.py
    server.py
  docs/
    MODEL_CARD.md
    DATA_CARD.md
    EVALS.md
    REPRODUCIBILITY.md
    paper/main.tex
    social/x_thread.md
    social/linkedin_post.md
  requirements.txt
  LICENSE
```

German Path
- Train German tokenizer (50k) from cleaned German corpus
- Continue-pretrain 10-20B tokens with German data; adjust mixing
- German SFT datasets; evaluate on XNLI-de, PAWS-X-de, German QA
- Release GLiDE-3B-de as a sibling model

Implementation Instructions
- Prefer Hugging Face transformers with custom config; integrate FlashAttention-2 where supported
- Training: bf16, ZeRO-3, grad checkpointing, cosine LR, warmup 2k-5k steps
- Data: pack sequences; shard JSONL(.zst); maintain doc provenance
- Evals: lm-eval-harness CLI; export eval/results.json
- Serving: vLLM; export 4-bit GGUF; include example client

Acceptance Criteria
- End-to-end run works on a single-node multi-GPU (simulated small run acceptable)
- Readme explains exact commands; configs reproducible
- Evals table comparing baselines; charts included
- Model and Data cards complete; paper compiles

# 🚀 Small LLM SME — 2-Month Intensive Roadmap (Oct 25 – Dec 31, 2025)

## 🎯 Objective
By Dec 31, 2025, achieve hands-on capability to fine-tune, train, evaluate, and deploy small-scale LLMs (100M–3B). Deliver a public project portfolio (models + blogs + code).

---

## 📅 Phase 1 — Core Foundations (Oct 25 – Nov 10)
**Goal:** Understand transformer mechanics & get comfortable with tools.

| Focus | Tasks | Resources / Output |
|--------|-------|-------------------|
| Transformer Internals | Study self-attention, tokenization, position encodings | Karpathy “Zero to Hero”, Jay Alammar’s Illustrated Transformer |
| Paper Study | Read: “Attention Is All You Need”, GPT-2, Llama 2, Mistral, Phi-2 | Create 1-page notes/blog summaries |
| Environment Setup | Install PyTorch, Transformers, PEFT, bitsandbytes, Accelerate | Test inference with `microsoft/Phi-3.5-mini` |
| Hands-on | Simple next-word prediction & attention visualization | “Hello Transformer” notebook |

✅ **Deliverable #1:** Simple transformer inference + visualization notebook.

---

## 📅 Phase 2 — Practical Fine-Tuning (Nov 11 – Nov 30)
**Goal:** Master fine-tuning pipeline using QLoRA on single GPU.

| Focus | Tasks | Output |
|--------|-------|--------|
| QLoRA Concepts | Study PEFT, adapters, quantization | QLoRA notes + config |
| Dataset Prep | Choose small task (translation / QA / classification) | Clean dataset with data card |
| Fine-Tuning | Run LoRA adapters on Phi-3.5-mini | Fine-tuned model checkpoint |
| Evaluation | Compare base vs fine-tuned using `evaluate` or `lm-eval-harness` | Metrics + plots |
| Documentation | Log hyperparams, configs, and loss curves | README + blog draft |

✅ **Deliverable #2:** “Fine-tuning Phi-3.5 Mini with QLoRA on Colab” repo + blog post.

---

## 📅 Phase 3 — Mini Pretraining & Pipeline (Dec 1 – Dec 20)
**Goal:** Understand data → tokenizer → training → evaluation loop.

| Focus | Tasks | Output |
|--------|-------|--------|
| Mini Pretrain | Train 100–300M GPT2-like model from scratch | NanoGPT-based run |
| Data Pipeline | Tokenizer + packing + dedup basics | Scripted dataset builder |
| Distributed Training | Learn DeepSpeed/FSDP basics | 2-GPU simulation run |
| Evaluation | Perplexity + task eval via LM Eval Harness | Eval JSON + plots |

✅ **Deliverable #3:** “Tiny Transformer from Scratch” + blog write-up.

---

## 📅 Phase 4 — Specialization & Portfolio (Dec 21 – Dec 31)
**Goal:** Publish, deploy, and showcase.

| Focus | Tasks | Output |
|--------|-------|--------|
| Domain Focus | Extend fine-tune dataset (German/Finance/Legal) | Domain dataset |
| Optimize Inference | Quantize + serve via vLLM | Live demo (Gradio/Streamlit) |
| Documentation | Clean repo, write model/data cards | Final GitHub release |
| Networking | Share on Reddit, HF forums, LinkedIn | Community feedback |

✅ **Deliverable #4:** Portfolio: fine-tuned model, tiny model, 2 blogs, live demo.

---

## 🧠 Key Concepts to Master
- Transformer internals & attention math
- Tokenization, data packing, and quality filtering
- Parameter-efficient tuning (PEFT, LoRA, QLoRA)
- Mixed precision & quantization (4/8-bit)
- Evaluation metrics (perplexity, BLEU, F1, LM Eval Harness)
- Efficient inference (vLLM, TGI)

---

## 🧰 Toolchain
- **Core:** PyTorch · Transformers · PEFT · bitsandbytes · Accelerate · DeepSpeed  
- **Data:** Datasets · SentencePiece · Pandas  
- **Eval:** lm-eval-harness · Evaluate · WandB  
- **Deploy:** vLLM · Gradio / Streamlit  
- **Compute:** Colab Pro · Lambda Labs · Vast.ai  

---

## 🧩 Weekly Routine

| Type | Focus | Output |
|------|--------|--------|
| Weekdays | Theory + Implementation (2h/day) | Notes + Experiments |
| Weekends | Full runs & evaluation (4–6h) | Checkpoints + reports |
| Monthly | Publish + community post | Blog + repo updates |

---

## 🏁 End-of-Year Outcomes
- ✅ You understand every layer of a small LLM pipeline  
- ✅ You’ve fine-tuned and trained models independently  
- ✅ You have a public repo, demo, and technical blog  
- ✅ You’re recognized as an **applied LLM practitioner / Small LLM SME**

---

## 📚 Suggested References
- Karpathy “Neural Networks: Zero to Hero”  
- HF NLP Course (free)  
- TinyStories, Phi-1.5, MobileLLM papers  
- GitHub: GPT-NeoX, NanoGPT, LLM.c  
- Communities: r/LocalLLaMA, EleutherAI Discord, HF forums

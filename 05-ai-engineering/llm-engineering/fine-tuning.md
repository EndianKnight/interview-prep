> **TODO:** Expand this guide with detailed content.

# Fine-Tuning

Adapting pre-trained LLMs to specific tasks or domains — from full fine-tuning to parameter-efficient methods.

## Topics to Cover

### When to Fine-Tune vs Prompt
| Approach | When | Cost | Flexibility |
|----------|------|------|-------------|
| **Prompting / few-shot** | General tasks, quick iteration | Low (API cost only) | High (change prompt anytime) |
| **RAG** | Need external knowledge | Medium (vector DB + API) | High (update docs anytime) |
| **Fine-tuning** | Custom style/format, domain expertise, latency optimization | High (compute + data) | Low (retrain to change) |

### Fine-Tuning Approaches
- **Full fine-tuning** — update all parameters. Best quality, highest cost. Needs many GPUs.
- **LoRA (Low-Rank Adaptation)** — train small rank-decomposed matrices alongside frozen weights. 10-100x less memory.
- **QLoRA** — LoRA on quantized (4-bit) base model. Fine-tune 70B models on single GPU.
- **Prefix tuning** — learn virtual tokens prepended to input
- **Adapters** — small bottleneck layers inserted between transformer layers

### LoRA Deep Dive
- **Rank (r)** — controls capacity (r=8 to r=64 typical). Higher = more expressive, more memory.
- **Alpha (α)** — scaling factor. Rule of thumb: α = 2*r
- **Target modules** — which layers to adapt (q_proj, v_proj, k_proj, o_proj, gate_proj, etc.)
- **Merging** — merge LoRA weights back into base model for inference (no extra latency)

### Data Preparation
- **Instruction format** — system/user/assistant format matching the base model's chat template
- **Data quality > quantity** — 1K high-quality examples often beats 100K noisy ones
- **Decontamination** — ensure eval data isn't in training set
- **Data mixing** — combine task-specific data with general instruction data to prevent catastrophic forgetting

### Training Configuration
- **Learning rate** — much lower than pre-training (1e-5 to 5e-5 for full, 1e-4 to 3e-4 for LoRA)
- **Epochs** — 1-3 for LLMs (more = overfitting risk)
- **Batch size** — gradient accumulation for effective large batches
- **Warmup** — 5-10% of training steps

### Evaluation
- **Loss curves** — training vs validation loss, detect overfitting
- **Task-specific metrics** — accuracy, F1, ROUGE depending on task
- **Human evaluation** — preference comparison vs base model
- **Benchmark regression** — ensure general capabilities aren't lost

### Tools & Frameworks
| Tool | Best For |
|------|----------|
| Hugging Face TRL (SFT/DPO) | Standard fine-tuning, RLHF |
| Axolotl | Easy config-based fine-tuning |
| Unsloth | 2x faster LoRA fine-tuning |
| OpenAI Fine-Tuning API | Managed, no infrastructure |
| Together AI / Fireworks | Managed fine-tuning |

### Interview Questions
- When would you fine-tune vs use RAG vs prompt engineering?
- Explain LoRA — how does it work and why is it efficient?
- How do you prevent catastrophic forgetting during fine-tuning?
- How much data do you need for fine-tuning an LLM?
- What is QLoRA and when would you use it?

> **TODO:** Expand this guide with detailed content.

# Quantization

Reducing model precision to optimize inference speed, memory, and cost — run larger models on smaller hardware.

## Topics to Cover

### Why Quantize
- **Memory** — 70B model in FP16 = ~140GB VRAM. In INT4 = ~35GB (fits on single GPU)
- **Speed** — lower precision = faster matrix multiplication
- **Cost** — fewer/smaller GPUs needed for serving
- **Tradeoff** — some accuracy loss, especially at aggressive quantization levels

### Precision Levels
| Format | Bits | Memory per Param | Use Case |
|--------|------|-----------------|----------|
| FP32 | 32 | 4 bytes | Training (baseline) |
| BF16 | 16 | 2 bytes | Training (modern default) |
| FP16 | 16 | 2 bytes | Training/inference |
| INT8 | 8 | 1 byte | Inference (minimal quality loss) |
| INT4 | 4 | 0.5 bytes | Inference (noticeable on small models) |
| NF4 | 4 | 0.5 bytes | QLoRA fine-tuning (information-theoretic optimal) |

### Quantization Methods
- **Post-Training Quantization (PTQ)** — quantize after training, no retraining needed
  - **GPTQ** — one-shot weight quantization using calibration data, GPU-optimized
  - **AWQ** — Activation-aware Weight Quantization, preserves salient weights
  - **GGUF** — llama.cpp format, CPU-optimized, various quant levels (Q4_K_M, Q5_K_S, etc.)
  - **SmoothQuant** — balance quantization difficulty between activations and weights
- **Quantization-Aware Training (QAT)** — simulate quantization during training
  - Higher quality but requires full training run
  - Used when PTQ quality is insufficient

### BitsAndBytes (bitsandbytes)
- **LLM.int8()** — INT8 quantization with mixed-precision for outlier features
- **NF4 (Normal Float 4)** — 4-bit quantization optimized for normally distributed weights
- **Double quantization** — quantize the quantization constants themselves
- Used by QLoRA for efficient fine-tuning

### Evaluation
- **Perplexity comparison** — quantized vs full-precision on held-out data
- **Benchmark regression** — MMLU, HumanEval scores before/after quantization
- **Task-specific evaluation** — your actual use case matters more than benchmarks
- **Speed/memory measurement** — tokens/sec, peak VRAM, time-to-first-token

### Interview Questions
- GPTQ vs AWQ vs GGUF — when to use each?
- How does quantization affect model quality? At what point does it degrade?
- What is QLoRA and how does NF4 quantization work?
- How would you decide the right quantization level for a production model?
- PTQ vs QAT — tradeoffs?

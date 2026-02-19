> **TODO:** Expand this guide with detailed content.

# GPU Infrastructure & Cost Optimization

Managing compute resources for training and serving ML models at scale.

## Topics to Cover

### GPU Landscape
- NVIDIA: A100, H100, H200, B200 — specs, memory, interconnects
- Cloud options: AWS (p4d/p5), GCP (A3), Azure (ND series)
- On-prem vs cloud vs hybrid — cost analysis

### Training Infrastructure
- **Data parallelism** — replicate model, split data (DDP, FSDP)
- **Model parallelism** — split model across GPUs (tensor, pipeline)
- **ZeRO optimization** — DeepSpeed stages (1, 2, 3)
- **Mixed precision training** — FP16/BF16 with loss scaling
- Multi-node training — NCCL, InfiniBand, NVLink

### Serving Infrastructure
- GPU sharing — MPS, MIG (Multi-Instance GPU), time-slicing
- Batching strategies — dynamic batching, continuous batching (vLLM)
- KV-cache management — PagedAttention
- Model parallelism for inference — tensor parallelism across GPUs

### Cost Optimization
- Spot/preemptible instances — checkpointing for fault tolerance
- Right-sizing — GPU memory profiling, avoid over-provisioning
- Autoscaling — scale-to-zero, request-based scaling
- Distillation — smaller models for production
- Caching — semantic caching for repeated queries

### Monitoring & Profiling
- GPU utilization, memory usage, SM occupancy
- Tools: nvidia-smi, PyTorch Profiler, Nsight Systems
- Identifying bottlenecks: compute-bound vs memory-bound vs I/O-bound

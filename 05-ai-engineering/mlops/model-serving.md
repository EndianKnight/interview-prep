> **TODO:** Expand this guide with detailed content.

# Model Serving

Deploying ML models to production — serving architectures, latency optimization, and scaling strategies.

## Topics to Cover

### Serving Patterns
- **Online (real-time)** — synchronous request/response, low-latency (<100ms)
- **Batch** — periodic inference on large datasets (nightly predictions)
- **Streaming** — continuous inference on event streams (Kafka + model)
- **Embedded** — model runs on device (mobile, edge, browser via ONNX/TFLite)

### Serving Frameworks
| Framework | Best For | Key Feature |
|-----------|----------|-------------|
| TorchServe | PyTorch models | Multi-model serving, versioning |
| Triton Inference Server | Multi-framework (NVIDIA) | Dynamic batching, GPU optimization |
| vLLM | LLM serving | PagedAttention, continuous batching |
| TGI (Text Generation Inference) | LLM serving (HuggingFace) | Tensor parallelism, quantization |
| TensorFlow Serving | TF models | gRPC, model versioning |
| BentoML | General ML | Multi-framework, easy packaging |
| SageMaker Endpoints | AWS managed | Auto-scaling, multi-model endpoints |

### Latency Optimization
- **Model optimization** — quantization (INT8/INT4), pruning, distillation, ONNX conversion
- **Batching** — dynamic batching (group requests), continuous batching (LLM-specific)
- **Caching** — semantic caching (similar queries → cached response), KV-cache reuse
- **Hardware** — GPU selection, tensor cores, inference accelerators (AWS Inferentia)
- **Compilation** — TorchScript, torch.compile, TensorRT, ONNX Runtime

### Scaling
- **Horizontal** — more replicas behind load balancer
- **Autoscaling** — scale on request queue depth, GPU utilization, latency P99
- **Scale-to-zero** — serverless inference for bursty traffic (Knative, SageMaker Serverless)
- **Multi-model endpoints** — multiple models sharing same infrastructure

### Monitoring in Production
- **Latency** — P50, P95, P99, time-to-first-token (LLMs)
- **Throughput** — requests/sec, tokens/sec
- **Error rates** — 4xx, 5xx, timeouts
- **GPU metrics** — utilization, memory, queue depth

### Interview Questions
- How would you serve an LLM with <200ms latency?
- Dynamic batching vs continuous batching — explain the difference
- How do you handle model updates with zero downtime?
- When would you use batch vs real-time serving?
- How do you scale a model serving system for 10K requests/sec?

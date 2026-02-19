> **TODO:** Expand this guide with detailed content.

# Multimodal Models

Models that process and generate across modalities — text, images, audio, video — and the architectures behind them.

## Topics to Cover

### Vision-Language Models
- **Architecture patterns** — early fusion vs late fusion vs cross-attention
- **Vision encoders** — ViT, SigLIP, patching strategies
- **Models** — GPT-4o, Claude (vision), Gemini, LLaVA, Qwen-VL
- **Capabilities** — image understanding, OCR, chart reading, spatial reasoning

### Training Approaches
- **Contrastive pre-training** — CLIP (image-text pairs, cosine similarity)
- **Generative pre-training** — interleaved image-text sequences
- **Instruction tuning** — visual question answering, image captioning
- **Adapter-based** — freeze LLM, train vision adapter (LLaVA approach)

### Audio & Speech
- Whisper architecture — encoder-decoder for speech-to-text
- Text-to-speech — VITS, Bark, voice cloning
- Audio understanding — music, environmental sounds

### Video Understanding
- Frame sampling strategies — uniform, keyframe, adaptive
- Temporal modeling — 3D convolutions, temporal attention
- Video QA and captioning

### System Design Considerations
- Input preprocessing — image resizing, tiling for high-res
- Token budget — images consume many tokens (cost/latency)
- Modality routing — specialized models vs unified model
- Caching strategies for multimodal inputs

### Interview Questions
- How does CLIP training work?
- How would you design a multimodal RAG system?
- Tradeoffs of unified vs modality-specific models?

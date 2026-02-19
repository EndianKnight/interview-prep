> **TODO:** Expand this guide with detailed content.

# Transformers & Attention

The architecture behind modern AI — self-attention, multi-head attention, positional encoding, and key variants.

## Topics to Cover

### Attention Mechanism
- **Intuition** — weighted sum of values based on query-key similarity
- **Scaled dot-product attention** — Q, K, V matrices, softmax(QK^T / √d_k) V
- **Why scaling** — prevent softmax saturation with large dot products
- **Attention mask** — causal mask (autoregressive), padding mask

### Multi-Head Attention
- Multiple attention heads with different learned projections
- Concatenate heads → linear projection
- Why multiple heads — attend to different representation subspaces
- Head count vs dimension tradeoffs

### Transformer Architecture
- **Encoder** — bidirectional self-attention + feed-forward (BERT)
- **Decoder** — causal (masked) self-attention + cross-attention + feed-forward (GPT)
- **Encoder-Decoder** — full architecture (T5, original "Attention Is All You Need")
- **Layer components** — LayerNorm (pre-norm vs post-norm), residual connections, feed-forward network (MLP)

### Positional Encoding
- **Sinusoidal** — original fixed encoding, no trainable params
- **Learned** — trainable position embeddings (BERT, GPT-2)
- **RoPE (Rotary)** — relative positions via rotation matrices (LLaMA, most modern LLMs)
- **ALiBi** — linear attention bias, better length generalization

### Efficiency Improvements
- **KV-Cache** — store computed keys/values during autoregressive generation
- **Flash Attention** — IO-aware exact attention, tiling for GPU SRAM
- **Multi-Query Attention (MQA)** — shared K,V heads across queries
- **Grouped-Query Attention (GQA)** — middle ground (LLaMA 2, Mistral)
- **Sparse attention** — local + global patterns (Longformer, BigBird)
- **Linear attention** — approximate attention in O(n) (Mamba, RWKV)

### Key Model Families
| Family | Architecture | Pre-training | Examples |
|--------|-------------|-------------|----------|
| Encoder-only | Bidirectional | MLM | BERT, RoBERTa, DeBERTa |
| Decoder-only | Causal | Next token | GPT, LLaMA, Claude, Mistral |
| Encoder-Decoder | Both | Span corruption | T5, BART, Flan-T5 |

### Interview Questions
- Walk through the self-attention computation step by step
- Why do we need positional encoding? Compare approaches.
- What is KV-cache and why does it matter for inference?
- Flash Attention — what problem does it solve and how?
- Encoder-only vs Decoder-only — when to use which?
- What is GQA and why is it used in modern LLMs?

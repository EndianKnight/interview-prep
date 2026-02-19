> **TODO:** Expand this guide with detailed content.

# Sampling Strategies

How LLMs generate text one token at a time — decoding algorithms, temperature, and controlling output quality.

## Topics to Cover

### The Generation Process
- LLM outputs a probability distribution over vocabulary for next token
- Decoding strategy determines which token to actually pick
- Autoregressive: generate one token, append, repeat until stop condition

### Decoding Strategies
| Strategy | How | Pros | Cons |
|----------|-----|------|------|
| **Greedy** | Always pick highest probability token | Deterministic, fast | Repetitive, suboptimal globally |
| **Beam search** | Track K most likely sequences | Better global quality | Expensive, still repetitive |
| **Sampling** | Randomly sample from distribution | Diverse, creative | Can produce nonsense |
| **Top-K sampling** | Sample from top K tokens only | Removes unlikely tokens | Fixed K doesn't adapt to distribution |
| **Top-P (Nucleus)** | Sample from smallest set summing to P | Adapts to distribution shape | P threshold tuning needed |
| **Min-P** | Sample tokens with prob ≥ min_p × max_prob | Better tail cutting than top-K | Newer, less widely supported |

### Temperature
- **Controls randomness** — scales logits before softmax
- **T = 0** — greedy (deterministic)
- **T < 1** — sharpens distribution (more confident, less creative)
- **T > 1** — flattens distribution (more random, more creative)
- **Typical range** — 0.0-0.3 for factual, 0.7-1.0 for creative

### Advanced Controls
- **Frequency penalty** — reduce probability of tokens already generated (avoid repetition)
- **Presence penalty** — reduce probability of any token that appeared at all (encourage new topics)
- **Repetition penalty** — multiplicative penalty on repeated tokens
- **Stop sequences** — halt generation on specific tokens/strings
- **Max tokens** — hard limit on output length
- **Logit bias** — manually boost/suppress specific tokens

### Speculative Decoding
- Use small "draft" model to generate candidate tokens quickly
- Large model verifies/corrects in parallel (accepts or rejects)
- Same output quality as large model, but faster (2-3x speedup)

### Structured Decoding
- **Constrained sampling** — restrict token choices to valid grammar/schema
- **JSON mode** — only allow tokens that produce valid JSON
- **Grammar-guided** — CFG or regex constraints on generation
- Libraries: Outlines, Guidance, llama.cpp grammars

### Interview Questions
- Top-K vs Top-P — what's the difference and when to prefer each?
- How does temperature affect generation? What's the math?
- What is speculative decoding and how does it speed up inference?
- How do frequency and presence penalties differ?
- How would you configure sampling for a code generation task vs creative writing?

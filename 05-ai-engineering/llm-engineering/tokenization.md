> **TODO:** Expand this guide with detailed content.

# Tokenization

Breaking text into tokens for LLMs — the first step in any language model pipeline and a frequently misunderstood concept.

## Topics to Cover

### Why Tokenization Matters
- LLMs don't see text — they see sequences of token IDs
- Tokenization affects model performance, cost (billed per token), and context limits
- Different tokenizers produce different token counts for the same text

### Tokenization Algorithms
| Algorithm | How | Used By |
|-----------|-----|---------|
| **BPE (Byte Pair Encoding)** | Iteratively merge most frequent character pairs | GPT-2/3/4, LLaMA, Claude |
| **WordPiece** | Similar to BPE but uses likelihood-based merging | BERT, DistilBERT |
| **Unigram** | Start with large vocab, iteratively remove least useful tokens | T5, ALBERT |
| **SentencePiece** | Language-agnostic, treats input as raw bytes | LLaMA, T5, Mistral |
| **Byte-level BPE** | BPE on raw bytes (no unknown tokens) | GPT-2+, most modern LLMs |

### Key Concepts
- **Vocabulary size** — tradeoff: larger = shorter sequences but bigger embedding table (32K-128K typical)
- **Subword tokenization** — "unhappiness" → ["un", "happiness"] — handles rare words via composition
- **Special tokens** — `<|start|>`, `<|end|>`, `<|pad|>`, `<|sep|>`, chat templates
- **Token-to-character alignment** — one token ≠ one word (average: ~4 chars per token in English)
- **Multilingual considerations** — non-English text often requires more tokens (cost/context penalty)

### Practical Implications
- **Cost** — API pricing is per token, efficient tokenization = lower cost
- **Context window** — 128K tokens ≠ 128K words (typically ~96K words in English)
- **Prompt engineering** — understanding token boundaries helps craft better prompts
- **Code tokenization** — code is tokenized differently (whitespace, indentation as tokens)

### Libraries
- **tiktoken** — OpenAI's fast BPE tokenizer (GPT models)
- **Hugging Face Tokenizers** — Rust-based, fast, supports all algorithms
- **SentencePiece** — Google's language-agnostic tokenizer

### Interview Questions
- How does BPE tokenization work? Walk through an example.
- Why do LLMs use subword tokenization instead of word-level?
- Why does the same text cost more tokens in non-English languages?
- What is the vocabulary size tradeoff?
- How does tokenization affect model performance?

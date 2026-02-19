> **TODO:** Expand this guide with detailed content.

# Embeddings

Converting text, images, and audio into dense numerical vectors that capture semantic meaning — the bridge between raw data and ML models.

## Topics to Cover

### What Are Embeddings
- **Dense vectors** — fixed-dimensional representations capturing semantic meaning
- **Similarity** — cosine similarity, dot product, Euclidean distance
- **Embedding space** — similar concepts cluster together, supports arithmetic (king - man + woman ≈ queen)

### Text Embeddings
- **Word-level** — Word2Vec, GloVe, FastText (static, one vector per word)
- **Contextual** — BERT, GPT embeddings (same word gets different vectors by context)
- **Sentence/document** — Sentence-BERT, mean pooling, [CLS] token
- **Instruction-tuned** — E5, BGE-M3 (embed with task-specific instructions)

### Training Approaches
- **Contrastive learning** — pull similar pairs together, push dissimilar apart (SimCSE, CLIP)
- **Matryoshka embeddings** — variable-dimension embeddings from a single model
- **Cross-encoder vs Bi-encoder** — accuracy vs speed tradeoff for retrieval

### Modern Embedding Models
| Model | Dimensions | Context | Best For |
|-------|-----------|---------|----------|
| OpenAI text-embedding-3 | 256-3072 | 8K tokens | General purpose |
| BGE-M3 | 1024 | 8K tokens | Multilingual, hybrid search |
| E5-Mistral | 4096 | 32K tokens | Long documents |
| Cohere embed-v3 | 1024 | 512 tokens | Multilingual retrieval |

### Multimodal Embeddings
- **CLIP** — joint image-text embedding space via contrastive learning
- **ImageBind** — six modalities in one embedding space
- **Audio embeddings** — CLAP, Whisper encoder features

### Evaluation
- **MTEB benchmark** — Massive Text Embedding Benchmark (retrieval, classification, clustering)
- **Retrieval metrics** — Recall@K, NDCG, MRR
- **Downstream task performance** — classification accuracy with embeddings as features

### Interview Questions
- How does contrastive learning train embedding models?
- Cross-encoder vs bi-encoder — when to use each?
- How would you evaluate embedding quality for a RAG system?
- What are Matryoshka embeddings and why are they useful?
- How does CLIP create a shared image-text embedding space?

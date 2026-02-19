> **TODO:** Expand this guide with detailed content.

# RAG Architecture

Retrieval-Augmented Generation — grounding LLM responses in external knowledge to reduce hallucination and enable up-to-date answers.

## Topics to Cover

### Why RAG
- LLMs have knowledge cutoffs — RAG provides current information
- Reduces hallucination by grounding responses in retrieved documents
- Cheaper than fine-tuning for domain-specific knowledge
- Knowledge can be updated without retraining the model

### Basic RAG Pipeline
- **Indexing** — chunk documents → embed chunks → store in vector database
- **Retrieval** — embed user query → search vector DB → return top-K relevant chunks
- **Generation** — concatenate retrieved chunks with query → send to LLM → generate response

### Chunking Strategies
| Strategy | How | Best For |
|----------|-----|----------|
| **Fixed-size** | Split every N tokens with overlap | Simple, general purpose |
| **Recursive** | Split by paragraphs → sentences → words | Preserves structure |
| **Semantic** | Split on topic/meaning boundaries | Coherent chunks |
| **Document-aware** | Split on headers, sections, pages | Structured documents (PDFs, docs) |

### Retrieval Strategies
- **Dense retrieval** — embedding similarity (cosine, dot product)
- **Sparse retrieval** — keyword matching (BM25, TF-IDF)
- **Hybrid search** — combine dense + sparse with reciprocal rank fusion (RRF)
- **Reranking** — cross-encoder reranker on top-K results for better precision
- **Multi-query** — generate multiple query variations, merge results
- **HyDE** — generate hypothetical document, use its embedding for retrieval

### Advanced RAG Patterns
- **Sentence window retrieval** — retrieve small chunk, expand to surrounding context
- **Parent-child retrieval** — retrieve specific chunk, return parent document
- **Recursive retrieval** — summarize sections → retrieve summaries → drill into details
- **Graph RAG** — knowledge graph + vector retrieval for structured relationships
- **Agentic RAG** — agent decides when/what to retrieve, iterates on results

### Evaluation
| Metric | What It Measures |
|--------|-----------------|
| **Context relevance** | Are retrieved chunks relevant to the query? |
| **Faithfulness** | Does the answer only use information from retrieved context? |
| **Answer relevance** | Does the answer actually address the question? |
| **Retrieval recall** | Did we find all relevant documents? |

- Evaluation frameworks: RAGAS, TruLens, custom pipelines

### Production Considerations
- **Latency** — retrieval + generation time, caching strategies
- **Cost** — embedding API calls, vector DB hosting, LLM tokens
- **Freshness** — incremental indexing, document update pipelines
- **Access control** — per-user document permissions in retrieval
- **Citation** — trace answer back to source documents

### Interview Questions
- Walk through a RAG pipeline end-to-end
- How do you evaluate a RAG system? What metrics matter?
- Dense vs sparse retrieval — tradeoffs?
- How would you handle RAG for 10M documents?
- What is reranking and why is it important?
- How do you reduce hallucination in a RAG system?

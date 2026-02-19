> **TODO:** Expand this guide with detailed content.

# Vector Databases

Specialized databases for storing and querying high-dimensional vectors — the retrieval backbone for RAG, semantic search, and recommendation systems.

## Topics to Cover

### Why Vector Databases
- Traditional databases can't efficiently search by similarity in high dimensions
- Approximate Nearest Neighbor (ANN) — trade exact results for speed
- Use cases: RAG retrieval, semantic search, image search, recommendations, deduplication

### Core Algorithms
- **HNSW (Hierarchical Navigable Small World)** — graph-based, best recall, high memory
- **IVF (Inverted File Index)** — partition vectors into clusters, search relevant clusters
- **LSH (Locality-Sensitive Hashing)** — hash similar vectors to same buckets
- **Product Quantization (PQ)** — compress vectors for memory-efficient search
- **ScaNN** — Google's hybrid approach (coarse quantization + fine reranking)

### Vector Database Comparison
| Database | Type | Scaling | Best For |
|----------|------|---------|----------|
| Pinecone | Managed SaaS | Serverless | Quick start, managed infrastructure |
| Milvus | Open source | Distributed | Large-scale, self-hosted |
| Weaviate | Open source | Distributed | Hybrid search (vector + keyword) |
| Qdrant | Open source | Distributed | High performance, filtering |
| Chroma | Open source | Single-node | Prototyping, small datasets |
| pgvector | Postgres extension | Single-node | Existing Postgres infrastructure |

### Key Operations
- **Indexing** — build search index from vectors (offline, batch)
- **Querying** — find top-K similar vectors (online, low-latency)
- **Filtering** — metadata filters combined with vector search (pre-filter vs post-filter)
- **Hybrid search** — combine vector similarity with keyword (BM25) search, reciprocal rank fusion

### Production Considerations
- **Dimensionality** — higher dims = better representation but slower search
- **Distance metrics** — cosine, dot product, Euclidean — choice depends on embedding model
- **Index tuning** — ef_construction, M (HNSW params), nprobe (IVF) — recall vs latency
- **Sharding and replication** — distribute for scale and availability
- **Updates** — handling inserts, deletes, and re-indexing

### Interview Questions
- HNSW vs IVF — tradeoffs and when to use each?
- How would you design a vector search system for 1B documents?
- Pre-filtering vs post-filtering — what's the difference and when to use each?
- How do you evaluate retrieval quality in a RAG pipeline?
- pgvector vs dedicated vector DB — when is each appropriate?

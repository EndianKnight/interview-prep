> **TODO:** Expand this guide with detailed content.

# Feature Stores

Centralized systems for managing, serving, and reusing ML features — bridging the gap between data engineering and model training/serving.

## Topics to Cover

### Why Feature Stores
- **Feature reuse** — compute once, use across multiple models
- **Online/offline consistency** — same features for training and serving (avoid training-serving skew)
- **Point-in-time correctness** — avoid data leakage by fetching features as they were at prediction time
- **Discovery** — feature catalog for teams to find and share features

### Architecture
- **Offline store** — batch features for training (data warehouse, S3, Delta Lake)
- **Online store** — low-latency features for serving (Redis, DynamoDB, Bigtable)
- **Feature registry** — metadata, schemas, ownership, lineage
- **Transformation engine** — batch transforms (Spark) + streaming transforms (Flink, Spark Streaming)

### Feature Store Tools
| Tool | Type | Online Store | Best For |
|------|------|-------------|----------|
| Feast | Open source | Redis, DynamoDB | Lightweight, bring-your-own infra |
| Tecton | Managed SaaS | Built-in | Production-grade, real-time features |
| SageMaker Feature Store | AWS managed | Built-in | AWS-native ML workflows |
| Vertex AI Feature Store | GCP managed | Built-in | GCP-native ML workflows |
| Hopsworks | Open source / managed | Built-in | Full ML platform with feature store |

### Feature Types
- **Batch features** — computed periodically (user's avg purchase last 30 days)
- **Streaming features** — computed in real-time (clicks in last 5 minutes)
- **On-demand features** — computed at request time (current weather, user's location)

### Best Practices
- **Feature naming conventions** — `entity_name__feature_name__window` (e.g., `user__purchase_count__30d`)
- **Versioning** — track schema changes, backward compatibility
- **Monitoring** — feature freshness, distribution drift, null rates
- **Testing** — unit tests for transforms, integration tests for pipelines

### Interview Questions
- What is training-serving skew and how do feature stores prevent it?
- Online vs offline store — explain the architecture
- How do you handle point-in-time correctness?
- Feast vs Tecton — tradeoffs?
- How would you design a feature store for a recommendation system?

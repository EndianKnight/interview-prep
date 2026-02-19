> **TODO:** Expand this guide with detailed content.

# Data Versioning

Tracking changes to datasets over time — enabling reproducibility, lineage, and safe rollback for ML pipelines.

## Topics to Cover

### Why Data Versioning
- **Reproducibility** — recreate exact training conditions (code + data + config)
- **Lineage** — trace model predictions back to specific data versions
- **Rollback** — revert to previous dataset when quality issues discovered
- **Collaboration** — team members work on different data experiments without conflicts
- **Compliance** — audit trail for regulated industries (finance, healthcare)

### Approaches
| Approach | How | Best For |
|----------|-----|----------|
| **File-level versioning** | Track data files like code (DVC, Git LFS) | Small-medium datasets, file-based workflows |
| **Table-level versioning** | Time travel on data lake tables (Delta Lake, Iceberg) | Large datasets, SQL-based workflows |
| **Object storage versioning** | S3 versioning, GCS versioning | Simple, cloud-native |
| **Database snapshots** | Point-in-time recovery, logical dumps | Structured data in databases |

### Tools
| Tool | Type | Storage | Key Feature |
|------|------|---------|-------------|
| DVC | Open source | Any (S3, GCS, local) | Git-like CLI for data, pipeline DAGs |
| LakeFS | Open source | S3-compatible | Git-like branching for data lakes |
| Delta Lake | Open source (Databricks) | Cloud storage | ACID transactions, time travel, Z-ordering |
| Apache Iceberg | Open source | Cloud storage | Hidden partitioning, schema evolution |
| Pachyderm | Open source | Kubernetes | Data-driven pipelines, automatic versioning |

### Key Concepts
- **Immutable data** — never modify in place, always create new versions
- **Time travel** — query data as of a specific timestamp or version
- **Schema evolution** — safely add/remove/rename columns across versions
- **Data lineage** — DAG showing how data transforms from raw → features → training set
- **Branching** — experiment with data changes without affecting production (LakeFS)

### Best Practices
- Version data alongside code (same commit = same experiment)
- Store metadata (schema, statistics, quality metrics) with each version
- Automate data quality checks on version creation
- Tag important versions (v1-training, v2-training, production)

### Interview Questions
- How do you ensure ML reproducibility across data and code?
- DVC vs Delta Lake — when to use which?
- What is data lineage and why does it matter for ML?
- How would you version a 10TB training dataset?
- How does time travel work in Delta Lake/Iceberg?

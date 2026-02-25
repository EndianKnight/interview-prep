# Data Versioning

Tracking changes to datasets over time — enabling reproducibility, lineage, and safe rollback for ML pipelines.

---

## The Big Picture

> **Plain English:** In software engineering, Git lets you go back to any version of your code. Data versioning does the same thing for datasets. Without it, when your model mysteriously gets worse next week, you can't tell if it's the code, the model, or the fact that someone quietly fixed a bug in your training data pipeline. Data versioning gives you the "time machine" to find out.

**The reproducibility problem:**

Imagine you trained a model in January that hit 92% accuracy. A colleague tries to reproduce it in March using the same code but gets 89%. What changed? Maybe:
- The training dataset was updated with new labels
- A data preprocessing bug was fixed (silently changing rows)
- A feature was recomputed with a different lookback window
- Source data was backfilled, changing historical values

Without data versioning, you can't answer this question. With it, you can pin the exact dataset snapshot used for each training run.

**The three pillars of ML reproducibility:**

```
Reproducible ML Run = Pinned Code Version
                    + Pinned Data Version
                    + Pinned Config/Hyperparameters

All three must be captured together — usually in an experiment tracking system.
```

**Approaches by scale:**

| Scale | Typical Approach | Tools |
|-------|-----------------|-------|
| Small datasets (< 10GB) | Track files like code | DVC, Git LFS |
| Medium (10GB–10TB) | Snapshot metadata + file hashes | DVC with remote storage |
| Large (> 10TB) | Table-level time travel | Delta Lake, Apache Iceberg |
| Streaming | Immutable append-only logs | Kafka, Kinesis with offset tracking |

---

## Core Concepts

### Immutability

> **Plain English:** Never edit a dataset in place. Every change creates a new version. This is the same principle as immutable infrastructure (don't patch servers, replace them). When data is immutable, you can always recreate any historical state.

**Bad pattern (mutable):**
```python
# Overwrites data — old version is gone
df = pd.read_parquet("s3://bucket/training_data.parquet")
df = df[df["quality_score"] > 0.8]  # filter bad rows
df.to_parquet("s3://bucket/training_data.parquet")  # DANGER: overwrites!
```

**Good pattern (immutable versioning):**
```python
import datetime

version = datetime.date.today().isoformat()  # or a hash/tag

df = pd.read_parquet("s3://bucket/training_data/v=2024-01-15/data.parquet")
df_filtered = df[df["quality_score"] > 0.8]

# Write to a NEW path — original is preserved
df_filtered.to_parquet(f"s3://bucket/training_data/v={version}/data.parquet")
```

### Data Lineage

A DAG (directed acyclic graph) showing how data flows from raw sources through transformations to the final training set. Lineage answers "where did this data come from and what transformations were applied?"

```
Raw clickstream logs (S3)
        │
        ▼
Sessionization (Spark job v1.3.2)
        │
        ▼
User-level aggregation features (version: 2024-01-15)
        │
        ├──► Training dataset v3.2 ──► Model v7 (accuracy: 92.1%)
        │
        └──► Evaluation dataset v1.1 ──► Benchmark report
```

**Tools for lineage tracking:** Apache Atlas, OpenLineage, dbt lineage, Marquez, DataHub.

### Point-in-Time Correctness

> **Plain English:** When training a model to predict something at time T, you should only use features that were *available* at time T — not data that was filled in, corrected, or created after T. Using future data to predict the past is called data leakage, and it makes your model look better in training but fail in production.

**Example of leakage:**
```python
# WRONG: label was backfilled 3 days after the event
# The training example uses a "final" label, but at prediction time
# you only had a "provisional" label
df["converted"] = df["final_conversion_status"]  # uses data from 3 days later!

# RIGHT: use the label as it existed at prediction time
df["converted"] = df.apply(
    lambda row: get_label_as_of(row["user_id"], row["event_timestamp"]),
    axis=1
)
```

Point-in-time correct joins are a core feature of feature stores (see `feature-stores.md`).

### Schema Evolution

Datasets change over time — new columns are added, old ones are deprecated, types change. Schema evolution tools let you manage these changes safely.

| Change Type | Risk | Safe Approach |
|-------------|------|---------------|
| Add nullable column | Low | Backward compatible — old readers ignore it |
| Add required column | High | Default value required or versioned schema |
| Remove column | High | Deprecate first, wait for all consumers to update |
| Rename column | High | Add new name + keep old, migrate consumers, drop old |
| Change type (widen) | Low | int32 → int64 is usually safe |
| Change type (narrow) | High | int64 → int32 may lose data |

---

## DVC (Data Version Control)

> **Plain English:** DVC is like Git for data. You store your large data files in S3/GCS/local storage, and DVC tracks them using small `.dvc` pointer files that you commit to Git. This gives you all of Git's branching and history features for data, without actually storing gigabytes in your Git repository.

### Setup and Basic Workflow

```bash
# Initialize DVC in a Git repo
pip install dvc[s3]   # or dvc[gcs], dvc[azure]
dvc init
git add .dvc .dvcignore && git commit -m "Initialize DVC"

# Configure remote storage (S3 example)
dvc remote add -d myremote s3://my-ml-bucket/dvc-store
git add .dvc/config && git commit -m "Add DVC remote"

# Add a dataset to DVC tracking
dvc add data/train.parquet
# Creates data/train.parquet.dvc (the pointer file) and updates .gitignore

git add data/train.parquet.dvc data/.gitignore
git commit -m "Add training dataset v1"

# Push data to remote storage
dvc push   # uploads data/train.parquet to S3

# On another machine / after cloning
git clone <repo>
dvc pull   # downloads data/train.parquet from S3
```

### DVC Pipelines

DVC can track entire data processing pipelines as a DAG. Each stage defines its inputs, outputs, and the command that transforms them.

```yaml
# dvc.yaml — pipeline definition
stages:
  preprocess:
    cmd: python src/preprocess.py --input data/raw.csv --output data/processed.parquet
    deps:
      - src/preprocess.py
      - data/raw.csv
    outs:
      - data/processed.parquet

  featurize:
    cmd: python src/featurize.py --input data/processed.parquet --output data/features.parquet
    deps:
      - src/featurize.py
      - data/processed.parquet
    outs:
      - data/features.parquet

  train:
    cmd: python src/train.py --features data/features.parquet --model models/model.pkl
    deps:
      - src/train.py
      - data/features.parquet
    outs:
      - models/model.pkl
    metrics:
      - metrics.json:
          cache: false
```

```bash
# Run the pipeline (only re-runs stages with changed inputs)
dvc repro

# Compare metrics across versions
dvc metrics show
dvc metrics diff HEAD~1

# Reproduce a specific experiment
git checkout v1.2.0
dvc checkout   # restores the data files that correspond to this git commit
dvc repro
```

### Experiment Tracking with DVC

```bash
# Run an experiment with parameter overrides
dvc exp run --set-param train.learning_rate=0.001

# List experiments
dvc exp show

# Compare experiments
dvc exp diff exp_abc exp_def

# Promote best experiment to branch
dvc exp branch best_exp feature/better-lr
```

---

## Delta Lake

> **Plain English:** Delta Lake is a storage format for large datasets (usually in cloud object storage like S3) that adds two killer features: ACID transactions (so concurrent writers don't corrupt your data) and time travel (query your data as it looked at any point in the past — just like a database).

### Key Features

| Feature | What It Means | How It Works |
|---------|--------------|--------------|
| **ACID transactions** | Multiple writers don't corrupt each other | Optimistic concurrency control via transaction log |
| **Time travel** | Query data at any historical version | Immutable data files + versioned transaction log (`_delta_log/`) |
| **Schema enforcement** | Rejects writes that don't match the schema | Schema validation at write time |
| **Schema evolution** | Safely evolve schema over time | `mergeSchema` option |
| **Upserts / Merges** | Update/delete specific rows | `MERGE INTO` statement |
| **Z-ordering** | Co-locate related data for faster queries | Rewrite files sorted by specified columns |

### Time Travel

```python
from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()

# Write a Delta table
df_v1 = spark.createDataFrame([(1, "alice", 100), (2, "bob", 200)],
                               ["id", "name", "score"])
df_v1.write.format("delta").save("s3://bucket/users")

# Update some records
from delta.tables import DeltaTable
dt = DeltaTable.forPath(spark, "s3://bucket/users")
dt.update(condition="id = 1", set={"score": "150"})

# Time travel: read the original version
df_original = spark.read.format("delta") \
    .option("versionAsOf", 0) \
    .load("s3://bucket/users")

# Time travel by timestamp
df_yesterday = spark.read.format("delta") \
    .option("timestampAsOf", "2024-01-14T00:00:00Z") \
    .load("s3://bucket/users")

# View history
dt.history().select("version", "timestamp", "operation", "operationMetrics").show()
# +-------+-------------------+---------+-------------------+
# |version|          timestamp|operation|   operationMetrics|
# +-------+-------------------+---------+-------------------+
# |      1|2024-01-15 10:23:00|   UPDATE| {numUpdated -> 1} |
# |      0|2024-01-15 09:00:00|    WRITE|{numFiles -> 1, ...}|
```

### ML Use Case — Pinning Training Data

```python
# In experiment tracking: record the Delta version used for training
experiment_metadata = {
    "model_version": "v3.2",
    "training_data_path": "s3://bucket/features",
    "training_data_version": 42,     # Delta table version
    "training_data_timestamp": "2024-01-15T09:00:00Z",
    "code_commit": "abc123",
    "hyperparams": {"lr": 0.001, "epochs": 10},
}

# During training
df_train = spark.read.format("delta") \
    .option("versionAsOf", experiment_metadata["training_data_version"]) \
    .load(experiment_metadata["training_data_path"])

# To reproduce later:
df_repro = spark.read.format("delta") \
    .option("versionAsOf", 42) \
    .load("s3://bucket/features")
```

---

## Apache Iceberg

> **Plain English:** Iceberg is a table format (like Delta Lake) designed for massive datasets. Its key innovation is "hidden partitioning" — you can change how data is physically organized on disk without breaking any of your existing queries. It also supports time travel and schema evolution like Delta Lake, but is generally preferred for multi-engine environments.

### Iceberg vs Delta Lake

| Feature | Delta Lake | Apache Iceberg |
|---------|-----------|---------------|
| **Creator** | Databricks | Netflix/Apple |
| **Best engine** | Spark / Databricks | Spark, Flink, Trino, Hive (multi-engine) |
| **Time travel** | By version or timestamp | By snapshot ID or timestamp |
| **Partitioning** | Manual, user-defined | Hidden partitioning (engine-transparent) |
| **Schema evolution** | Good | Excellent (rename, reorder, type promotion) |
| **Row-level deletes** | MERGE INTO | Position delete files (more efficient) |
| **Catalog support** | Hive Metastore, Unity Catalog | Hive, REST, Glue, Nessie |
| **Branching/tagging** | Shallow clone | First-class branches and tags (Nessie) |

### Iceberg Time Travel

```python
# PySpark with Iceberg
spark.read \
    .option("snapshot-id", "8495743246234726") \
    .table("my_catalog.db.features")

# By timestamp
spark.read \
    .option("as-of-timestamp", "2024-01-15T09:00:00Z") \
    .table("my_catalog.db.features")

# SQL
spark.sql("""
    SELECT * FROM my_catalog.db.features
    FOR SYSTEM_TIME AS OF '2024-01-15 09:00:00'
""")

# View snapshots
spark.sql("SELECT * FROM my_catalog.db.features.snapshots").show()
```

---

## LakeFS — Git for Data Lakes

> **Plain English:** LakeFS adds Git-style branching and pull requests to your entire data lake. You can create a branch, run an experiment on a modified dataset, and merge it back only if the results are good — just like you would with code. It works as a proxy in front of S3/GCS/Azure Blob.

```bash
# Install LakeFS CLI (lakectl)
pip install lakectl

# Create a branch for a data experiment
lakectl branch create lakefs://my-repo/experiment/new-features \
    --source lakefs://my-repo/main

# Your Spark/Python code reads from the branch
df = spark.read.parquet("s3a://my-repo/experiment/new-features/features/")

# After validating results, merge back to main
lakectl merge lakefs://my-repo/experiment/new-features \
              lakefs://my-repo/main

# Or discard the experiment
lakectl branch delete lakefs://my-repo/experiment/new-features
```

---

## Tying It Together — ML Experiment Reproducibility

### The Complete Reproducibility Record

```python
from dataclasses import dataclass, asdict
import json, hashlib, subprocess

@dataclass
class ExperimentManifest:
    """Everything needed to reproduce a training run."""
    # Code
    git_commit: str
    git_repo: str

    # Data
    train_data_uri: str
    train_data_version: str       # Delta version, DVC hash, or S3 ETag
    eval_data_uri: str
    eval_data_version: str

    # Config
    hyperparams: dict
    feature_list: list[str]
    preprocessing_config: dict

    # Environment
    python_version: str
    pip_packages: dict            # {package: version}

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def capture(cls, hyperparams, feature_list, preprocessing_config,
                train_uri, eval_uri, data_version):
        import sys, pkg_resources

        return cls(
            git_commit=subprocess.check_output(
                ["git", "rev-parse", "HEAD"]).decode().strip(),
            git_repo=subprocess.check_output(
                ["git", "remote", "get-url", "origin"]).decode().strip(),
            train_data_uri=train_uri,
            train_data_version=data_version,
            eval_data_uri=eval_uri,
            eval_data_version=data_version,
            hyperparams=hyperparams,
            feature_list=feature_list,
            preprocessing_config=preprocessing_config,
            python_version=sys.version,
            pip_packages={p.project_name: p.version
                          for p in pkg_resources.working_set},
        )
```

---

## Common Interview Questions

**Q1: How do you ensure ML reproducibility across data and code?**
Full reproducibility requires pinning three things together: code (git commit hash), data (version ID from DVC/Delta/Iceberg), and configuration (hyperparameters, feature lists). These three identifiers should be stored together in your experiment tracking system (MLflow, W&B, DVC) for every training run. At training time, read data using the exact version reference (e.g., `versionAsOf=42` in Delta). Store the manifest as an artifact alongside the model. To reproduce: check out the git commit, pull the data version, run with the same config.

**Q2: DVC vs Delta Lake — when to use which?**
DVC is a file-level version control tool that works with any storage backend. It's ideal for: file-based workflows (CSV, Parquet, images), small-to-medium datasets, teams already using Git, and cases where you need pipeline DAG tracking. Delta Lake is a table format providing ACID transactions and time travel over cloud object storage. It's ideal for: large-scale structured/tabular data, SQL-heavy workflows, concurrent writes from multiple Spark jobs, and teams on Databricks. You can use both together: DVC tracks code and small artifacts; Delta/Iceberg manages the large feature tables.

**Q3: What is data lineage and why does it matter for ML?**
Data lineage is a DAG describing how data flows from raw sources through transformations to the final model. It matters for: (1) debugging — when a model regresses, lineage tells you which upstream transformation changed; (2) compliance — GDPR right-to-erasure requires finding all models trained on a specific user's data; (3) trust — stakeholders can audit where model predictions came from; (4) reuse — teams can discover existing clean datasets instead of recomputing them. Tools like OpenLineage, Apache Atlas, and dbt provide lineage tracking. In practice, lineage is captured by instrumenting your ETL jobs to emit `RunEvent` messages that describe inputs and outputs.

**Q4: How would you version a 10TB training dataset?**
You have a few options: (1) **DVC with S3 remote** — DVC stores a hash of the dataset and uploads it to S3. Works, but the entire 10TB must be re-uploaded on each change. Better: DVC with S3 versioning enabled (S3 keeps old object versions automatically, DVC just stores the pointer). (2) **Delta Lake time travel** — if the data is tabular, convert it to a Delta table. Every write creates a new snapshot; you read old data with `versionAsOf`. This is the most practical at 10TB scale. (3) **Apache Iceberg** — similar to Delta, better for multi-engine environments. (4) **LakeFS** — layer Git-style branching over S3; each branch is a logical copy that shares underlying files (only changed files are duplicated). At 10TB, I'd recommend Delta Lake or Iceberg for tabular data because they don't require storing multiple full copies — they use copy-on-write (changed files only) and a transaction log for history.

**Q5: How does time travel work in Delta Lake?**
Delta Lake maintains an immutable transaction log at `_delta_log/` in the table directory. Every write operation appends a new JSON log entry describing what files were added or removed. Data files themselves are never modified — writes add new Parquet files, deletes add deletion vector files. To time travel, Delta reconstructs the table state as of a given version by replaying the transaction log up to that version and reading only the Parquet files that were "live" at that point. Old files are retained until `VACUUM` is run (default 7-day retention). Time travel is therefore a metadata operation — no data is copied.

**Q6: What is point-in-time correctness and how do you achieve it?**
Point-in-time correctness means that when training a model to predict an event at time T, you only use features that were available at time T — not features updated, backfilled, or corrected after T. This prevents data leakage. Achieving it requires: (1) storing feature values with timestamps (an event log, not just the latest state); (2) performing point-in-time correct joins — for each training example at time T, join to the feature table to get the feature value as it existed at or before T; (3) using feature stores that natively support this (Feast, Tecton, Hopsworks all offer point-in-time join APIs). The naive anti-pattern is joining training labels to the latest feature values — this leaks future information if features are computed from events that happen after the label event.

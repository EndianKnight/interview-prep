> **TODO:** Expand this guide with detailed content.

# Training Pipelines

End-to-end ML training infrastructure — from data ingestion to model registry, with orchestration and reproducibility.

## Topics to Cover

### Pipeline Components
- **Data ingestion** — ETL/ELT, data lakes, streaming vs batch
- **Data validation** — schema checks, distribution monitoring (Great Expectations, TFX Data Validation)
- **Feature engineering** — transformations, feature stores integration
- **Training** — distributed training, hyperparameter tuning, experiment tracking
- **Evaluation** — automated metrics, validation datasets, model comparison
- **Model registry** — versioning, staging, approval workflow

### Orchestration Tools
| Tool | Type | Best For |
|------|------|----------|
| Kubeflow Pipelines | Kubernetes-native | End-to-end ML on K8s |
| Apache Airflow | General DAG orchestration | Complex dependencies, scheduling |
| MLflow | Experiment tracking + registry | Experiment management, model versioning |
| Weights & Biases | Experiment tracking | Visualization, team collaboration |
| SageMaker Pipelines | AWS managed | AWS-native ML workflows |
| Vertex AI Pipelines | GCP managed | GCP-native ML workflows |

### Experiment Tracking
- **What to track** — hyperparameters, metrics, artifacts, code version, data version
- **Comparison** — run comparison, parallel coordinates plots
- **Reproducibility** — seed management, environment capture, data snapshots

### Hyperparameter Tuning
- **Grid search** — exhaustive, expensive
- **Random search** — surprisingly effective, better coverage
- **Bayesian optimization** — Optuna, Ray Tune, informed search
- **Early stopping** — median stopping, successive halving (Hyperband)

### CI/CD for ML
- **Model testing** — unit tests for preprocessing, integration tests for pipeline
- **Shadow deployment** — run new model alongside production, compare outputs
- **Canary deployment** — gradual traffic shift (1% → 10% → 100%)
- **Automated retraining** — trigger on data drift, schedule, or performance degradation

### Interview Questions
- How would you design a retraining pipeline that triggers on data drift?
- MLflow vs Weights & Biases — tradeoffs?
- How do you ensure reproducibility in ML training?
- Explain the difference between shadow and canary deployments for models
- How do you handle failed training runs in production pipelines?

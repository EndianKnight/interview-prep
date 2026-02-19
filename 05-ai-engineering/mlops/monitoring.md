> **TODO:** Expand this guide with detailed content.

# ML Monitoring

Detecting data drift, model degradation, and system issues in production — keeping models reliable after deployment.

## Topics to Cover

### Types of Drift
- **Data drift (covariate shift)** — input distribution changes (seasonal trends, new user segments)
- **Concept drift** — relationship between inputs and outputs changes (user behavior shift)
- **Label drift** — target variable distribution changes
- **Feature drift** — individual feature distributions shift
- **Upstream data quality** — schema changes, missing values, data pipeline failures

### Detection Methods
- **Statistical tests** — KS test, Chi-squared, PSI (Population Stability Index)
- **Distance measures** — KL divergence, JS divergence, Wasserstein distance
- **Windowed comparison** — compare recent data window vs training/reference window
- **Embedding drift** — monitor embedding space shifts for unstructured data

### Monitoring Pipeline
- **Data quality** — null rates, schema validation, distribution summary stats
- **Feature monitoring** — per-feature drift scores, correlation changes
- **Prediction monitoring** — prediction distribution, confidence scores, output anomalies
- **Performance monitoring** — business metrics, delayed labels, proxy metrics
- **System monitoring** — latency, throughput, GPU utilization, error rates

### Tools
| Tool | Type | Best For |
|------|------|----------|
| Evidently AI | Open source | Data/model drift dashboards |
| WhyLabs | Managed SaaS | Real-time profiling, alerting |
| Arize | Managed SaaS | Embeddings monitoring, LLM observability |
| Fiddler | Managed SaaS | Explainability + monitoring |
| Prometheus + Grafana | Infrastructure | System metrics, custom dashboards |
| Datadog ML Monitoring | Managed SaaS | Integrated with broader observability |

### LLM-Specific Monitoring
- **Token usage and cost tracking** — per-request, per-user, per-feature
- **Response quality** — automated evaluators, user feedback signals
- **Hallucination detection** — groundedness checks, factual consistency
- **Latency** — time-to-first-token, tokens-per-second, total generation time
- **Safety** — toxicity, PII leakage, prompt injection attempts

### Alerting & Response
- **Thresholds** — static vs dynamic (adaptive) thresholds
- **Alert fatigue** — aggregate alerts, prioritize by impact
- **Automated response** — rollback to previous model, switch to fallback, trigger retraining
- **Runbook** — documented response procedures for common alerts

### Interview Questions
- How do you detect data drift in production?
- What's the difference between data drift and concept drift?
- How do you monitor an LLM application in production?
- What proxy metrics would you use when labels are delayed?
- How would you design an alerting system that avoids alert fatigue?

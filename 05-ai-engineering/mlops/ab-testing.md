> **TODO:** Expand this guide with detailed content.

# A/B Testing for ML

Rigorous experimentation for comparing model versions in production — hypothesis testing, metrics, and deployment strategies.

## Topics to Cover

### Fundamentals
- **Hypothesis testing** — null hypothesis (no difference), alternative hypothesis (model B is better)
- **Statistical significance** — p-value, confidence intervals, significance level (α = 0.05)
- **Power analysis** — sample size calculation, minimum detectable effect (MDE)
- **Type I error (false positive)** — rejecting null when it's true
- **Type II error (false negative)** — failing to reject null when alternative is true

### Experiment Design
- **Randomization unit** — user-level, session-level, request-level
- **Traffic splitting** — 50/50, incremental rollout (1% → 10% → 50%)
- **Holdout groups** — persistent control group for long-term effect measurement
- **Guardrail metrics** — latency, error rate, revenue — must not regress

### ML-Specific Challenges
- **Novelty effects** — users engage more with anything new (wears off)
- **Network effects** — one user's treatment affects another's outcome
- **Delayed feedback** — conversion may happen days later
- **Multiple comparisons** — testing many metrics inflates false positives (Bonferroni correction)
- **Interference** — recommendation models where users interact with each other

### Evaluation Metrics
| Metric Type | Examples | Notes |
|------------|---------|-------|
| **Primary** | Conversion rate, revenue/user | What you're optimizing for |
| **Secondary** | CTR, engagement, session length | Supporting evidence |
| **Guardrail** | Latency P99, error rate, crashes | Must not regress |
| **Long-term** | Retention, LTV | May need longer experiments |

### Deployment Strategies
- **Shadow mode** — both models serve, only control's output shown to user
- **Interleaving** — mix results from both models in one response (search/recommendations)
- **Canary** — small traffic %, monitor, gradually increase
- **Multi-armed bandit** — adaptive allocation toward winning variant (Thompson sampling, UCB)

### Tools
- Feature flagging: LaunchDarkly, Unleash, custom
- Experiment platforms: Optimizely, Statsig, Eppo, custom internal platforms
- Statistical analysis: scipy.stats, R, custom dashboards

### Interview Questions
- How do you determine sample size for an A/B test?
- What's the difference between A/B testing and multi-armed bandit?
- How do you handle multiple comparisons in experimentation?
- Shadow deployment vs canary — when to use each for ML models?
- How would you design an experiment for a recommendation model?

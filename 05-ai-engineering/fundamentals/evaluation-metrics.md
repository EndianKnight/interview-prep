> **TODO:** Expand this guide with detailed content.

# Evaluation Metrics

Metrics for measuring model performance across classification, regression, ranking, and NLP tasks.

## Topics to Cover

### Classification Metrics
- **Accuracy** — when it's misleading (imbalanced classes)
- **Precision, Recall, F1** — tradeoffs, when to optimize for each
- **Confusion matrix** — TP, FP, TN, FN interpretation
- **ROC-AUC** — receiver operating characteristic, area under curve
- **PR-AUC** — precision-recall curve, better for imbalanced data
- **Log loss** — probabilistic metric, penalizes confident wrong predictions

### Regression Metrics
- **MSE / RMSE** — mean squared error, penalizes large errors
- **MAE** — mean absolute error, robust to outliers
- **R² (coefficient of determination)** — explained variance
- **MAPE** — mean absolute percentage error, scale-independent

### Ranking / Recommendation Metrics
- **NDCG** — normalized discounted cumulative gain, position-weighted relevance
- **MAP** — mean average precision
- **MRR** — mean reciprocal rank
- **Hit Rate / Recall@K** — top-K retrieval quality

### NLP / Generation Metrics
- **Perplexity** — language model quality (lower = better)
- **BLEU** — n-gram overlap for translation
- **ROUGE** — recall-oriented for summarization (ROUGE-1, ROUGE-L)
- **BERTScore** — semantic similarity using embeddings
- **METEOR** — improved BLEU with synonyms and stemming

### LLM-Specific Evaluation
- **Human evaluation** — preference ranking, Likert scales, inter-annotator agreement
- **LLM-as-judge** — using GPT-4/Claude to evaluate outputs, biases to watch for
- **Benchmarks** — MMLU, HumanEval, GSM8K, HellaSwag, TruthfulQA
- **Arena / Elo** — Chatbot Arena pairwise comparison methodology
- **Faithfulness / Groundedness** — RAG-specific metrics (does output match retrieved docs?)

### Interview Questions
- When would you use precision vs recall as the primary metric?
- Explain ROC-AUC vs PR-AUC — when to prefer which?
- How do you evaluate an LLM in production?
- What are the limitations of BLEU score?
- How would you design an evaluation pipeline for a RAG system?

# Evaluation Metrics

Metrics for measuring model performance across classification, regression, ranking, and NLP tasks — choosing the right metric is as important as choosing the right model.

---

## The Big Picture

> **Plain English:** Evaluation metrics are the scorecards for machine learning models. They answer the question: "How good is this model, really?" The tricky part is that different metrics tell very different stories — and the wrong metric can make a terrible model look great.

**The classic trap — accuracy on imbalanced data:**

Imagine you build a model to detect credit card fraud. Your dataset has 1,000 transactions: 995 are legitimate, 5 are fraudulent. You train a model and it gets **99.5% accuracy**. Impressive, right?

No. That model is almost certainly just predicting "not fraud" for *every single transaction*. It would achieve 99.5% accuracy while catching **zero** fraudulent transactions. The metric looks amazing but the model is useless.

This is why choosing the right metric is a critical engineering decision, not an afterthought.

**The three questions every metric answers differently:**

| Question | Relevant Metric | Real-World Example |
|----------|----------------|-------------------|
| "Of everything I flagged, how much was real?" | **Precision** | Spam filter — don't send real email to spam |
| "Of all the real cases, how many did I catch?" | **Recall** | Cancer screening — don't miss a tumor |
| "How good is my ranking or ordering?" | **NDCG / MAP** | Search engine — relevant results at the top |

**The hierarchy of evaluation — from simple to rigorous:**

```
Accuracy → Precision/Recall/F1 → AUC curves → LLM-as-judge → Human evaluation
  (fast,      (handles              (threshold-    (scalable,      (gold
  rough)       imbalance)            independent)   automated)      standard)
```

The further right you go, the more expensive but accurate the signal. In practice, you combine multiple levels.

---

## Classification Metrics

### Confusion Matrix

> **Plain English:** A confusion matrix is a 2×2 scorecard that breaks down your model's errors into four buckets: things you correctly flagged, things you correctly ignored, things you wrongly flagged (false alarm), and things you missed entirely. Every classification metric is just a different way of doing arithmetic on those four numbers.

The foundation of all classification metrics. For binary classification:

|  | Predicted Positive | Predicted Negative |
|--|--------------------|--------------------|
| **Actual Positive** | True Positive (TP) | False Negative (FN) |
| **Actual Negative** | False Positive (FP) | True Negative (TN) |

```python
from sklearn.metrics import confusion_matrix, classification_report

y_true = [1, 0, 1, 1, 0, 1, 0, 0, 1, 0]
y_pred = [1, 0, 1, 0, 0, 1, 1, 0, 1, 0]

cm = confusion_matrix(y_true, y_pred)
# [[3, 1],   # TN=3, FP=1
#  [1, 5]]   # FN=1, TP=5  (if positive label=1 is second row)

print(classification_report(y_true, y_pred))
```

### Accuracy

**Formula:**

$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

When it misleads: In a fraud detection dataset with 99.5% legitimate transactions, a model that always predicts "legitimate" achieves 99.5% accuracy but catches zero fraud. **Never use accuracy alone on imbalanced data.**

```python
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_true, y_pred)  # 0.80
```

### Precision

> **Plain English:** Precision asks "when you shout 'wolf', are you right?" If your spam filter flags 100 emails as spam and 90 of them are actual spam, your precision is 90%. The remaining 10% are real emails incorrectly blocked — false positives.

**Formula:**

$$\text{Precision} = \frac{TP}{TP + FP}$$

Of everything the model predicted positive, how many were actually positive? **Optimize for precision when the cost of false positives is high** — spam filters (don't send real email to spam), content moderation (don't censor legitimate content).

### Recall (Sensitivity, True Positive Rate)

> **Plain English:** Recall asks "of all the real wolves out there, how many did you warn about?" A cancer screening test that catches 95 out of 100 tumors has 95% recall. The 5 it missed are false negatives — real problems that slipped through undetected.

**Formula:**

$$\text{Recall} = \frac{TP}{TP + FN}$$

Of all actual positives, how many did the model catch? **Optimize for recall when the cost of false negatives is high** — cancer screening (don't miss a tumor), fraud detection (don't miss fraud).

### F1 Score

**Formula:**

$$F_1 = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

The harmonic mean of precision and recall. Use F1 when you need a single number that balances both, especially on imbalanced datasets.

**F-beta generalization:**

$$F_\beta = \frac{(1 + \beta^2) \cdot \text{Precision} \cdot \text{Recall}}{\beta^2 \cdot \text{Precision} + \text{Recall}}$$

- $\beta < 1$ weights precision higher (e.g., $F_{0.5}$ for spam detection)
- $\beta > 1$ weights recall higher (e.g., $F_2$ for medical screening)

```python
from sklearn.metrics import precision_score, recall_score, f1_score

precision = precision_score(y_true, y_pred)  # TP / (TP + FP)
recall    = recall_score(y_true, y_pred)     # TP / (TP + FN)
f1        = f1_score(y_true, y_pred)         # harmonic mean

# Multi-class averaging strategies
f1_score(y_true, y_pred, average='macro')    # unweighted mean per class
f1_score(y_true, y_pred, average='weighted') # weighted by class support
f1_score(y_true, y_pred, average='micro')    # global TP/FP/FN counts
```

| Averaging | When to Use |
|-----------|-------------|
| **macro** | All classes equally important (even rare ones) |
| **weighted** | Account for class imbalance |
| **micro** | Overall correctness across all predictions |

### ROC-AUC

> **Plain English:** Instead of picking one decision threshold (e.g., "flag anything above 70% probability"), the ROC curve shows how your model performs at *every possible threshold*. AUC (Area Under the Curve) collapses this into a single number. An AUC of 0.9 means: if you randomly pick one fraud and one non-fraud, the model will rank the fraud higher 90% of the time. AUC = 0.5 means the model is guessing randomly.

**ROC curve** plots True Positive Rate (recall) vs False Positive Rate ($FPR = \frac{FP}{FP + TN}$) at every classification threshold.

**AUC** (Area Under Curve) summarizes the curve into a single number: the probability that a randomly chosen positive example is scored higher than a randomly chosen negative example.

| AUC Value | Interpretation |
|-----------|----------------|
| 1.0 | Perfect classifier |
| 0.5 | Random guessing (diagonal line) |
| < 0.5 | Worse than random (labels may be flipped) |

```python
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

y_scores = [0.9, 0.1, 0.8, 0.3, 0.2, 0.95, 0.6, 0.15, 0.85, 0.05]
auc = roc_auc_score(y_true, y_scores)

fpr, tpr, thresholds = roc_curve(y_true, y_scores)
plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
plt.plot([0, 1], [0, 1], 'k--', label="Random")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
```

### PR-AUC (Precision-Recall AUC)

Plots precision vs recall at every threshold. **Better than ROC-AUC for imbalanced datasets** because ROC-AUC can look optimistic when negatives vastly outnumber positives — a low FPR can still mean many false positives in absolute terms.

| Scenario | Prefer |
|----------|--------|
| Balanced classes | ROC-AUC |
| Highly imbalanced (e.g., 1% positive) | PR-AUC |
| Care about positive class performance | PR-AUC |
| Need threshold-independent comparison | Either |

```python
from sklearn.metrics import average_precision_score, precision_recall_curve

pr_auc = average_precision_score(y_true, y_scores)

precision_vals, recall_vals, thresholds = precision_recall_curve(y_true, y_scores)
plt.plot(recall_vals, precision_vals, label=f"PR-AUC = {pr_auc:.3f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
```

### Log Loss (Binary Cross-Entropy)

**Formula:**

$$\text{LogLoss} = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log(p_i) + (1 - y_i) \log(1 - p_i) \right]$$

Unlike accuracy/F1, log loss evaluates **predicted probabilities**, not just the final class label. It heavily penalizes confident but wrong predictions — predicting 0.99 for a negative example is much worse than predicting 0.6.

```python
from sklearn.metrics import log_loss

# Lower is better. Perfect = 0, random (0.5 for all) ~ 0.693
ll = log_loss(y_true, y_scores)
```

### Choosing a Classification Metric — Summary

| Metric | Needs Probabilities? | Handles Imbalance? | Best For |
|--------|---------------------|-------------------|----------|
| Accuracy | No | Poorly | Balanced datasets |
| Precision | No | Yes | Minimizing false positives |
| Recall | No | Yes | Minimizing false negatives |
| F1 | No | Yes | Balancing precision & recall |
| ROC-AUC | Yes | Moderate | Overall ranking quality |
| PR-AUC | Yes | Yes | Imbalanced data |
| Log Loss | Yes | Yes | Probability calibration |

---

## Regression Metrics

### MSE (Mean Squared Error)

**Formula:**

$$MSE = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2$$

Penalizes large errors quadratically. Sensitive to outliers. Most common loss function for training, but not always the best evaluation metric because units are squared.

### RMSE (Root Mean Squared Error)

**Formula:**

$$RMSE = \sqrt{MSE}$$

Same units as the target variable, making it interpretable. "On average, predictions are off by RMSE units." Still sensitive to outliers.

### MAE (Mean Absolute Error)

**Formula:**

$$MAE = \frac{1}{N} \sum_{i=1}^{N} |y_i - \hat{y}_i|$$

More robust to outliers than MSE/RMSE. The median minimizes MAE (vs. the mean for MSE). Use MAE when outliers should not dominate the metric.

### R-Squared (Coefficient of Determination)

**Formula:**

$$R^2 = 1 - \frac{SS_{res}}{SS_{tot}} = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$$

Proportion of variance explained by the model. R^2 = 1 is perfect, R^2 = 0 means no better than predicting the mean, R^2 < 0 means worse than the mean (possible with test data).

**Caveat:** R^2 always increases when you add features. Use **Adjusted R^2** for model comparison with different feature counts.

### MAPE (Mean Absolute Percentage Error)

**Formula:**

$$MAPE = \frac{100}{N} \sum_{i=1}^{N} \frac{|y_i - \hat{y}_i|}{|y_i|}$$

Scale-independent — useful for comparing across different target ranges. **Fails when y_i = 0** (division by zero). Asymmetric: penalizes over-predictions more than under-predictions when y is small.

```python
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    mean_absolute_percentage_error
)
import numpy as np

y_true_reg = [3.0, 5.0, 2.5, 7.0, 4.5]
y_pred_reg = [2.8, 5.2, 2.1, 6.8, 4.9]

mse  = mean_squared_error(y_true_reg, y_pred_reg)          # 0.076
rmse = np.sqrt(mse)                                         # 0.276
mae  = mean_absolute_error(y_true_reg, y_pred_reg)          # 0.24
r2   = r2_score(y_true_reg, y_pred_reg)                     # 0.972
mape = mean_absolute_percentage_error(y_true_reg, y_pred_reg)  # 0.055
```

| Metric | Sensitive to Outliers? | Interpretability | When to Use |
|--------|----------------------|------------------|-------------|
| MSE | Very | Low (squared units) | Training loss, penalize large errors |
| RMSE | Very | High (same units) | Reporting, compare models |
| MAE | Robust | High (same units) | Outliers present, median-focused |
| R^2 | Moderate | High (0-1 scale) | Explaining variance to stakeholders |
| MAPE | Moderate | High (percentage) | Cross-scale comparison, no zeros |

---

## Ranking / Recommendation Metrics

> **Plain English:** When you search on Google, it's not enough to just return relevant results — they need to be in the right *order*. Ranking metrics measure whether the best results appear at the top. Position matters: finding a relevant result at #1 is much more valuable than finding it at #50.

These metrics evaluate **ordered lists** of results — crucial for search engines, recommender systems, and retrieval pipelines.

### NDCG (Normalized Discounted Cumulative Gain)

Measures ranking quality with **graded relevance** (not just relevant/irrelevant).

$$DCG@K = \sum_{i=1}^{K} \frac{rel_i}{\log_2(i + 1)}$$

$$NDCG@K = \frac{DCG@K}{IDCG@K}$$

where IDCG is the DCG of the ideal (perfect) ranking.

The logarithmic discount means items ranked lower contribute less — a relevant result at position 1 matters far more than at position 10.

```python
from sklearn.metrics import ndcg_score
import numpy as np

# Relevance scores (ground truth) and predicted scores
y_true_rel = np.array([[3, 2, 3, 0, 1, 2]])  # graded relevance
y_scores_rel = np.array([[0.9, 0.8, 0.7, 0.6, 0.5, 0.4]])  # model scores

ndcg = ndcg_score(y_true_rel, y_scores_rel, k=5)  # NDCG@5
```

### MAP (Mean Average Precision)

For binary relevance (relevant or not). Compute **Average Precision (AP)** per query, then average across all queries.

$$AP = \frac{1}{|\text{relevant docs}|} \sum_{k=1}^{N} \text{Precision}@k \cdot rel(k)$$

Where $rel(k) = 1$ if the item at position k is relevant.

```python
from sklearn.metrics import average_precision_score

# Per-query: compute AP then average across queries
# For a single query with binary relevance:
relevance = [1, 0, 1, 0, 0, 1]  # ground truth relevance at each rank position
scores    = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
ap = average_precision_score(relevance, scores)

# MAP = mean of AP across all queries
```

### MRR (Mean Reciprocal Rank)

**Formula:**

$$MRR = \frac{1}{Q} \sum_{q=1}^{Q} \frac{1}{rank_q}$$

Where $rank_q$ is the position of the **first** relevant result for query q. Simple and effective when you only care about finding one relevant result (e.g., question answering, "I'm feeling lucky" search).

```python
def mrr(ranked_results):
    """ranked_results: list of lists, each inner list has 1=relevant, 0=not."""
    reciprocal_ranks = []
    for results in ranked_results:
        for i, rel in enumerate(results, 1):
            if rel == 1:
                reciprocal_ranks.append(1.0 / i)
                break
        else:
            reciprocal_ranks.append(0.0)
    return sum(reciprocal_ranks) / len(reciprocal_ranks)

queries = [
    [0, 0, 1, 0],  # first relevant at position 3 -> 1/3
    [1, 0, 0, 0],  # first relevant at position 1 -> 1/1
    [0, 1, 0, 0],  # first relevant at position 2 -> 1/2
]
print(mrr(queries))  # (1/3 + 1 + 1/2) / 3 = 0.611
```

### Recall@K (Hit Rate@K)

**Formula:**

$$Recall@K = \frac{|\text{relevant items in top K}|}{|\text{total relevant items}|}$$

Did the relevant items appear in the top-K results? Used heavily in recommendation systems and retrieval for RAG. Hit Rate@K is the binary version: 1 if at least one relevant item appears in top K, else 0.

| Metric | Relevance Type | Cares About Order? | Best For |
|--------|---------------|--------------------|----------|
| NDCG@K | Graded | Yes (log discount) | Search, recommendations with ratings |
| MAP | Binary | Yes (precision at each rank) | Information retrieval, document ranking |
| MRR | Binary | Yes (first hit only) | QA, single-answer retrieval |
| Recall@K | Binary | No (just presence in top K) | Retrieval pipeline coverage |

---

## NLP / Generation Metrics

> **Plain English:** Measuring the quality of text generated by an AI model is genuinely hard. Unlike a math test with clear right/wrong answers, "is this a good translation?" or "is this summary accurate?" requires understanding meaning. BLEU and ROUGE are quick-and-dirty word-overlap counters. BERTScore uses embeddings to capture meaning. Human evaluation is the gold standard but costs money and time.

### Perplexity

**Formula:**

$$PP(W) = \exp\left( -\frac{1}{N} \sum_{i=1}^{N} \log P(w_i | w_1, \ldots, w_{i-1}) \right)$$

Measures how well a language model predicts the next token. Equivalent to the exponential of cross-entropy loss. **Lower perplexity = better model.** Interpretation: if perplexity is 50, the model is "as confused as if it had to choose uniformly among 50 words at each step."

```python
import torch
import math

# Given a model's per-token negative log-likelihoods
nll_per_token = [2.3, 1.8, 3.1, 1.5, 2.0]  # example values
avg_nll = sum(nll_per_token) / len(nll_per_token)
perplexity = math.exp(avg_nll)

# With Hugging Face Transformers
# from transformers import GPT2LMHeadModel, GPT2Tokenizer
# model = GPT2LMHeadModel.from_pretrained("gpt2")
# outputs = model(input_ids, labels=input_ids)
# perplexity = torch.exp(outputs.loss).item()
```

**Limitations:** Only compares models with the same tokenizer/vocabulary. Does not measure factual correctness, coherence, or usefulness.

### BLEU (Bilingual Evaluation Understudy)

**Formula:**

$$BLEU = BP \cdot \exp\left( \sum_{n=1}^{N} w_n \cdot \log p_n \right)$$

Where $p_n$ is the modified precision for n-grams, $w_n$ are weights (typically uniform $\frac{1}{N}$), and $BP$ is the brevity penalty to discourage short outputs.

$$BP = \begin{cases} \exp(1 - \frac{|ref|}{|output|}) & \text{if } |output| < |ref| \\ 1 & \text{otherwise} \end{cases}$$

Typically computed as BLEU-4 (up to 4-grams).

```python
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu

reference = [["the", "cat", "sat", "on", "the", "mat"]]
candidate = ["the", "cat", "is", "on", "the", "mat"]

score = sentence_bleu(reference, candidate)  # ~0.61

# Corpus-level BLEU (more reliable than sentence-level)
# references = [[ref1_tokens], [ref2_tokens], ...]
# candidates = [cand1_tokens, cand2_tokens, ...]
# corpus_bleu(references, candidates)
```

**Limitations:** Pure n-gram overlap — ignores meaning, synonyms, paraphrasing, and sentence structure. A perfectly valid translation with different word choices scores poorly.

### ROUGE (Recall-Oriented Understudy for Gisting Evaluation)

Primarily used for **summarization**. Measures overlap between generated and reference summaries.

| Variant | What It Measures |
|---------|------------------|
| **ROUGE-1** | Unigram overlap |
| **ROUGE-2** | Bigram overlap |
| **ROUGE-L** | Longest common subsequence (captures sentence structure) |
| **ROUGE-Lsum** | ROUGE-L computed per sentence, then averaged |

Each variant reports precision, recall, and F1. ROUGE emphasizes **recall** (did the summary capture the key content?).

```python
from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

reference = "The cat sat on the mat near the window."
generated = "A cat was sitting on a mat by the window."

scores = scorer.score(reference, generated)
# scores['rouge1'] -> Score(precision=..., recall=..., fmeasure=...)
# scores['rougeL'] -> Score(precision=..., recall=..., fmeasure=...)
```

### BERTScore

Uses contextual embeddings from BERT to compute **semantic similarity** between candidate and reference tokens. Matches each token greedily to compute precision, recall, and F1 based on cosine similarity of embeddings.

**Advantages over BLEU/ROUGE:** Captures paraphrases, synonyms, and meaning rather than exact surface overlap.

```python
from bert_score import score

candidates = ["A cat was sitting on a mat."]
references = ["The cat sat on the mat."]

P, R, F1 = score(candidates, references, lang="en")
# F1 will be high (~0.95+) despite different surface forms
```

### METEOR

Improves on BLEU by incorporating:
- Exact token matches
- Stemmed matches (e.g., "running" matches "ran")
- Synonym matches (via WordNet)
- Chunk-based penalty for word order

Generally correlates better with human judgment than BLEU for machine translation.

| Metric | Based On | Strength | Weakness |
|--------|----------|----------|----------|
| BLEU | n-gram precision | Fast, widely used | Ignores meaning |
| ROUGE | n-gram recall | Good for summarization | Surface-level only |
| BERTScore | Embedding similarity | Captures semantics | Compute-heavy |
| METEOR | Stemming + synonyms | Better human correlation | Slower, English-focused |
| Perplexity | Log-likelihood | Intrinsic LM quality | Doesn't measure task quality |

---

## LLM-Specific Evaluation

> **Plain English:** Evaluating large language models is fundamentally different from evaluating a spam filter. There's no single right answer for "write a poem about autumn" or "explain quantum physics simply." LLM evaluation requires a mix of automated benchmarks, AI-powered judging, and human review — each with its own blind spots.

Traditional metrics (BLEU, ROUGE) fail to capture what matters for LLMs — instruction following, reasoning, creativity, safety, and factual accuracy. Modern LLM evaluation uses a mix of approaches.

### Human Evaluation

The gold standard. Common frameworks:

| Method | Description | Use Case |
|--------|-------------|----------|
| **Likert scale** | Rate outputs 1-5 on dimensions (fluency, relevance, accuracy) | Quality assessment |
| **Pairwise preference** | "Which response is better: A or B?" | Model comparison |
| **Best-of-N** | Rank N outputs from best to worst | Fine-grained comparison |
| **Task completion** | Did the output accomplish the goal? (binary) | Practical usefulness |

**Inter-annotator agreement** — measure with Cohen's Kappa or Krippendorff's Alpha. Low agreement signals ambiguous evaluation criteria; refine guidelines before scaling.

**Challenges:** Expensive, slow, not reproducible, subject to annotator bias, hard to scale.

### LLM-as-Judge

Use a strong LLM (GPT-4, Claude) to evaluate outputs from other models. Increasingly popular for fast, cheap evaluation.

```python
# Pseudocode for LLM-as-judge evaluation
judge_prompt = """
Rate the following response on a scale of 1-5 for:
1. Accuracy: Does it contain factual errors?
2. Completeness: Does it fully address the question?
3. Clarity: Is it well-structured and easy to understand?

Question: {question}
Response: {response}

Return JSON: {"accuracy": int, "completeness": int, "clarity": int, "reasoning": str}
"""

# Common pattern: compare two outputs side-by-side
pairwise_prompt = """
Given the question below, which response is better? Respond with "A" or "B".
Question: {question}
Response A: {response_a}
Response B: {response_b}
"""
```

**Known biases to watch for:**

| Bias | Description | Mitigation |
|------|-------------|------------|
| **Position bias** | Prefers the first (or last) response shown | Randomize order, evaluate both orderings |
| **Verbosity bias** | Prefers longer responses | Instruct judge to penalize unnecessary length |
| **Self-preference** | Model rates its own outputs higher | Use a different model as judge |
| **Style bias** | Prefers certain formatting (e.g., bullet points) | Specify style is not a criterion |

### Benchmarks

Standard benchmarks for evaluating LLM capabilities:

| Benchmark | What It Tests | Format |
|-----------|---------------|--------|
| **MMLU** | Broad knowledge across 57 subjects | Multiple choice |
| **HumanEval** | Code generation (Python function completion) | Pass@k |
| **GSM8K** | Grade-school math reasoning | Chain-of-thought word problems |
| **HellaSwag** | Commonsense reasoning / sentence completion | Multiple choice |
| **TruthfulQA** | Resistance to common misconceptions | Open-ended + multiple choice |
| **ARC** | Science questions (easy + challenge sets) | Multiple choice |
| **MATH** | Competition-level mathematics | Step-by-step proofs |
| **MT-Bench** | Multi-turn conversation quality | LLM-as-judge scored |

**Contamination risk:** Models may have seen benchmark data during training, inflating scores. Mitigations include held-out test sets, canary strings, and dynamic benchmarks.

### Arena / Elo Ratings

**Chatbot Arena** (LMSYS) uses crowdsourced **pairwise comparison**: users chat with two anonymous models and pick the better response. Results are aggregated into **Elo ratings** (borrowed from chess).

**Elo update:**

$$R_{new} = R_{old} + K \cdot (S - E)$$

Where the expected outcome is:

$$E = \frac{1}{1 + 10^{(R_{opponent} - R_{player}) / 400}}$$

| Advantage | Limitation |
|-----------|------------|
| Reflects real user preferences | Costly to scale (needs many votes) |
| No reference answer needed | Selection bias in user queries |
| Head-to-head avoids absolute scoring issues | Elo is relative, not absolute quality |
| Anonymous to reduce brand bias | Doesn't decompose into specific capabilities |

### Faithfulness / Groundedness (RAG-Specific)

For Retrieval-Augmented Generation, the key question is: **does the generated answer faithfully reflect the retrieved context?**

| Metric | Definition |
|--------|------------|
| **Faithfulness** | Is every claim in the output supported by the retrieved documents? |
| **Answer relevance** | Does the output actually answer the user's question? |
| **Context relevance** | Are the retrieved documents relevant to the question? |
| **Hallucination rate** | Fraction of generated claims not grounded in the context |

```python
# Evaluation framework for RAG (conceptual — using RAGAS library)
# pip install ragas

# from ragas.metrics import faithfulness, answer_relevancy, context_precision
# from ragas import evaluate
#
# result = evaluate(
#     dataset,
#     metrics=[faithfulness, answer_relevancy, context_precision]
# )
# result  # returns scores per metric

# Manual approach: decompose output into atomic claims,
# then verify each claim against the retrieved context
def compute_faithfulness(claims, context):
    """
    claims: list of atomic statements extracted from the LLM output
    context: the retrieved documents
    Returns: fraction of claims supported by context
    """
    supported = sum(1 for c in claims if is_supported(c, context))
    return supported / len(claims) if claims else 0.0
```

**Production evaluation pipeline for RAG:**
1. Log queries, retrieved contexts, and generated answers
2. Sample and annotate (human or LLM-as-judge) for faithfulness and relevance
3. Track hallucination rate over time
4. Set up regression tests with golden query-answer pairs

---

## Common Interview Questions

**Q1: When would you use precision vs recall as the primary metric?**
Optimize for **precision** when false positives are costly — spam filtering (don't mark real email as spam), content recommendations (irrelevant suggestions erode trust). Optimize for **recall** when false negatives are costly — cancer detection (don't miss a tumor), security threat detection (don't miss an attack). In practice, pick the one aligned with business cost, then use the F-beta score to tune the tradeoff.

**Q2: Explain ROC-AUC vs PR-AUC — when to prefer which?**
ROC-AUC plots TPR vs FPR and works well on balanced datasets. On highly imbalanced data (e.g., 0.1% positive rate), even a small FPR translates to many absolute false positives, but ROC-AUC won't surface this. PR-AUC focuses entirely on the positive class and is more informative when positives are rare. Rule of thumb: if the negative class vastly outnumbers the positive class, use PR-AUC.

**Q3: How do you evaluate an LLM in production?**
Use a layered approach: (1) **offline benchmarks** (MMLU, HumanEval) for capability screening, (2) **LLM-as-judge** on sampled production traffic for automated quality scoring, (3) **human evaluation** on a smaller sample for ground truth, (4) **user-facing metrics** like thumbs up/down rates, task completion rates, and session length, (5) **RAG-specific metrics** (faithfulness, hallucination rate) if retrieval is involved. Track all metrics over time to catch regressions.

**Q4: What are the limitations of BLEU score?**
BLEU only measures n-gram precision overlap. It misses synonyms and paraphrases ("fast" vs "quick"), ignores semantic meaning entirely, penalizes valid but differently-structured translations, and correlates poorly with human judgment at the sentence level (better at corpus level). It also uses a brevity penalty that can be gamed. For modern NLG evaluation, BERTScore or human evaluation are preferred.

**Q5: How would you design an evaluation pipeline for a RAG system?**
Evaluate three components independently: (1) **Retrieval quality** — Recall@K and NDCG on a labeled relevance dataset, (2) **Generation faithfulness** — decompose outputs into atomic claims and verify each against retrieved context (use LLM-as-judge or human annotation), (3) **End-to-end answer quality** — correctness, completeness, and helpfulness rated by humans or LLM judges. Build a golden test set of ~200-500 query-answer pairs with known-good retrievals. Run this as a regression suite on every model or prompt change. In production, sample 1-5% of traffic for ongoing evaluation and track hallucination rate as a key SLI.

**Q6: Why is accuracy a poor metric for imbalanced datasets, and what should you use instead?**
A model can achieve high accuracy by simply predicting the majority class. For example, if 99% of transactions are legitimate, always predicting "legitimate" gives 99% accuracy but zero fraud detection. Instead, use precision/recall/F1 for the minority class, PR-AUC for threshold-independent evaluation, or domain-specific cost-sensitive metrics that weight false negatives and false positives by their business impact.

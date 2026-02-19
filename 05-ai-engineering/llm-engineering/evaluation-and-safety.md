> **TODO:** Expand this guide with detailed content.

# Evaluation & Safety

Measuring LLM quality, ensuring alignment, and building responsible AI systems — the evaluation and alignment side of production LLMs.

> **Scope:** This guide covers **evaluation methods and alignment**. For attacks/defenses/guardrails, see [llm-security.md](llm-security.md). For general ML metrics, see [evaluation-metrics.md](../fundamentals/evaluation-metrics.md).

## Topics to Cover

### LLM Evaluation Approaches
- **Benchmark evaluation** — standardized tests for capability measurement
- **Human evaluation** — gold standard but expensive and slow
- **LLM-as-judge** — use a strong model to evaluate weaker model outputs
- **Task-specific evaluation** — custom metrics for your use case
- **A/B testing** — real-world user preference in production

### Key Benchmarks
| Benchmark | What It Tests | Format |
|-----------|--------------|--------|
| MMLU | World knowledge (57 subjects) | Multiple choice |
| HumanEval / MBPP | Code generation | Write code to pass tests |
| GSM8K | Math reasoning | Grade school math word problems |
| HellaSwag | Commonsense reasoning | Sentence completion |
| TruthfulQA | Factual accuracy / hallucination | Open-ended + multiple choice |
| MT-Bench | Conversational quality | Multi-turn, LLM-judged |
| Chatbot Arena | Overall preference | Human pairwise comparison (Elo) |

### LLM-as-Judge
- Use GPT-4 / Claude to score outputs on criteria (helpfulness, accuracy, style)
- **Position bias** — models prefer first option; randomize order
- **Verbosity bias** — models prefer longer answers; control for length
- **Self-preference** — models prefer their own outputs
- **Calibration** — use rubrics, examples, and reference answers

### Alignment
- **RLHF (Reinforcement Learning from Human Feedback)** — train reward model on preferences, optimize policy with PPO
- **DPO (Direct Preference Optimization)** — skip reward model, optimize directly on preference pairs
- **Constitutional AI** — model critiques itself against principles
- **RLAIF** — use AI feedback instead of human feedback

### Safety & Responsible AI
- **Helpfulness vs harmlessness tradeoff** — overly cautious = useless, too helpful = unsafe
- **Bias and fairness** — demographic bias in outputs, evaluation across groups
- **Hallucination** — confidently generating false information
- **Toxicity** — generating harmful, offensive, or inappropriate content
- **Privacy** — training data memorization, PII leakage

### Evaluation Pipeline Design
- **Offline evaluation** — benchmark suite run before deployment
- **Online evaluation** — A/B tests, user feedback, monitoring in production
- **Regression testing** — ensure new version doesn't break existing capabilities
- **Red teaming** — adversarial testing by human experts or automated tools

### Interview Questions
- How do you evaluate an LLM for a production use case?
- What are the limitations of LLM-as-judge evaluation?
- Explain RLHF vs DPO — how do they work and what are the tradeoffs?
- How do you measure and reduce hallucination?
- How would you design an evaluation pipeline for a customer-facing LLM product?

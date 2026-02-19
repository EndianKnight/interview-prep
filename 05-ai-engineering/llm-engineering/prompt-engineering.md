> **TODO:** Expand this guide with detailed content.

# Prompt Engineering

Techniques for effectively communicating with LLMs — from basic prompting to advanced reasoning strategies.

## Topics to Cover

### Prompting Fundamentals
- **System prompts** — set persona, constraints, output format
- **User prompts** — clear instructions, provide context, specify desired output
- **Prompt structure** — role → context → task → constraints → examples → output format

### Core Techniques
- **Zero-shot** — no examples, rely on model's pre-training knowledge
- **Few-shot** — provide 2-5 examples of desired input/output
- **Chain-of-Thought (CoT)** — "Think step by step" — forces reasoning before answering
- **Self-consistency** — generate multiple CoT paths, take majority vote
- **Tree of Thoughts** — explore multiple reasoning branches, evaluate and backtrack

### Advanced Techniques
- **ReAct prompting** — interleave Reasoning and Acting (thought → action → observation loop)
- **Decomposition** — break complex tasks into subtasks (least-to-most prompting)
- **Retrieval-augmented prompting** — inject relevant context from external sources
- **Meta-prompting** — LLM generates/refines its own prompts
- **Prompt chaining** — output of one prompt feeds into next prompt

### Prompt Design Patterns
| Pattern | When | Example |
|---------|------|---------|
| **Role assignment** | Need domain expertise | "You are a senior database engineer..." |
| **Output formatting** | Need structured output | "Respond in JSON with fields: ..." |
| **Constraint setting** | Limit scope/behavior | "Only use information from the provided context" |
| **Step-by-step** | Complex reasoning | "First analyze X, then evaluate Y, finally recommend Z" |
| **Negative prompting** | Avoid common mistakes | "Do NOT include speculation. Only cite given sources." |

### Optimization & Iteration
- **Prompt versioning** — track prompt changes like code (git, prompt management tools)
- **Temperature tuning** — low (0-0.3) for factual, high (0.7-1.0) for creative
- **Prompt length vs cost** — shorter prompts = faster + cheaper, but less context
- **Evaluation** — systematic testing with evaluation datasets, not just vibes

### Common Pitfalls
- Ambiguous instructions → model guesses intent
- Too many constraints → model ignores some
- Examples that don't match task → confusing signal
- Prompt injection vulnerability → untrusted input in prompt

### Interview Questions
- What is Chain-of-Thought prompting and when does it help?
- Few-shot vs fine-tuning — when to use each?
- How do you systematically improve a prompt?
- What is self-consistency and how does it improve reliability?
- How do you prevent prompt injection in production?

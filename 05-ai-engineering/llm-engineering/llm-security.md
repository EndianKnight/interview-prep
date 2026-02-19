> **TODO:** Expand this guide with detailed content.

# LLM Security

Attacks, defenses, and guardrails for production LLM applications — a critical topic for AI engineering interviews.

## Topics to Cover

### Prompt Injection
- **Direct injection** — user overrides system prompt ("ignore previous instructions")
- **Indirect injection** — malicious content in retrieved documents / tool outputs
- **Defense** — input sanitization, privilege separation, output filtering
- Prompt injection vs jailbreaking (different threat models)

### Jailbreaking
- Techniques — role-playing, encoding tricks, multi-turn escalation
- DAN-style attacks, persona manipulation
- Why it's hard to fully prevent (model generalization)

### Data Exfiltration
- Extracting training data / system prompts
- Tool-use exploitation — tricking model into leaking data via API calls
- Markdown/image injection for data exfiltration

### Guardrails & Defenses
- **Input guards** — classify user input before sending to model
- **Output guards** — filter/classify model output before returning
- **System prompt hardening** — clear boundaries, instruction hierarchy
- **Frameworks** — Guardrails AI, NeMo Guardrails, Llama Guard
- **Constitutional AI** — training models to self-police

### Content Safety
- Toxicity detection, hate speech filtering
- PII detection and redaction
- NSFW content filtering
- Bias detection and mitigation

### Red Teaming
- Systematic adversarial testing
- Automated red teaming tools
- Evaluation benchmarks — HarmBench, TruthfulQA
- Bug bounty programs for AI systems

### Production Patterns
- Defense in depth — multiple layers of protection
- Monitoring and alerting on suspicious patterns
- Rate limiting and abuse detection
- Audit logging for compliance

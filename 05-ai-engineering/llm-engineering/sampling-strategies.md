> **TODO:** Expand this guide with detailed content.

# Sampling Strategies

How LLMs generate text one token at a time.

- **Greedy Search:** Always pick the most likely token.
- **Beam Search:** Keep track of K most likely sequences.
- **Temperature:** Controls randomness (Low = deterministic, High = creative).
- **Top-K Sampling:** Pick from top K tokens.
- **Top-P (Nucleus) Sampling:** Pick from smallest set of tokens whose cumulative probability exceeds P.

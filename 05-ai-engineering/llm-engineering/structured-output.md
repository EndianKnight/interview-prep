> **TODO:** Expand this guide with detailed content.

# Structured Output & Constrained Generation

Forcing LLMs to produce valid, parseable output — JSON, function calls, and schema-constrained responses.

## Topics to Cover

### Why Structured Output
- Downstream systems need parseable data (APIs, databases, UIs)
- Free-form text is unreliable for programmatic consumption
- Validation and error handling at scale

### Techniques
- **JSON mode** — model outputs valid JSON (OpenAI, Anthropic)
- **Function calling / Tool use** — model selects and fills function schemas
- **Grammar-constrained decoding** — restrict token generation to valid grammar (llama.cpp, Outlines)
- **Schema-guided generation** — provide JSON Schema, model fills values
- **Pydantic / Zod integration** — define output types in code, validate automatically

### Implementation Patterns
- Retry with error feedback — parse failure → send error back → regenerate
- Streaming structured output — partial JSON parsing
- Nested schemas — complex objects with arrays and references
- Enum constraints — restrict values to predefined options
- Optional fields and defaults

### Evaluation
- Schema compliance rate
- Field accuracy (correct values, not just valid structure)
- Latency impact of constrained decoding
- Handling edge cases (empty arrays, null values, unicode)

### Best Practices
- Keep schemas simple — fewer fields = higher reliability
- Provide examples in the prompt
- Use descriptions in schema fields to guide the model
- Validate at the application layer, not just the model layer

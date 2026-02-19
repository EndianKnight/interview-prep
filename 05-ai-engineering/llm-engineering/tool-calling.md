> **TODO:** Expand this guide with detailed content.

# Tool Calling (Function Calling)

The mechanism by which LLMs interact with external systems — API formats, schema design, and execution patterns.

> **Scope:** This guide covers the **technical mechanism** of tool calling. For agent architectures that use tools, see [agents-and-tools.md](agents-and-tools.md). For the standardized protocol, see [mcp.md](mcp.md).

## Topics to Cover

### How Tool Calling Works
- Model receives tool definitions (name, description, parameter schema)
- Model outputs structured tool call (function name + arguments as JSON)
- Application executes the tool and returns results to the model
- Model incorporates results into its response

### API Formats
| Provider | Format | Key Feature |
|----------|--------|-------------|
| OpenAI | `tools` array with JSON Schema | Parallel tool calls, forced tool use |
| Anthropic | `tools` array with JSON Schema | Tool use blocks in content, streaming |
| Google (Gemini) | `function_declarations` | Grounding with Google Search |
| Open source (Ollama, vLLM) | OpenAI-compatible | Varies by model capability |

### Schema Design
- **Clear descriptions** — model relies on description to decide when/how to use tool
- **Parameter types** — string, number, boolean, enum, array, object
- **Required vs optional** — mark required params, provide defaults for optional
- **Enum constraints** — restrict values to valid options
- **Nested objects** — for complex inputs (address with street, city, zip)

### Execution Patterns
- **Single tool call** — model calls one tool, gets result, responds
- **Parallel tool calls** — model requests multiple tools simultaneously
- **Sequential (chained)** — model calls tool A, uses result to call tool B
- **Forced tool use** — require model to use a specific tool (or any tool)
- **Tool choice** — let model decide whether to use tools or respond directly

### Error Handling
- Tool execution failure → return error message to model → model adapts
- Timeout handling — set max execution time, return timeout error
- Validation — validate tool arguments before execution
- Retry logic — model can retry with corrected arguments

### Best Practices
- Provide 5-20 tools max (too many confuses the model)
- Write descriptions from the model's perspective ("Use this to look up...")
- Include examples in descriptions for complex tools
- Test edge cases: model calls wrong tool, invalid arguments, missing required fields

### Interview Questions
- How does function calling work under the hood?
- How do you handle errors when a tool call fails?
- Parallel vs sequential tool calls — when does each happen?
- How do you design tool schemas for reliability?
- What are the security considerations for tool calling?

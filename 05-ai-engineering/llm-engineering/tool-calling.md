> **TODO:** Expand this guide with detailed content.

# Tool Calling (Function Calling)

Enabling LLMs to interact with external systems.

- **Mechanism:** LLM outputs structured JSON (e.g., `{"tool": "weather", "args": {"city": "London"}}`) → App executes tool → App feeds result back to LLM.
- **Formats:** OpenAI Function Calling, Anthropic Tool Use.
- **Libraries:** LangChain Tools, LlamaIndex Tools.

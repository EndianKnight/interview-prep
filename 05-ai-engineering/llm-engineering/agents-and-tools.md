> **TODO:** Expand this guide with detailed content.

# Agents & Tools

LLM-powered autonomous agents — architectures, frameworks, and system design for building agent applications.

> **Scope:** This guide covers agent **architectures and frameworks**. For agent design patterns (ReAct, reflection, etc.), see [agentic-patterns.md](agentic-patterns.md). For tool calling mechanics, see [tool-calling.md](tool-calling.md).

## Topics to Cover

### What Is an Agent
- LLM that can reason, plan, and take actions via tools
- Core loop: perceive → reason → act → observe → repeat
- Agents vs chains — agents decide *which* actions to take dynamically

### Agent Architecture
- **LLM (brain)** — reasoning, planning, decision-making
- **Tools** — external capabilities (APIs, databases, code execution, search)
- **Memory** — context from current and past interactions
- **Planning** — breaking tasks into steps, tracking progress
- **Observation** — processing tool results, updating understanding

### Agent Frameworks
| Framework | Language | Best For |
|-----------|----------|----------|
| LangChain / LangGraph | Python, JS | Graph-based agent workflows, stateful agents |
| LlamaIndex | Python | Data-centric agents, RAG agents |
| Claude Agent SDK | Python | Anthropic-native, tool use + computer use |
| CrewAI | Python | Multi-agent systems, role-based agents |
| AutoGen | Python | Multi-agent conversation, Microsoft ecosystem |
| Semantic Kernel | C#, Python | Microsoft enterprise, planner-based |

### Memory Systems
| Type | Persistence | Example |
|------|------------|---------|
| **Conversation buffer** | Session | Last N messages in context |
| **Summary memory** | Session | Compressed summary of conversation |
| **Vector store memory** | Long-term | Embed and retrieve past interactions |
| **Entity memory** | Long-term | Track facts about specific entities |

### Tool Design for Agents
- Tools should be atomic, well-described, and composable
- Include input validation and clear error messages
- Consider idempotency for tools with side effects
- Limit tool set to what's needed (too many tools = confusion)

### System Design Considerations
- **Reliability** — retry logic, fallback strategies, human-in-the-loop escalation
- **Cost control** — token budgets, max iterations, caching
- **Observability** — trace each step (reasoning, tool calls, results)
- **Safety** — sandbox tool execution, approval for destructive actions
- **Evaluation** — end-to-end task success rate, not just individual step quality

### Interview Questions
- How would you architect an agent system for customer support?
- How do you handle agent failures and infinite loops?
- LangChain vs LlamaIndex — when to use which?
- How do you manage cost and latency in a multi-step agent?
- How do you evaluate an agent system end-to-end?

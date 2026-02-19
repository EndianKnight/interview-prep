> **TODO:** Expand this guide with detailed content.

# Model Context Protocol (MCP)

An open standard for connecting AI models to data sources and tools — solving the N×M integration problem.

> **Scope:** This guide covers the MCP protocol and ecosystem. For general tool calling mechanics, see [tool-calling.md](tool-calling.md). For agent architectures, see [agents-and-tools.md](agents-and-tools.md).

## Topics to Cover

### The Problem MCP Solves
- Without MCP: every AI app must build custom integrations for every tool (N apps × M tools)
- With MCP: standard protocol means any MCP client works with any MCP server
- Analogous to USB — universal connector for AI ↔ tools

### Architecture
- **MCP Hosts** — AI applications that connect to servers (Claude Desktop, IDEs, custom apps)
- **MCP Clients** — protocol clients within hosts that manage server connections
- **MCP Servers** — lightweight services that expose tools, resources, and prompts
- **Transport** — stdio (local) or SSE/HTTP (remote)

### Core Primitives
| Primitive | Description | Example |
|-----------|-------------|---------|
| **Tools** | Functions the model can call | `query_database`, `create_issue`, `send_email` |
| **Resources** | Data the model can read | File contents, database schemas, API docs |
| **Prompts** | Reusable prompt templates | "Summarize this PR", "Review this code" |

### Protocol Details
- **JSON-RPC 2.0** based — request/response + notifications
- **Capability negotiation** — client and server declare supported features
- **Lifecycle** — initialize → list tools/resources → call tools → shutdown
- **Streaming** — support for long-running operations

### Building MCP Servers
- SDKs available: Python, TypeScript, Java, Kotlin
- Define tools with name, description, input schema (JSON Schema)
- Handle tool execution and return results
- Test with MCP Inspector

### Ecosystem
- **Official servers** — GitHub, Google Drive, Postgres, Slack, filesystem
- **Community servers** — hundreds of community-built integrations
- **Registries** — discovering and sharing MCP servers

### Security Considerations
- Server runs with user's permissions — principle of least privilege
- Input validation on all tool arguments
- Sandboxing for filesystem/command execution tools
- OAuth for remote server authentication

### Interview Questions
- What problem does MCP solve and how is it different from direct tool calling?
- Explain the MCP architecture (hosts, clients, servers)
- How would you build an MCP server for a custom internal tool?
- What are the security considerations for MCP in production?
- Tools vs Resources vs Prompts — when to use each?

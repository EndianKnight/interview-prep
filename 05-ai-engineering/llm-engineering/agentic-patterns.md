> **TODO:** Expand this guide with detailed content.

# Agentic Patterns

Design patterns for building autonomous LLM agents — reasoning strategies, multi-agent coordination, and control flow.

> **Scope:** This guide covers agent **design patterns**. For agent architectures and frameworks, see [agents-and-tools.md](agents-and-tools.md).

## Topics to Cover

### Single-Agent Patterns

#### ReAct (Reason + Act)
- Interleave reasoning traces with tool actions
- Loop: Thought → Action → Observation → Thought → ...
- Most common agent pattern, supported by all frameworks

#### Plan-and-Solve
- Generate full plan upfront, then execute step by step
- Variants: plan-and-execute, least-to-most decomposition
- Better for complex multi-step tasks than pure ReAct

#### Reflection / Self-Correction
- Generate output → critique it → revise based on critique
- Can use same model or separate "critic" model
- Reflexion: maintain memory of past attempts and failures

#### Tool-Augmented Generation
- LLM decides when to call tools during generation
- Toolformer-style: model learns to insert tool calls inline
- Code interpreter: generate and execute code for computation

### Multi-Agent Patterns

#### Supervisor / Manager
- Orchestrator agent delegates to specialist agents
- Manager handles routing, aggregation, error handling
- Example: research agent + writing agent + review agent

#### Debate / Adversarial
- Multiple agents argue different positions
- Consensus or judge model selects best response
- Improves reasoning quality through diverse perspectives

#### Pipeline / Assembly Line
- Each agent handles one stage, passes to next
- Analyst → Writer → Reviewer → Publisher
- Clear separation of concerns

#### Swarm / Collaborative
- Agents work in parallel on different aspects
- Combine results (merge, vote, synthesize)
- Good for tasks that are naturally parallelizable

### Control Flow Patterns
- **Sequential** — step by step, each depends on previous
- **Parallel** — independent subtasks run simultaneously
- **Conditional** — branch based on intermediate results
- **Loop** — retry/iterate until quality threshold met
- **Human-in-the-loop** — pause for human approval at critical steps

### State Management
- **Conversation state** — messages exchanged so far
- **Task state** — current plan, completed steps, remaining work
- **World state** — external system state (database, files, APIs)
- **Checkpointing** — save state for recovery from failures

### Interview Questions
- ReAct vs Plan-and-Solve — when to use each?
- How do you design a multi-agent system? What are the coordination challenges?
- How do you prevent infinite loops in agentic systems?
- What is the reflection pattern and when does it improve quality?
- How would you add human-in-the-loop to an agent workflow?

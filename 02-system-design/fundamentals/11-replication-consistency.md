# Replication & Consistency Models

> TODO: Detailed guide with examples, diagrams, and interview questions

## Topics to Cover
- Leader-follower (master-slave) replication
- Multi-leader replication
- Leaderless replication (Dynamo-style)
- Synchronous vs asynchronous replication
- Consistency models: strong, eventual, causal, read-your-writes
- **CRDT** (Conflict-free Replicated Data Types) — types, use cases, tradeoffs
- **Vector clocks** — logical timestamps for conflict detection
- **Quorum reads/writes** — R + W > N formula, sloppy quorum
- Conflict resolution strategies (last-write-wins, merge, application-level)

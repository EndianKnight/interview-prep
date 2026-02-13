# Synchronization & Locking

Coordinating access to shared resources in distributed systems — from database locks to distributed mutexes.

---

## Why Distributed Locking?

In a single process, you use mutexes. In a distributed system, you need **distributed locks**:
- **Prevent double-processing** — only one worker handles a job
- **Protect shared resources** — only one service writes to a resource
- **Leader election** — only one instance acts as leader
- **Rate limiting** — distributed token bucket coordination

---

## Distributed Lock Implementations

### Redis-Based Locking (Simple)

```mermaid
sequenceDiagram
    participant C1 as Client 1
    participant C2 as Client 2
    participant Redis

    C1->>Redis: SET lock:resource NX EX 10
    Redis-->>C1: OK (lock acquired)

    C2->>Redis: SET lock:resource NX EX 10
    Redis-->>C2: nil (lock not acquired)

    Note over C1: Do work...

    C1->>Redis: DEL lock:resource (release)
```

```
SET lock:resource <unique-token> NX EX 10
  NX = only set if not exists
  EX = expire after 10 seconds (safety net)
  unique-token = prevent releasing someone else's lock
```

**Release safely (Lua script for atomicity):**
```lua
if redis.call("get", KEYS[1]) == ARGV[1] then
    return redis.call("del", KEYS[1])
else
    return 0
end
```

### Redlock (Multi-Node Redis)

For single-node Redis, if the node crashes, the lock is lost. Redlock uses multiple independent Redis nodes:

```mermaid
graph LR
    Client -->|"SET NX"| R1[Redis 1]
    Client -->|"SET NX"| R2[Redis 2]
    Client -->|"SET NX"| R3[Redis 3]
    Client -->|"SET NX"| R4[Redis 4]
    Client -->|"SET NX"| R5[Redis 5]
```

1. Try to acquire lock on all N nodes (e.g., 5)
2. Lock is acquired if **majority** (3+) succeed within time limit
3. Lock validity = TTL - time spent acquiring

**Controversy:** Martin Kleppmann argues Redlock has edge cases (GC pauses, clock drift). For critical correctness, use ZooKeeper or database locks.

### ZooKeeper-Based Locking

```mermaid
sequenceDiagram
    participant C1 as Client 1
    participant C2 as Client 2
    participant ZK as ZooKeeper

    C1->>ZK: Create /locks/resource-001 (ephemeral + sequential)
    ZK-->>C1: Created /locks/resource-001

    C2->>ZK: Create /locks/resource-002 (ephemeral + sequential)
    ZK-->>C2: Created /locks/resource-002

    Note over C1: Lowest sequence → lock holder
    Note over C2: Watch /locks/resource-001

    C1->>ZK: Delete /locks/resource-001
    ZK->>C2: Watch triggered
    Note over C2: Now lowest → lock acquired
```

**Advantages over Redis:**
- **Ephemeral nodes** — lock auto-released if client dies (no stale locks)
- **Sequential nodes** — fair ordering (no thundering herd)
- **Strong consistency** — ZAB consensus guarantees

### Database-Based Locking

```sql
-- Pessimistic lock
SELECT * FROM orders WHERE id = 123 FOR UPDATE;
-- Row is locked until transaction commits/rolls back

-- Advisory lock (PostgreSQL)
SELECT pg_advisory_lock(123);
-- Application-level lock, not tied to a row

-- Optimistic lock
UPDATE orders SET status = 'completed', version = version + 1
WHERE id = 123 AND version = 5;
-- Returns 0 rows if version changed (someone else updated first)
```

---

## Optimistic vs Pessimistic Locking

```mermaid
graph TD
    subgraph "Pessimistic (lock first, then work)"
        P1["1. Acquire lock"] --> P2["2. Read data"]
        P2 --> P3["3. Modify data"]
        P3 --> P4["4. Release lock"]
    end

    subgraph "Optimistic (work first, check on save)"
        O1["1. Read data + version"] --> O2["2. Modify locally"]
        O2 --> O3["3. Write if version unchanged"]
        O3 -->|"Version match"| O4["Success"]
        O3 -->|"Version mismatch"| O5["Retry"]
    end
```

| Feature | Optimistic | Pessimistic |
|---------|-----------|-------------|
| **When to lock** | At save time (version check) | Before reading |
| **Contention** | Best for low contention | Best for high contention |
| **Blocking** | Non-blocking (retry on conflict) | Blocking (waits for lock) |
| **Throughput** | Higher (under low contention) | Lower (locks held longer) |
| **Complexity** | Handle retries in application | Handle deadlocks |
| **Example** | `@Version` in JPA, ETags | `SELECT FOR UPDATE`, Redis SETNX |

---

## Fencing Tokens

Prevent stale lock holders from making changes after their lock expires:

```mermaid
sequenceDiagram
    participant C1 as Client 1
    participant C2 as Client 2
    participant Lock as Lock Service
    participant DB as Storage

    C1->>Lock: Acquire lock
    Lock-->>C1: Token #33

    Note over C1: GC pause... lock expires

    C2->>Lock: Acquire lock
    Lock-->>C2: Token #34

    C2->>DB: Write (token #34)
    DB-->>C2: OK

    C1->>DB: Write (token #33)
    DB-->>C1: REJECTED (token #33 < #34)
```

**Key insight:** The storage service must reject writes with a fencing token lower than what it has seen. This prevents the "zombie leader" problem.

---

## Comparison of Approaches

| Approach | Consistency | Performance | Complexity | Best For |
|----------|-----------|-------------|-----------|----------|
| **Redis SETNX** | Weak (single node) | Very fast | Simple | Non-critical locks, rate limiting |
| **Redlock** | Moderate | Fast | Moderate | Moderate criticality |
| **ZooKeeper** | Strong | Slower | High (ZK cluster) | Critical locks, leader election |
| **DB locks** | Strong (within DB) | Moderate | Simple | Already using DB, per-row locks |
| **Optimistic (version)** | Strong | Fast (low contention) | Moderate | Concurrent writes, CAS |

---

## Common Interview Questions

1. **"How do you implement a distributed lock?"** → Redis SETNX with TTL for simple cases. ZooKeeper ephemeral nodes for critical correctness. Always include fencing tokens.
2. **"Optimistic vs pessimistic locking?"** → Optimistic: read + version check on write (low contention). Pessimistic: lock before reading (high contention). Start with optimistic.
3. **"What happens if the lock holder crashes?"** → TTL expiration (Redis) or ephemeral node deletion (ZooKeeper). Always have a timeout as safety net.
4. **"What's the fencing token problem?"** → Stale lock holder (GC pause, network delay) writes after new holder. Fix: monotonically increasing fencing tokens rejected by storage.
5. **"Redlock — safe or not?"** → Works for most cases but not 100% safe under extreme conditions (GC pauses, clock drift). Use ZooKeeper for life-or-death correctness.

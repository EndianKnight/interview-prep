# Event-Driven Architecture

Building systems around events — decouple services, enable real-time processing, and create auditable, replayable systems.

---

## Core Concepts

```mermaid
graph LR
    Producer["Event Producer<br/>(Order Service)"] -->|"OrderCreated"| Broker["Event Broker<br/>(Kafka)"]
    Broker --> C1["Consumer 1<br/>(Inventory Service)"]
    Broker --> C2["Consumer 2<br/>(Notification Service)"]
    Broker --> C3["Consumer 3<br/>(Analytics Service)"]
```

| Concept | Description |
|---------|-------------|
| **Event** | A record of something that happened (immutable fact) |
| **Producer** | Service that publishes events |
| **Consumer** | Service that reacts to events |
| **Broker** | Middleware that routes events (Kafka, RabbitMQ, SNS) |
| **Event schema** | Structure of the event payload (use Avro, Protobuf, or JSON Schema) |

### Event Types

| Type | Description | Example |
|------|-------------|---------|
| **Domain event** | Business fact that happened | `OrderPlaced`, `PaymentProcessed` |
| **Integration event** | Event shared across services | `UserCreated` → notify all services |
| **Command** | Request to do something | `ProcessPayment` (imperative, not a fact) |

---

## Event Sourcing

Instead of storing current state, store the **sequence of events** that produced it.

```mermaid
graph TD
    subgraph "Traditional (State-Based)"
        TB[Account Table<br/>id: 1, balance: 150]
    end

    subgraph "Event Sourcing"
        E1["Event 1: AccountCreated(balance: 0)"]
        E2["Event 2: MoneyDeposited(amount: 200)"]
        E3["Event 3: MoneyWithdrawn(amount: 50)"]
        E1 --> E2 --> E3
        E3 -->|"Replay events"| State["Current State<br/>balance: 150"]
    end
```

| Pros | Cons |
|------|------|
| Full audit trail (every change recorded) | Event store grows forever (snapshots needed) |
| Time travel (reconstruct any past state) | Complex queries (need projections/CQRS) |
| Replay to fix bugs or build new views | Event schema evolution is tricky |
| Natural fit for event-driven architecture | Steeper learning curve |

### Event Store

```
Event Store (append-only log):
┌────┬──────────────┬──────────┬───────────────────────────┐
│ ID │ Aggregate ID │ Version  │ Event Data                │
├────┼──────────────┼──────────┼───────────────────────────┤
│ 1  │ order-123    │ 1        │ OrderCreated{items: [...]} │
│ 2  │ order-123    │ 2        │ PaymentReceived{amount: 50}│
│ 3  │ order-123    │ 3        │ OrderShipped{tracking: ...}│
└────┴──────────────┴──────────┴───────────────────────────┘
```

**Tools:** EventStoreDB, Kafka (as event store), Axon Framework, PostgreSQL (with append-only pattern).

---

## CQRS (Command Query Responsibility Segregation)

Separate the read and write models for different optimization.

```mermaid
graph TD
    Command["Command<br/>(Create Order)"] --> WriteModel["Write Model<br/>(Normalized, ACID)"]
    WriteModel --> EventStore["Event Store / DB"]
    EventStore -->|"Publish events"| Projector["Event Projector"]
    Projector --> ReadModel["Read Model<br/>(Denormalized, fast)"]
    Query["Query<br/>(Get Order Details)"] --> ReadModel
```

### CQRS Models

| Model | Optimized For | Storage | Example |
|-------|--------------|---------|---------|
| **Write** | Consistency, validation | Normalized SQL, event store | PostgreSQL |
| **Read** | Query performance | Denormalized, cached | Elasticsearch, Redis, materialized views |

| Pros | Cons |
|------|------|
| Optimize reads and writes independently | Complexity (two models to maintain) |
| Scale read and write separately | Eventual consistency between models |
| Different tech for each (SQL + Elasticsearch) | Must handle projection lag |

**When to use:** Read/write patterns differ significantly, high read:write ratio, complex queries on read side.

---

## Outbox Pattern

Ensure events are published reliably when database state changes.

**Problem:** Writing to DB and publishing to Kafka isn't atomic — one might succeed and the other fail.

```mermaid
sequenceDiagram
    participant App
    participant DB as Database
    participant CDC as CDC / Poller
    participant Kafka

    App->>DB: BEGIN TRANSACTION
    App->>DB: INSERT INTO orders (...)
    App->>DB: INSERT INTO outbox (event_data)
    App->>DB: COMMIT

    Note over CDC: Polls outbox table or uses CDC

    CDC->>DB: Read new outbox entries
    CDC->>Kafka: Publish event
    CDC->>DB: Mark outbox entry as published
```

**Two approaches:**
1. **Polling publisher** — periodically query outbox table for unpublished events
2. **CDC (Change Data Capture)** — use Debezium to capture DB changes and publish to Kafka

---

## Event-Driven Patterns

### Event Notification

Fire-and-forget: "something happened, react if you care."

```mermaid
graph LR
    Order["Order Service<br/>OrderPlaced"] -->|Event| Kafka
    Kafka --> Inventory["Inventory: Reserve stock"]
    Kafka --> Email["Email: Send confirmation"]
    Kafka --> Analytics["Analytics: Track order"]
```

### Event-Carried State Transfer

Event includes all the data the consumer needs (no callback needed):

```json
{
  "type": "OrderPlaced",
  "data": {
    "orderId": "123",
    "userId": "456",
    "items": [{"productId": "789", "qty": 2, "price": 29.99}],
    "total": 59.98,
    "shippingAddress": { "city": "NYC", "zip": "10001" }
  }
}
```

**Pros:** Consumer is self-sufficient (no API calls needed). **Cons:** Large events, data coupling.

### Choreography vs Orchestration

```mermaid
graph TD
    subgraph "Choreography (decentralized)"
        CE1["OrderCreated"] --> CE2["PaymentCharged"]
        CE2 --> CE3["InventoryReserved"]
        CE3 --> CE4["OrderConfirmed"]
    end
```

```mermaid
graph TD
    subgraph "Orchestration (centralized)"
        Orch["Order Saga<br/>Orchestrator"]
        Orch -->|1| Pay["Charge Payment"]
        Orch -->|2| Inv["Reserve Inventory"]
        Orch -->|3| Ship["Schedule Shipping"]
    end
```

| | Choreography | Orchestration |
|---|---|---|
| **Control** | Distributed (each service decides) | Centralized (orchestrator decides) |
| **Coupling** | Lower (events only) | Higher (orchestrator knows all steps) |
| **Visibility** | Hard to see full flow | Easy (one place to see workflow) |
| **Best for** | Simple flows, few services | Complex flows, many steps |

---

## Schema Evolution

Events live forever — how do you change their schema?

| Strategy | How | Tradeoff |
|----------|-----|----------|
| **Backward compatible** | Only add optional fields | Safe, recommended default |
| **Schema registry** | Centralize schemas (Confluent Schema Registry) | Enforced compatibility |
| **Versioned events** | `OrderCreatedV1`, `OrderCreatedV2` | Clear, but clutters codebase |
| **Upcasting** | Transform old events to new format on read | Flexible, complex |

---

## Common Interview Questions

1. **"What is event sourcing?"** → Store events (facts) instead of current state. Replay events to derive state. Full audit trail, time travel, replayable.
2. **"What is CQRS?"** → Separate read and write models. Write model optimized for consistency, read model for query performance. Eventually consistent.
3. **"How do you ensure events are published reliably?"** → Outbox pattern: write event to DB table in same transaction as state change. CDC or poller publishes to Kafka.
4. **"Choreography vs orchestration?"** → Choreography: each service reacts to events (decentralized). Orchestration: central coordinator manages workflow (centralized). Choreography for simple, orchestration for complex.
5. **"How do you handle event schema changes?"** → Backward-compatible changes only (add optional fields). Use schema registry for enforcement.

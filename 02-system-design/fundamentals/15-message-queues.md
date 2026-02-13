# Message Queues

Asynchronous communication between services — decouple producers from consumers for scalability, reliability, and resilience.

---

## Why Message Queues?

```mermaid
graph LR
    subgraph Without Queue
        A1[Service A] -->|Direct call, tightly coupled| B1[Service B]
    end
```

```mermaid
graph LR
    subgraph With Queue
        A2[Service A<br/>Producer] -->|Publish| Q[Message Queue]
        Q -->|Consume| B2[Service B<br/>Consumer]
    end
```

| Benefit | How |
|---------|-----|
| **Decoupling** | Producer doesn't know/care about consumer |
| **Buffering** | Queue absorbs traffic spikes |
| **Reliability** | Messages persist even if consumer is down |
| **Scalability** | Add more consumers to increase throughput |
| **Async processing** | Producer doesn't wait for consumer to finish |

---

## Message Queue vs Event Stream

```mermaid
graph TD
    subgraph "Message Queue (point-to-point)"
        P1[Producer] --> Q1[Queue]
        Q1 -->|"Message consumed<br/>and deleted"| C1[Consumer]
    end

    subgraph "Event Stream (pub/sub + replay)"
        P2[Producer] --> T1[Topic / Log]
        T1 -->|Read offset 5| CG1[Consumer Group A]
        T1 -->|Read offset 3| CG2[Consumer Group B]
    end
```

| Feature | Message Queue | Event Stream |
|---------|-------------|-------------|
| **Message lifecycle** | Deleted after consumption | Retained (configurable, even forever) |
| **Replay** | ❌ Can't re-read | ✅ Replay from any offset |
| **Consumer groups** | Each message to one consumer | Each group gets all messages independently |
| **Ordering** | Per-queue FIFO | Per-partition ordering |
| **Use case** | Task processing, job queues | Event sourcing, analytics, data pipelines |
| **Examples** | RabbitMQ, SQS, ActiveMQ | Kafka, Amazon Kinesis, Pulsar |

---

## Apache Kafka — Deep Dive

### Architecture

```mermaid
graph TD
    P1[Producer 1] --> T["Topic: orders"]
    P2[Producer 2] --> T

    T --> Part0["Partition 0"]
    T --> Part1["Partition 1"]
    T --> Part2["Partition 2"]

    Part0 --> B1[Broker 1]
    Part1 --> B2[Broker 2]
    Part2 --> B3[Broker 3]

    CG1[Consumer Group A] --> Part0
    CG1 --> Part1
    CG1 --> Part2

    CG2[Consumer Group B] --> Part0
    CG2 --> Part1
    CG2 --> Part2
```

### Key Concepts

| Concept | Description |
|---------|-------------|
| **Topic** | Named feed of messages (like a category) |
| **Partition** | Ordered, immutable log within a topic |
| **Offset** | Position of a message within a partition |
| **Producer** | Publishes messages to topics |
| **Consumer** | Reads messages from topics |
| **Consumer Group** | Set of consumers sharing the work (each partition → one consumer in group) |
| **Broker** | Kafka server that stores partitions |
| **Replication Factor** | Number of copies of each partition (typically 3) |

### Kafka Guarantees

| Guarantee | How |
|-----------|-----|
| **Ordering** | Within a partition only. Use same key for related messages. |
| **At-least-once** | Default. Consumer may process duplicates after restart. |
| **Exactly-once** | With idempotent producers + transactional consumers. |
| **Durability** | Messages replicated to N brokers, configurable acks. |

### Producer Acknowledgments

| `acks` | Meaning | Durability | Latency |
|--------|---------|-----------|---------|
| `0` | Don't wait for any ACK | Lowest (may lose) | Fastest |
| `1` | Wait for leader ACK | Medium | Medium |
| `all` | Wait for all ISR replicas | Highest | Slowest |

---

## RabbitMQ

Traditional message broker with rich routing capabilities.

```mermaid
graph LR
    P[Producer] --> X[Exchange]
    X -->|Routing key: user.created| Q1[Queue: user-events]
    X -->|Routing key: order.created| Q2[Queue: order-events]
    X -->|Routing key: *.created| Q3[Queue: all-created]
    Q1 --> C1[Consumer 1]
    Q2 --> C2[Consumer 2]
    Q3 --> C3[Consumer 3]
```

### Exchange Types

| Type | Routing | Use Case |
|------|---------|----------|
| **Direct** | Exact routing key match | Task queues |
| **Fanout** | Broadcast to all bound queues | Notifications |
| **Topic** | Wildcard routing key (`user.*`, `#.created`) | Flexible pub/sub |
| **Headers** | Match on message headers | Complex routing rules |

### Kafka vs RabbitMQ

| Feature | Kafka | RabbitMQ |
|---------|-------|----------|
| **Model** | Distributed log (stream) | Message broker (queue) |
| **Ordering** | Per-partition | Per-queue |
| **Replay** | ✅ (seek to offset) | ❌ (consumed = done) |
| **Throughput** | Very high (millions/sec) | High (tens of thousands/sec) |
| **Routing** | Partition key | Exchange + routing key (flexible) |
| **Use case** | Event streaming, data pipelines | Task queues, RPC, complex routing |

---

## Amazon SQS

Fully managed message queue — simplest option for AWS workloads.

| Feature | SQS Standard | SQS FIFO |
|---------|-------------|----------|
| **Ordering** | Best-effort | Strict FIFO |
| **Deduplication** | No (at-least-once) | 5-minute dedup window |
| **Throughput** | Nearly unlimited | 300 msg/s (batching: 3000) |
| **Use case** | General async, high throughput | Ordering matters |

### SQS Features
- **Visibility timeout** — message hidden from other consumers while being processed
- **Dead letter queue (DLQ)** — failed messages after N retries go here
- **Long polling** — reduce empty responses, lower cost
- **Delay queues** — delay delivery of new messages

---

## Delivery Guarantees

| Guarantee | Description | Example |
|-----------|-------------|---------|
| **At-most-once** | May lose messages, never duplicate | Fire-and-forget logging |
| **At-least-once** | May duplicate, never lose | Default Kafka, SQS |
| **Exactly-once** | No loss, no duplicates | Kafka transactions, hardest to achieve |

**Practical approach:** Use **at-least-once + idempotent consumers**. Make consumers handle duplicates gracefully (idempotency keys, upserts).

---

## Patterns

### Dead Letter Queue (DLQ)

```mermaid
graph LR
    Q[Main Queue] --> Consumer
    Consumer -->|Success| Done[Process complete]
    Consumer -->|Failure x3| DLQ[Dead Letter Queue]
    DLQ --> Alert[Alert + Manual review]
```

### Fan-out

```mermaid
graph TD
    Event[Order Created] --> Topic[SNS Topic]
    Topic --> Q1[SQS: Send confirmation email]
    Topic --> Q2[SQS: Update inventory]
    Topic --> Q3[SQS: Update analytics]
```

### Competing Consumers

```mermaid
graph LR
    Q[Task Queue] --> C1[Worker 1]
    Q --> C2[Worker 2]
    Q --> C3[Worker 3]
```

Each message processed by exactly one worker. Scale by adding workers.

---

## Common Interview Questions

1. **"When would you use a message queue?"** → Async processing (email, notifications), decoupling services, buffering spikes, reliable task processing.
2. **"Kafka vs RabbitMQ?"** → Kafka for event streaming, replay, high throughput. RabbitMQ for task queues, complex routing, lower throughput.
3. **"How do you ensure messages aren't lost?"** → Persistent messages + consumer acknowledgments + replication. Use DLQ for failed processing.
4. **"How do you handle duplicate messages?"** → Idempotent consumers: use message ID as dedup key, upserts instead of inserts, idempotency keys for payments.
5. **"How do you handle message ordering?"** → Use partition key (Kafka) to ensure related messages go to same partition. Or FIFO queues (SQS FIFO).

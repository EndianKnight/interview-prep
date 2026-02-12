# System Design

A structured approach to high-level system design interviews for senior engineers.

## How to Approach a System Design Interview

```
1. Requirements Clarification (3-5 min)
   ├── Functional requirements
   ├── Non-functional requirements (latency, availability, consistency)
   └── Capacity estimation & constraints

2. High-Level Design (10-15 min)
   ├── API design
   ├── Core components & data flow
   └── Database schema

3. Deep Dive (10-15 min)
   ├── Scaling bottlenecks
   ├── Tradeoff analysis
   └── Component-specific design

4. Wrap Up (3-5 min)
   ├── Summarize tradeoffs
   ├── Discuss monitoring & alerting
   └── Future improvements
```

## Fundamentals

Core building blocks every system design relies on:

| # | Topic | Guide |
|---|-------|-------|
| 1 | Networking Basics | [networking-basics](fundamentals/01-networking-basics.md) |
| 2 | DNS & Domain Resolution | [dns](fundamentals/02-dns.md) |
| 3 | Proxies & CDNs | [proxies-and-cdns](fundamentals/03-proxies-and-cdns.md) |
| 4 | Load Balancing | [load-balancing](fundamentals/04-load-balancing.md) |
| 5 | API Design | [api-design](fundamentals/05-api-design.md) |
| 6 | Scalability | [scalability](fundamentals/06-scalability.md) |
| 7 | Caching | [caching](fundamentals/07-caching.md) |
| 8 | Databases | [databases](fundamentals/08-databases.md) |
| 9 | Data Partitioning & Sharding | [data-partitioning](fundamentals/09-data-partitioning.md) |
| 10 | Replication & Consistency Models | [replication-consistency](fundamentals/10-replication-consistency.md) |
| 11 | Consistent Hashing | [consistent-hashing](fundamentals/11-consistent-hashing.md) |
| 12 | CAP Theorem | [cap-theorem](fundamentals/12-cap-theorem.md) |
| 13 | Storage & File Systems | [storage-systems](fundamentals/13-storage-systems.md) |
| 14 | Message Queues | [message-queues](fundamentals/14-message-queues.md) |
| 15 | Batch vs Stream Processing | [batch-stream-processing](fundamentals/15-batch-stream-processing.md) |
| 16 | Search & Indexing | [search-and-indexing](fundamentals/16-search-and-indexing.md) |
| 17 | Unique ID Generation | [unique-id-generation](fundamentals/17-unique-id-generation.md) |
| 18 | Coordination & Consensus | [coordination](fundamentals/18-coordination.md) |
| 19 | Synchronization & Locking | [synchronization-and-locking](fundamentals/19-synchronization-and-locking.md) |
| 20 | Monitoring & Observability | [monitoring-observability](fundamentals/20-monitoring-observability.md) |
| 21 | Security & Authentication | [security-authentication](fundamentals/21-security-authentication.md) |
| 22 | Capacity Estimation | [capacity-estimation](fundamentals/22-capacity-estimation.md) |
| 23 | Tradeoff Analysis | [tradeoff-analysis](fundamentals/23-tradeoff-analysis.md) |
| 24 | Cloud Technologies | [cloud-technologies](fundamentals/24-cloud-technologies.md) |

## Case Studies

Classic system design interview problems:

| Problem | Guide |
|---------|-------|
| URL Shortener | [url-shortener.md](case-studies/url-shortener.md) |
| Rate Limiter | [rate-limiter.md](case-studies/rate-limiter.md) |
| Chat System | [chat-system.md](case-studies/chat-system.md) |
| News Feed | [news-feed.md](case-studies/news-feed.md) |
| Notification Service | [notification-service.md](case-studies/notification-service.md) |
| Distributed Cache | [distributed-cache.md](case-studies/distributed-cache.md) |
| Video Streaming | [video-streaming.md](case-studies/video-streaming.md) |

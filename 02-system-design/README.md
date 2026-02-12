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
| 9 | Data Modeling | [data-modeling](fundamentals/09-data-modeling.md) |
| 10 | Data Partitioning & Sharding | [data-partitioning](fundamentals/10-data-partitioning.md) |
| 11 | Replication & Consistency Models | [replication-consistency](fundamentals/11-replication-consistency.md) |
| 12 | Consistent Hashing | [consistent-hashing](fundamentals/12-consistent-hashing.md) |
| 13 | CAP Theorem | [cap-theorem](fundamentals/13-cap-theorem.md) |
| 14 | Storage & File Systems | [storage-systems](fundamentals/14-storage-systems.md) |
| 15 | Message Queues | [message-queues](fundamentals/15-message-queues.md) |
| 16 | Batch vs Stream Processing | [batch-stream-processing](fundamentals/16-batch-stream-processing.md) |
| 17 | Search & Indexing | [search-and-indexing](fundamentals/17-search-and-indexing.md) |
| 18 | Unique ID Generation | [unique-id-generation](fundamentals/18-unique-id-generation.md) |
| 19 | Coordination & Consensus | [coordination](fundamentals/19-coordination.md) |
| 20 | Synchronization & Locking | [synchronization-and-locking](fundamentals/20-synchronization-and-locking.md) |
| 21 | Monitoring & Observability | [monitoring-observability](fundamentals/21-monitoring-observability.md) |
| 22 | Security & Authentication | [security-authentication](fundamentals/22-security-authentication.md) |
| 23 | Capacity Estimation | [capacity-estimation](fundamentals/23-capacity-estimation.md) |
| 24 | Tradeoff Analysis | [tradeoff-analysis](fundamentals/24-tradeoff-analysis.md) |
| 25 | Cloud Technologies | [cloud-technologies](fundamentals/25-cloud-technologies.md) |
| 26 | Rate Limiting & Throttling | [rate-limiting](fundamentals/26-rate-limiting.md) |
| 27 | Microservices Patterns | [microservices-patterns](fundamentals/27-microservices-patterns.md) |
| 28 | Event-Driven Architecture | [event-driven-architecture](fundamentals/28-event-driven-architecture.md) |

## Case Studies

Classic system design interview problems:

| Problem | Key Concepts | Guide |
|---------|-------------|-------|
| URL Shortener | Hashing, base62, read-heavy | [url-shortener](case-studies/url-shortener.md) |
| Rate Limiter | Token bucket, sliding window | [rate-limiter](case-studies/rate-limiter.md) |
| Chat System | WebSocket, presence, fanout | [chat-system](case-studies/chat-system.md) |
| News Feed | Fanout, ranking, caching | [news-feed](case-studies/news-feed.md) |
| Notification Service | Push/pull, priority, templates | [notification-service](case-studies/notification-service.md) |
| Distributed Cache | Consistent hashing, eviction | [distributed-cache](case-studies/distributed-cache.md) |
| Video Streaming | CDN, adaptive bitrate, encoding | [video-streaming](case-studies/video-streaming.md) |
| Typeahead / Autocomplete | Trie, ranking, prefix search | [typeahead-autocomplete](case-studies/typeahead-autocomplete.md) |
| Web Crawler | BFS, politeness, dedup | [web-crawler](case-studies/web-crawler.md) |
| Ride Sharing (Uber) | Geospatial index, matching, ETA | [ride-sharing](case-studies/ride-sharing.md) |
| Collaborative Editing | OT/CRDT, real-time sync | [collaborative-editing](case-studies/collaborative-editing.md) |
| Payment System | Idempotency, distributed txn | [payment-system](case-studies/payment-system.md) |
| File Storage (Dropbox) | Chunking, sync, dedup | [file-storage](case-studies/file-storage.md) |
| Ticket Booking | Seat locking, concurrency | [ticket-booking](case-studies/ticket-booking.md) |

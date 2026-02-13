# Capacity Estimation

Back-of-the-envelope calculations for system design interviews — quick math to validate your design can handle the load.

---

## Key Numbers to Memorize

### Powers of 2

| Power | Value | Approx |
|-------|-------|--------|
| 2^10 | 1,024 | ~1 Thousand (1 KB) |
| 2^20 | 1,048,576 | ~1 Million (1 MB) |
| 2^30 | 1,073,741,824 | ~1 Billion (1 GB) |
| 2^40 | | ~1 Trillion (1 TB) |

### Latency Numbers

| Operation | Latency |
|-----------|---------|
| L1 cache reference | 1 ns |
| L2 cache reference | 4 ns |
| Main memory (RAM) | 100 ns |
| SSD random read | 100 μs |
| HDD random seek | 10 ms |
| Same datacenter round trip | 0.5 ms |
| Cross-continent round trip | 150 ms |
| Read 1 MB from memory | 250 μs |
| Read 1 MB from SSD | 1 ms |
| Read 1 MB from HDD | 20 ms |
| Read 1 MB over 1 Gbps network | 10 ms |

**Key insight:** Memory is 100x faster than SSD, 1000x faster than HDD. Network is slow. Cache aggressively.

### Time Conversions

| Period | Seconds |
|--------|---------|
| 1 day | 86,400 ≈ **~100K** |
| 1 month | 2,592,000 ≈ **~2.5M** |
| 1 year | 31,536,000 ≈ **~30M** |

---

## Estimation Framework

### Step 1: Define Scale

```
"Design Twitter"
- DAU (Daily Active Users): 300M
- Each user: 2 tweets/day (read-heavy: 100 reads per 1 write)
- Tweet size: ~300 bytes text + metadata
```

### Step 2: Calculate QPS

```
Write QPS:
  300M users × 2 tweets/day = 600M tweets/day
  600M / 100K seconds/day = 6,000 tweets/sec
  Peak (2-3x): ~15,000 tweets/sec

Read QPS:
  100:1 read/write ratio
  6,000 × 100 = 600,000 reads/sec
  Peak: ~1.5M reads/sec
```

### Step 3: Calculate Storage

```
Daily:
  600M tweets × 300 bytes = 180 GB/day

Yearly:
  180 GB × 365 = ~65 TB/year

5-year retention:
  65 TB × 5 = ~325 TB (text only)

With media (images, video):
  10% of tweets have images (~500 KB each)
  60M × 500 KB = 30 TB/day → ~11 PB/year
```

### Step 4: Calculate Bandwidth

```
Outgoing (reads):
  600K reads/sec × 300 bytes = 180 MB/sec = ~1.5 Gbps
  With media: 600K × 0.1 × 500KB = 30 GB/sec = ~240 Gbps
```

### Step 5: Calculate Infrastructure

```
Application servers (assuming 10K QPS per server):
  1.5M read QPS / 10K = 150 servers

Cache (80% cache hit rate):
  Top 20% of tweets (hot data): ~13 TB cached
  Redis nodes: 13 TB / 100 GB per node = ~130 Redis nodes

Database:
  325 TB / 2 TB per shard = ~163 shards (with replicas: ~500 DB instances)
```

---

## Quick Reference Formulas

| What | Formula |
|------|---------|
| **QPS** | DAU × actions_per_day / 86400 |
| **Peak QPS** | QPS × 2 (or 3 for spiky systems) |
| **Storage/day** | QPS × object_size × 86400 |
| **Storage/year** | Storage/day × 365 |
| **Bandwidth** | QPS × object_size |
| **Servers needed** | Peak QPS / QPS_per_server |
| **Cache size** | Hot data % × total data |

---

## Common Estimation Scenarios

### URL Shortener
```
Scale: 100M URLs/day created, 10:1 read/write
Write: 100M / 86400 ≈ 1,200/sec
Read:  12,000/sec
Storage: 100M × 500 bytes × 365 days × 5 years ≈ 90 TB
```

### Chat System (WhatsApp-scale)
```
Scale: 2B users, 50M DAU
Messages: 40 messages/user/day
QPS: 50M × 40 / 86400 ≈ 23,000 msg/sec
Storage: 2B msg/day × 100 bytes ≈ 200 GB/day → 73 TB/year
```

### Video Streaming (YouTube-scale)
```
Scale: 1B DAU, 5 videos/day average
Video: 300 MB average (multiple resolutions)
Storage: 500K new videos/day × 300 MB = 150 TB/day
Bandwidth: 5B views/day × 10 MB avg stream = 50 PB/day → ~4.6 Tbps
```

---

## Tips for Interviews

1. **Round aggressively** — use 100K for seconds/day, 2.5M for seconds/month
2. **State assumptions** — "I'll assume 300 bytes per message including metadata"
3. **Show your work** — interviewers care about the process, not exact numbers
4. **Sanity check** — does the answer make sense? Is 1 TB reasonable for a startup?
5. **Start with QPS** — everything flows from QPS (storage, bandwidth, servers)
6. **Peak vs average** — always mention peak (2-3x average), design for peak

---

## Common Interview Questions

1. **"Estimate the storage for a URL shortener"** → URLs/day × bytes per URL × retention period. 100M URLs/day × 500B × 5 years ≈ 90 TB.
2. **"How many servers do you need?"** → Peak QPS / capacity per server. Varies by workload (CPU-bound: ~1K QPS, I/O-bound: ~10K QPS).
3. **"How much bandwidth does a video streaming service need?"** → Views/day × average stream size / 86400. YouTube-scale: ~5 Tbps.
4. **"How do you handle the read-write ratio?"** → High read ratio (100:1) → cache layer + read replicas. High write ratio → message queues + sharding.

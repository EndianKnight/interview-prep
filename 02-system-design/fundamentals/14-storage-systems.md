# Storage Systems

How data is stored at the infrastructure level — block storage, object storage, file systems, and the data structures that power them.

---

## Storage Types

```mermaid
graph TD
    Storage[Storage Types]
    Storage --> Block["Block Storage<br/>EBS, SAN"]
    Storage --> File["File Storage<br/>NFS, EFS"]
    Storage --> Object["Object Storage<br/>S3, GCS, MinIO"]

    Block --> B1["Low-level: raw blocks"]
    Block --> B2["Use: databases, VMs"]
    File --> F1["Hierarchical: directories + files"]
    File --> F2["Use: shared filesystems, NAS"]
    Object --> O1["Flat: bucket + key + metadata"]
    Object --> O2["Use: media, backups, logs, data lake"]
```

| Feature | Block Storage | File Storage | Object Storage |
|---------|-------------|-------------|----------------|
| **Abstraction** | Raw blocks (sectors) | Files in directories | Objects in buckets |
| **Access** | Mount as disk, byte-level | NFS/SMB mount, file-level | REST API (GET/PUT) |
| **Performance** | Highest (low latency) | Moderate | Lower (HTTP overhead) |
| **Scalability** | Limited (single server) | Moderate | Virtually unlimited |
| **Cost** | $$$$ | $$$ | $ |
| **Use case** | Databases, OS disks | Shared files, CMS | Media, backups, data lake |
| **Examples** | AWS EBS, Azure Disk | AWS EFS, NFS | AWS S3, GCS, MinIO |

---

## Write-Ahead Log (WAL)

The foundational durability mechanism for databases.

```mermaid
sequenceDiagram
    participant App
    participant WAL as WAL (Append-Only Log)
    participant DB as Data Files

    App->>WAL: 1. Write operation to log (sequential, fast)
    WAL-->>App: 2. ACK (durable now)
    Note over WAL,DB: 3. Background: flush to data files
    WAL->>DB: Apply changes (async)
```

**Why WAL?**
- Sequential writes to log are **10-100x faster** than random writes to data files
- On crash: replay WAL to recover uncommitted changes
- **Used by:** PostgreSQL, MySQL (InnoDB redo log), SQLite, Kafka, etcd

---

## LSM Tree (Log-Structured Merge Tree)

Optimized for **write-heavy** workloads. Used by Cassandra, RocksDB, LevelDB, HBase.

```mermaid
graph TD
    Write[Write] --> MemTable["MemTable (in-memory sorted)"]
    MemTable -->|"Flush when full"| L0["Level 0 SSTable (on disk)"]
    L0 -->|"Compaction"| L1["Level 1 SSTable"]
    L1 -->|"Compaction"| L2["Level 2 SSTable"]

    Read[Read] --> MemTable
    Read --> L0
    Read --> L1
    Read --> L2
    Read -.->|"Check Bloom filter<br/>before each level"| BF[Bloom Filter]
```

| Operation | How | Performance |
|-----------|-----|-------------|
| **Write** | Append to memtable → flush to SSTable | O(1) amortized — very fast |
| **Read** | Check memtable → L0 → L1 → ... | O(log N) — slower (check multiple levels) |
| **Compaction** | Merge SSTables, remove tombstones | Background I/O |

**Trade-off:** Fast writes, slower reads. Use **Bloom filters** to skip levels that don't contain the key.

### LSM vs B+ Tree

| Feature | LSM Tree | B+ Tree |
|---------|----------|---------|
| **Write** | Fast (sequential, append-only) | Slower (random I/O for page updates) |
| **Read** | Slower (check multiple levels) | Fast (single tree traversal) |
| **Space** | Write amplification from compaction | More predictable |
| **Best for** | Write-heavy (logs, time-series, ingestion) | Read-heavy (OLTP, general purpose) |
| **Used by** | Cassandra, RocksDB, LevelDB | PostgreSQL, MySQL, SQLite |

---

## B+ Tree

The default data structure for most relational database indexes.

```mermaid
graph TD
    Root["Root Node [40]"]
    Root --> L["Internal [10, 25]"]
    Root --> R["Internal [55, 70]"]
    L --> LL["Leaf [5, 10]"]
    L --> LM["Leaf [15, 20, 25]"]
    L --> LR["Leaf [30, 35]"]
    R --> RL["Leaf [40, 45, 50, 55]"]
    R --> RM["Leaf [60, 65, 70]"]
    R --> RR["Leaf [75, 80, 90]"]
    LL --> LM
    LM --> LR
    LR --> RL
    RL --> RM
    RM --> RR
```

Key properties:
- **All data in leaf nodes** (internal nodes are just guides)
- **Leaf nodes linked** for efficient range scans
- **Balanced** — O(log n) for all operations
- **High fan-out** — few levels even for billions of rows (typically 3-4 levels)

---

## Merkle Tree

Used for **data integrity verification** in distributed systems.

```mermaid
graph TD
    Root["Root Hash<br/>H(H12 + H34)"]
    Root --> H12["H12<br/>H(H1 + H2)"]
    Root --> H34["H34<br/>H(H3 + H4)"]
    H12 --> H1["H1<br/>H(Block 1)"]
    H12 --> H2["H2<br/>H(Block 2)"]
    H34 --> H3["H3<br/>H(Block 3)"]
    H34 --> H4["H4<br/>H(Block 4)"]
```

**How it works:**
- Hash every data block at the leaves
- Parent = hash of children
- To verify any block, only need O(log N) hashes (not entire dataset)

**Used by:**
- **Cassandra** — anti-entropy repair (detect out-of-sync replicas)
- **HDFS** — data integrity checks
- **Git** — commit tree integrity
- **Blockchain** — transaction verification
- **BitTorrent** — piece verification

---

## Distributed File Systems

### HDFS (Hadoop Distributed File System)

```mermaid
graph TD
    Client[Client] --> NN[NameNode<br/>Metadata: file → blocks → locations]
    NN --> DN1[DataNode 1]
    NN --> DN2[DataNode 2]
    NN --> DN3[DataNode 3]

    DN1 -->|"Block A (replica)"| DN2
    DN2 -->|"Block A (replica)"| DN3
```

| Feature | Detail |
|---------|--------|
| **Block size** | 128MB (vs 4KB in regular FS) — optimized for large files |
| **Replication** | Default 3 replicas across racks |
| **NameNode** | Single master for metadata (SPOF → standby NameNode) |
| **DataNode** | Stores actual data blocks |
| **Best for** | Batch processing (MapReduce, Spark), data lakes |
| **Not for** | Low-latency random reads, small files |

---

## Object Storage (S3-style)

```mermaid
graph LR
    Client -->|"PUT /bucket/key"| API[S3 API Gateway]
    API --> Meta[(Metadata Store<br/>key → location)]
    API --> Storage[(Distributed<br/>Block Storage)]

    Client -->|"GET /bucket/key"| API
```

### S3 Key Concepts

| Concept | Detail |
|---------|--------|
| **Bucket** | Top-level container (like a namespace) |
| **Key** | Object identifier (can include `/` for folder-like structure) |
| **Versioning** | Keep all versions of an object |
| **Storage classes** | Standard → Infrequent Access → Glacier (cheaper, slower) |
| **Lifecycle rules** | Auto-transition objects between storage classes |
| **Consistency** | Strong read-after-write (since 2020) |
| **Durability** | 99.999999999% (11 nines) |

### When to Use Object Storage
- Media files (images, videos, audio)
- Backups and archives
- Log storage and data lakes
- Static website hosting
- ML training data

---

## Common Interview Questions

1. **"Block vs object storage?"** → Block for databases (low-latency, byte-level access). Object for media/backups (cheap, scalable, REST API).
2. **"How does a database ensure durability?"** → Write-Ahead Log: write to sequential log first, flush to data files asynchronously. On crash, replay WAL.
3. **"LSM tree vs B+ tree?"** → LSM: optimized for writes (Cassandra, RocksDB). B+: optimized for reads (PostgreSQL, MySQL).
4. **"How do distributed systems verify data integrity?"** → Merkle trees: compare root hashes. If different, traverse tree to find divergent blocks. O(log N) instead of comparing everything.
5. **"How would you store petabytes of data?"** → Object storage (S3) for raw data, HDFS for processing, tiered storage classes for cost optimization.

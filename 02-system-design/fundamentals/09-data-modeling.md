# Data Modeling

How to structure and organize data for your system — the bridge between requirements and database design.

---

## Why Data Modeling Matters

- **Shapes your queries** — the data model determines what's easy vs hard to query
- **Drives performance** — good models avoid expensive joins and full scans
- **Affects scalability** — normalized vs denormalized has massive scaling implications
- **Guides schema evolution** — well-modeled data is easier to extend

---

## Normalization vs Denormalization

### Normalization (Eliminate redundancy)

```mermaid
erDiagram
    USERS ||--o{ ORDERS : places
    ORDERS ||--|{ ORDER_ITEMS : contains
    ORDER_ITEMS }o--|| PRODUCTS : references

    USERS {
        int id PK
        string name
        string email
    }
    ORDERS {
        int id PK
        int user_id FK
        datetime created_at
    }
    ORDER_ITEMS {
        int id PK
        int order_id FK
        int product_id FK
        int quantity
    }
    PRODUCTS {
        int id PK
        string name
        decimal price
    }
```

### Normal Forms (Quick Reference)

| Form | Rule | Example Violation |
|------|------|-------------------|
| **1NF** | Atomic values, no repeating groups | `tags = "java,python"` in one column |
| **2NF** | 1NF + no partial dependencies | Non-key depends on part of composite key |
| **3NF** | 2NF + no transitive dependencies | `zip → city` stored in user table |

### When to Normalize vs Denormalize

| Factor | Normalize | Denormalize |
|--------|-----------|-------------|
| **Reads** | Slower (joins needed) | Faster (data co-located) |
| **Writes** | Faster (update one place) | Slower (update multiple copies) |
| **Consistency** | Easy (single source of truth) | Hard (must sync copies) |
| **Storage** | Efficient (no duplication) | More storage |
| **Best for** | Write-heavy, OLTP, consistency | Read-heavy, OLAP, scale |

**Interview rule of thumb:** Start normalized. Denormalize strategically for performance when you can identify the hot read paths.

---

## Data Modeling for NoSQL

NoSQL requires **query-driven modeling** — design your data model around your access patterns, not around entities.

### SQL vs NoSQL Modeling Mindset

| SQL | NoSQL |
|-----|-------|
| "What data do I have?" | "What queries do I need?" |
| Normalize, join at query time | Denormalize, pre-join at write time |
| One model serves many query patterns | Different tables/collections for different queries |
| Schema first, queries later | Queries first, schema follows |

### DynamoDB Single-Table Design Example

Instead of separate tables, use one table with composite keys:

| PK | SK | Data |
|----|-----|------|
| `USER#123` | `PROFILE` | `{name, email, ...}` |
| `USER#123` | `ORDER#001` | `{total, status, ...}` |
| `USER#123` | `ORDER#002` | `{total, status, ...}` |
| `ORDER#001` | `ITEM#1` | `{product, qty, ...}` |

- **Get user profile:** `PK = USER#123, SK = PROFILE`
- **Get user's orders:** `PK = USER#123, SK begins_with ORDER#`
- **Get order items:** `PK = ORDER#001, SK begins_with ITEM#`

---

## Entity Relationships

```mermaid
graph TD
    subgraph Relationship Types
        A["One-to-One (1:1)"] -->|Example| A1["User → Profile"]
        B["One-to-Many (1:N)"] -->|Example| B1["User → Orders"]
        C["Many-to-Many (M:N)"] -->|Example| C1["Students ↔ Courses"]
    end
```

### Modeling Strategies

| Relationship | SQL | NoSQL (Document) | NoSQL (Key-Value) |
|-------------|-----|------|------|
| **1:1** | Same table or FK | Embedded subdocument | Single key |
| **1:N** | FK on child table | Embedded array (if bounded) or reference | Composite key |
| **M:N** | Junction table | Array of references + denormalize | GSI / inverted index |

### Embedding vs Referencing (Document DBs)

| Factor | Embed | Reference |
|--------|-------|-----------|
| Read together? | ✅ Embed | ❌ Reference |
| Bounded size? | ✅ Embed | ❌ Reference (unbounded) |
| Updates? | Infrequent → Embed | Frequent → Reference |
| Size limit? | < 16MB (MongoDB) | No limit |

**Rule:** Embed what you read together. Reference what changes independently or grows unboundedly.

---

## Schema Evolution

### SQL Schema Changes
```sql
-- Adding a column (backward compatible)
ALTER TABLE users ADD COLUMN phone VARCHAR(20) DEFAULT NULL;

-- Renaming (breaking if code references old name)
ALTER TABLE users RENAME COLUMN name TO full_name;

-- Expand-contract pattern for zero-downtime
-- 1. Add new column
-- 2. Backfill data
-- 3. Update code to write to both columns
-- 4. Update code to read from new column
-- 5. Drop old column
```

### NoSQL Schema Changes
- Document DBs handle schema evolution naturally (just add fields)
- Old documents without new fields → handle with defaults in application code
- Use schema versioning: `{ "_schemaVersion": 2, ... }`

---

## Time-Series Data Modeling

| Approach | How | Example |
|----------|-----|---------|
| **Wide row** | One row per entity, columns per timestamp | Cassandra: `sensor_id → {t1: v1, t2: v2}` |
| **Narrow row** | One row per data point | `(sensor_id, timestamp) → value` |
| **Bucketing** | Group by time window | `(sensor_id, hour_bucket) → [data points]` |

**Best practices:**
- Partition by entity + time bucket (avoid hot partitions)
- Use TTL for automatic data expiration
- Pre-aggregate for dashboards (1-min, 5-min, 1-hour rollups)

---

## Data Modeling Patterns

| Pattern | When | How |
|---------|------|-----|
| **Materialized view** | Pre-compute expensive queries | Store result of complex query as a table, refresh periodically |
| **Event sourcing** | Need full audit trail | Store events, derive current state by replaying |
| **CQRS** | Read/write models differ significantly | Separate read and write databases |
| **Polymorphic** | Multiple entity types in one table | Discriminator column + nullable fields |
| **Soft delete** | Need to "undo" deletes | `deleted_at` timestamp instead of DELETE |
| **Audit trail** | Track all changes | Separate history table or event log |

---

---

## Distributed ID Generation

When sharding or using distributed databases, auto-increment IDs don't work (collisions, single point of failure).

| Strategy | Example | Pros | Cons |
|----------|---------|------|------|
| **UUID (v4)** | `f47ac10b-58cc...` | No coordination needed, collision-proof | Large (128-bit), unindexable (random), fragmentation |
| **UUID (v7)** | `017F22E2...` | Time-ordered (sortable) | Still large (128-bit) |
| **Snowflake ID** | Twitter / Discord | Time-ordered, 64-bit (fits in bigint), scalable | Requires custom infrastructure (ZooKeeper/Etcd) |
| **Ticket Server** | Flickr approach | Simple auto-increment (central DB) | Single point of failure, added latency |

### Snowflake ID Anatomy (64-bit)

```mermaid
graph LR
    Sign[1 bit<br/>Unused] --> Time[41 bits<br/>Timestamp (ms)]
    Time --> Machine[10 bits<br/>Machine ID]
    Machine --> Seq[12 bits<br/>Sequence #]
```

**Recommendation:** Use **UUID v7** (if DB supports it) or **Snowflake ID** (if you need 64-bit integers). Avoid UUID v4 for primary keys in B-Tree/B+Tree databases (MySQL/Postgres) due to index fragmentation.

---

## Sharding & Partitioning Strategies

How to split data across multiple nodes when it creates too much for one server.

### Partitioning Criteria

| Strategy | Description | Best For | Issues |
|----------|-------------|----------|--------|
| **Key-Based (Hash)** | `hash(id) % N` | Distributing write load evenly | Resharding is expensive (moves 1/N keys) |
| **Range-Based** | `A-M` node 1, `N-Z` node 2 | Range queries (get users created in Jan) | Hot spots (e.g., all recent data on one node) |
| **Directory-Based** | Lookup table `key → shard_id` | Full control over placement | Lookup table becomes bottleneck/SPoF |

### Choosing a Shard Key (Partition Key)

The most critical decision in distributed schema design.

1.  **Cardinality:** Must be high (many unique values).
    *   *Bad:* `status` (only 5 values = only 5 shards).
    *   *Good:* `user_id`, `device_id`.
2.  **Access Pattern:** Should match your most frequent queries to avoid **Scatter-Gather**.
    *   *Query:* "Get all orders for user 123" → Shard by `user_id` (all orders on one shard).
    *   *Query:* "Get all orders from today" → Sharding by `user_id` forces query to *all* shards (slow).
3.  **Write Distribution:** Avoid "hot keys" where one shard takes all traffic.
    *   *Bad:* `created_at` (all new writes hit the "today" shard).

---

## Hierarchical Data Patterns (Trees & Graphs)

Storing trees (org charts, comment threads, folders) in SQL.

| Pattern | Schema | Query Child | Query Subtree | Move Subtree |
|---------|--------|-------------|---------------|--------------|
| **Adjacency List** | `parent_id` column | Easy (`WHERE parent_id = X`) | Hard (Requires recursive CTEs) | Easy (Update 1 row) |
| **Path Enumeration** | `path` column (`1/5/12`) | Easy (`LIKE '1/5/%'`) | Easy | Hard (Update all children paths) |
| **Nested Sets** | `left` / `right` values | Hard | Very Fast (`l > P.left AND r < P.right`) | Very Hard (Recalculate all L/R) |
| **Closure Table** | Separate `(ancestor, descendant)` table | Easy | Easy (Join closure table) | Hard (Many rows to update) |

**Recommendation:**
*   Use **Adjacency List** with Recursive CTEs (Common Table Expressions) for most standard apps (Postgres/MySQL 8.0 support this).
*   Use **Closure Table** for deep hierarchies with frequent subtree queries but infrequent writes.

---

## Hybrid Modeling (JSON in SQL)

Modern SQL databases (Postgres JSONB, MySQL JSON) allow schema-less data (NoSQL) alongside relational data.

### When to use JSON columns?

✅ **Good Use Cases:**
*   **Dynamic Attributes:** E-commerce products (T-shirts have `size`, Laptops have `cpu_speed`).
*   **External Data:** Storing responses from 3rd party APIs (webhooks).
*   **Sparse Data:** User settings/configs where most columns would be NULL.

❌ **Bad Use Cases:**
*   **Foreign Keys:** Don't store `{"author_id": 123}` inside JSON. Relational integrity breaks.
*   **Search Targets:** If you frequently `WHERE` or `GROUP BY` a field, pull it out to a dedicated column.
*   **High Update Frequency:** Updating one field in JSON usually rewrites the whole JSON blob (write amplification).

---

## Common Interview Questions

1.  **"How would you model the data for X?"** → Identify entities, relationships, access patterns. Start normalized. Denormalize for hot read paths. Choose Sharding Key early if scale is mentioned.
2.  **"SQL or NoSQL for this?"** → SQL if complex queries/transactions/strict consistency. NoSQL if need to scale writes, flexible schema, or specific access patterns (key-value, time-series).
3.  **"How do you handle schema changes without downtime?"** → Expand-contract pattern: add new column → backfill → migrate code → drop old column.
4.  **"How do you model many-to-many in NoSQL?"** → Denormalize: store references in both sides + use GSI/secondary index for reverse lookups.
5.  **"How do you model comments (threaded)?"** → Adjacency list (parent_id) for simple depth. Path enumeration (`/1/4/9`) for complex sorting. Recursive CTEs for retrieval.
6.  **"How do you generate unique IDs at scale?"** → Snowflake ID (time-sortable, 64-bit) or UUID v7. Explain why Auto-Increment fails (sharding) and UUID v4 is bad for DB indexing (fragmentation).
7.  **"How do you shard a multi-tenant SaaS app?"** → Shard by `tenant_id` (customer_id). Keeps all customer data together. Easy to backup/restore per customer. Risk: "Whale" tenant creates a hot shard.


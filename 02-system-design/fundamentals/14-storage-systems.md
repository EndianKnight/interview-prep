# Storage & File Systems

> TODO: Detailed guide with examples, diagrams, and interview questions

## Topics to Cover
- Block storage vs object storage vs file storage
- HDFS architecture
- **LSM Tree** — write-optimized, compaction, used in Cassandra/LevelDB/RocksDB
- **B+ Tree** — read-optimized, used in MySQL/PostgreSQL, comparison with LSM
- **Write-Ahead Log (WAL)** — crash recovery, durability guarantees
- **Merkle Tree** — data integrity verification, anti-entropy repair, used in Dynamo/Cassandra
- SSTable and memtable
- Compaction strategies (size-tiered vs leveled)
- Column-oriented vs row-oriented storage
- Data compression techniques

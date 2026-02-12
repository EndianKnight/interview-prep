# Locking & Synchronization

> TODO: Detailed guide with interview-relevant examples

## Topics to Cover
- **Mutex / Lock** — mutual exclusion, reentrant locks
- **Read-write locks** — shared readers, exclusive writers
- **Semaphores** — counting semaphore, binary semaphore
- **Condition variables** — wait/notify, spurious wakeups
- **Deadlock** — conditions (Coffman), detection, prevention, avoidance
- **Database locks** — row-level, table-level, optimistic vs pessimistic locking, SELECT FOR UPDATE
- **Distributed locks** — Redis (Redlock), ZooKeeper, etcd, fencing tokens
- **Compare-and-swap (CAS)** — lock-free programming, atomic operations
- Spin locks — when to use, CPU considerations
- Lease-based locking and TTL

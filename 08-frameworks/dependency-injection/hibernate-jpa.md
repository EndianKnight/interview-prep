# Hibernate & JPA Internals

> TODO: Interview-focused guide — ORM internals beyond basic usage

## Topics to Cover
- **First-level cache** (Session cache) — per-transaction, automatic, dirty checking
- **Second-level cache** — shared across sessions, providers (EhCache, Redis), `@Cacheable`
- **Query cache** — caching query results, invalidation gotchas
- **Dirty checking** — how Hibernate detects changes, flush modes
- **Batch processing** — `hibernate.jdbc.batch_size`, StatelessSession for bulk ops
- **Inheritance strategies** — `SINGLE_TABLE`, `JOINED`, `TABLE_PER_CLASS` — tradeoffs
- **Optimistic locking** — `@Version`, handling `OptimisticLockException`
- **Pessimistic locking** — `LockModeType.PESSIMISTIC_WRITE`, `SELECT FOR UPDATE`
- **Connection pooling** — HikariCP configuration, sizing
- **Common pitfalls** — N+1, cartesian product with multiple bags, entity equality (equals/hashCode)

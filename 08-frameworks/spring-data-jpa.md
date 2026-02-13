# Spring Data JPA

> TODO: Interview-focused guide — ORM concepts and Spring's data access layer

## Topics to Cover
- **JPA basics** — Entity, `@Id`, `@GeneratedValue`, `@Column`, `@Table`
- **Relationships** — `@OneToMany`, `@ManyToOne`, `@ManyToMany`, `@OneToOne`, cascade types, orphan removal
- **Repository pattern** — `JpaRepository`, `CrudRepository`, derived query methods (`findByNameAndAge`)
- **Custom queries** — `@Query` (JPQL + native), Specifications, QueryDSL
- **N+1 problem** — what it is, how to detect, fix with `@EntityGraph`, `JOIN FETCH`
- **Lazy vs eager loading** — `FetchType.LAZY` vs `EAGER`, LazyInitializationException
- **Transactions** — `@Transactional`, propagation levels, isolation levels, read-only optimization
- **Entity lifecycle** — managed, detached, transient, removed states
- **Auditing** — `@CreatedDate`, `@LastModifiedDate`, `@CreatedBy`
- **Pagination & sorting** — `Pageable`, `Sort`, `Page` vs `Slice`
- **Database migrations** — Flyway vs Liquibase integration

# Immutability & Defensive Programming

> TODO: Detailed guide with examples in Java/C++/Python

## Topics to Cover

### Immutable Objects
- What makes an object immutable — all fields final, no setters, no leaking mutable internals
- Java: `final` fields, `String`, `Integer`, `Collections.unmodifiableList()`
- C++: `const`, `constexpr`
- Python: `tuple`, `frozenset`, `@dataclass(frozen=True)`
- Benefits: thread safety, hashability, simpler reasoning

### Value Objects vs Entities
- Value objects — defined by attributes, no identity (Money, Address, DateRange)
- Entities — defined by identity, mutable state (User, Order)
- Java records (16+) — immutable value types with auto-generated equals/hashCode
- When to use which in LLD

### Defensive Copying
- Returning copies of mutable internal state
- Copying mutable constructor arguments
- Unmodifiable wrappers vs deep copies
- Performance tradeoffs

### Defensive Programming Techniques
- Input validation — null checks, range checks, format validation
- Assertions — precondition checks in development
- Fail-fast on invalid state
- Making invalid states unrepresentable (type-driven design)

### Thread Safety Through Immutability
- Immutable objects are inherently thread-safe
- Avoiding synchronization overhead
- Persistent data structures — structural sharing

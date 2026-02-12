# Generics & Type Safety

> TODO: Detailed guide with examples in Java/C++/Python

## Topics to Cover

### Why Generics
- Type-safe containers without casting
- Code reuse with type parameters
- Compile-time error detection vs runtime ClassCastException

### Java Generics
- Generic classes: `class Box<T> { T value; }`
- Generic methods: `<T> T max(T a, T b)`
- Bounded types: `<T extends Comparable<T>>`
- Wildcards: `?`, `? extends T` (upper bound), `? super T` (lower bound)
- **PECS** — Producer Extends, Consumer Super
- Type erasure — generics are compile-time only, no reified types
- Diamond operator `<>`, raw types (avoid)

### C++ Templates
- Function templates and class templates
- Template specialization
- SFINAE and concepts (C++20)
- Templates vs Java generics — code generation vs erasure

### Python Typing
- `typing.Generic[T]`, `TypeVar`
- Type hints for containers: `List[int]`, `Dict[str, Any]`
- Protocol (structural subtyping, duck typing with types)
- Runtime vs static checking (mypy)

### Variance
- **Covariance** — subtype relationship preserved (`List<Dog>` → `List<Animal>`?)
- **Contravariance** — subtype relationship reversed
- **Invariance** — no relationship
- Arrays vs generics variance (Java arrays are covariant, generics are invariant)

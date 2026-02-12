# Enums, Constants & Type Modeling

> TODO: Detailed guide with examples in Java/C++/Python

## Topics to Cover

### Enums
- **Java enums** — enum with fields, methods, abstract methods, implementing interfaces
- **C++ enums** — `enum class` (scoped), underlying types
- **Python enums** — `enum.Enum`, `IntEnum`, `auto()`
- Enum as state machine — `OrderStatus.PENDING → CONFIRMED → SHIPPED → DELIVERED`
- Enum with behavior — each constant overrides a method (strategy via enum)

### Constants & Magic Values
- Why magic numbers/strings are bad
- Constant classes vs enums
- Configuration vs compile-time constants
- Java: `static final`, C++: `constexpr`, Python: module-level UPPER_CASE

### Advanced Type Modeling
- **Sealed classes/interfaces (Java 17+)** — restricted class hierarchies
- **Algebraic data types** — sum types (sealed) + product types (records)
- **Pattern matching** — `switch` with sealed types, exhaustiveness checking
- Type-safe IDs — `UserId` vs `OrderId` instead of raw `long`
- Making invalid states unrepresentable through types

### Interview Application
- Using enums for states, types, strategies in LLD problems
- Type-safe design decisions to discuss during walkthroughs
- When to use enum vs class hierarchy vs strategy pattern

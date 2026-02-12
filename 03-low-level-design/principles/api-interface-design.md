# API & Interface Design

> TODO: Detailed guide with examples in Java/C++/Python

## Topics to Cover

### Method Design
- Clear naming — verb + noun (getUser, calculateTotal)
- Parameter count — keep ≤ 3, use parameter objects for more
- Return types — avoid returning null, prefer Optional or empty collections
- Method length — single level of abstraction per method

### Interface Design Principles
- **Program to interfaces, not implementations**
- Thin interfaces (ISP) — role-based, not entity-based
- Default methods (Java 8+) — evolving interfaces without breaking clients
- Marker interfaces vs annotations

### Fluent Interfaces & Builder
- Method chaining — `builder.setName("x").setAge(25).build()`
- When fluent is good (configuration, queries) vs misleading
- Builder pattern for complex construction

### Law of Demeter (Principle of Least Knowledge)
- Only talk to immediate friends — no `a.getB().getC().doThing()`
- Train wreck anti-pattern
- Solutions: delegate methods, tell-don't-ask

### Design by Contract
- Preconditions — what callers must ensure (assert/validate inputs)
- Postconditions — what the method guarantees
- Invariants — what must always be true about the object
- Defensive programming vs contract-based trust

### Versioning & Evolution
- Backward compatibility — adding methods is safe, changing signatures is not
- Deprecated methods — @Deprecated with migration path
- Sealed interfaces (Java 17+) — controlled implementations

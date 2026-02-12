# SOLID Principles

> TODO: Detailed guide with examples in Java/C++/Python

## Topics to Cover

### S — Single Responsibility Principle
- A class should have only one reason to change
- Violation: class handles both validation AND persistence
- Fix: split into focused classes

### O — Open/Closed Principle
- Open for extension, closed for modification
- Achieve via polymorphism, strategy pattern, dependency injection
- Violation: growing switch/if-else chains for new types

### L — Liskov Substitution Principle
- Subtypes must be substitutable for base types
- Classic violation: Square extending Rectangle
- Contract rules: preconditions, postconditions, invariants

### I — Interface Segregation Principle
- No client should depend on methods it doesn't use
- Fat interfaces vs thin, role-specific interfaces
- Violation: `Worker` interface with `work()`, `eat()`, `sleep()`

### D — Dependency Inversion Principle
- High-level modules depend on abstractions, not concrete implementations
- Dependency injection — constructor, setter, interface injection
- Inversion of Control (IoC) containers

# Composition vs Inheritance

> TODO: Detailed guide with examples in Java/C++/Python

## Topics to Cover

### Inheritance (is-a)
- When to use: true subtype relationship (Dog is-a Animal)
- Advantages: code reuse, polymorphism via overriding
- Problems: fragile base class, tight coupling, rigid hierarchies, diamond problem

### Composition (has-a)
- When to use: object needs behavior from another object
- Delegation pattern — forwarding calls to composed objects
- Advantages: flexible, swappable at runtime, avoids deep hierarchies

### "Favor Composition Over Inheritance"
- Why GoF recommends this
- Composition through interfaces — Strategy, Decorator, Bridge patterns
- Real-world refactoring: converting inheritance to composition

### Decision Framework
- Use inheritance when: genuine is-a, Liskov substitution holds
- Use composition when: has-a, need runtime flexibility, avoiding deep hierarchies
- Hybrid: inherit interface (implements), compose behavior (delegation)

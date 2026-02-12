# Coupling & Cohesion

> TODO: Detailed guide with examples in Java/C++/Python

## Topics to Cover

### Coupling (aim for LOW)
- **Tight coupling** — depends on concrete implementations, internal details
- **Loose coupling** — depends on interfaces/abstractions
- Types (worst → best): content → common → control → stamp → data → message
- Techniques to reduce: dependency injection, interfaces, observer, mediator

### Cohesion (aim for HIGH)
- **High cohesion** — focused, well-defined responsibility
- **Low cohesion** — grab-bag of unrelated functionality (God class)
- Types (worst → best): coincidental → logical → temporal → sequential → functional
- Signs of low cohesion: class name has "Manager", "Helper", "Utils" with 20+ methods

### Measuring Quality
- How coupling and cohesion relate to SOLID
- Package/module level cohesion
- Microservices: each service = high cohesion, loose coupling between services

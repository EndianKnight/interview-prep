# Dependency Injection

> TODO: Detailed guide with examples in Java/C++/Python

## Topics to Cover

### What & Why
- Inversion of Control (IoC) — don't create your dependencies, receive them
- Tight coupling without DI vs loose coupling with DI
- Benefits: testability, flexibility, single responsibility

### Injection Types
- **Constructor injection** — preferred, immutable, all dependencies upfront
- **Setter injection** — optional dependencies, mutable
- **Interface injection** — inject via interface contract
- **Method injection** — pass dependency per method call

### DI Without Frameworks (Manual DI)
- Wiring dependencies in main/composition root
- Factory pattern for complex object creation
- When manual DI is sufficient (small projects, interviews)

### DI Frameworks
- **Spring (Java)** — @Autowired, @Component, @Bean, application context
- **Guice (Java)** — @Inject, modules, bindings
- **C++** — no standard framework, manual DI or Boost.DI
- **Python** — dependency-injector library, or just constructor params

### Service Locator (Anti-Pattern)
- What it is — global registry for dependencies
- Why it's considered an anti-pattern vs DI
- Hidden dependencies, hard to test

### Interview Application
- How DI enables mocking in unit tests
- DI in LLD problems — injecting strategies, repositories, notification services
- Explaining DI to interviewer in class diagram context

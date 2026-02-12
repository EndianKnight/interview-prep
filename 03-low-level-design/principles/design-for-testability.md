# Design for Testability

> TODO: Detailed guide with examples in Java/C++/Python

## Topics to Cover

### What Makes Code Testable
- Dependencies are injectable (not `new`'d inside methods)
- No hidden global/static state
- Single responsibility — small, focused units to test
- Pure functions where possible — same input → same output
- Seams — points where behavior can be substituted for testing

### Test Doubles
- **Mock** — verifies interactions (was method called with these args?)
- **Stub** — returns predetermined responses
- **Spy** — records calls for later verification
- **Fake** — simplified working implementation (in-memory DB)
- **Dummy** — placeholder, never actually used
- Mocking frameworks: Mockito (Java), Google Mock (C++), unittest.mock (Python)

### Patterns That Help Testability
- **Dependency injection** — swap real services with mocks
- **Strategy pattern** — inject different algorithms for testing
- **Repository pattern** — abstract data access, test with in-memory repo
- **Interface-based design** — mock any dependency behind an interface

### Anti-Patterns That Hurt Testability
- Singleton (hard to mock, shared state across tests)
- Static methods (can't override/mock easily)
- Deep inheritance (hard to instantiate in isolation)
- Law of Demeter violations (need to mock entire chain)
- Constructors that do work (side effects during setup)

### Test Organization
- Arrange-Act-Assert (AAA) pattern
- Test naming: `should_returnEmpty_when_noItemsExist`
- One assertion per test vs behavior-focused tests
- Test isolation — no shared mutable state between tests

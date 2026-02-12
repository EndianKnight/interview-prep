# Error Handling Patterns

> TODO: Detailed guide with examples in Java/C++/Python

## Topics to Cover

### Exception Types
- Checked vs unchecked exceptions (Java) — when to use each
- Standard exceptions: IllegalArgumentException, NullPointerException, IOException
- Custom exception hierarchies — base exception per module/domain
- C++ exceptions — std::exception hierarchy, noexcept

### Error Handling Strategies
- **Exceptions** — throw on failure, catch at appropriate level
- **Error codes / return values** — C-style, Go-style (value, error)
- **Result/Either types** — Rust Result<T,E>, functional error handling
- **Optional/Nullable** — Optional (Java), std::optional (C++), None (Python)
- Choosing between strategies for different layers (API, service, data)

### Patterns
- **Fail-fast** — detect errors early, throw immediately
- **Fail-safe** — degrade gracefully, provide defaults
- **Retry with backoff** — transient failures, exponential backoff, jitter
- **Null Object pattern** — avoid null checks with no-op implementations
- **Try-with-resources / RAII** — automatic cleanup (Java try-with-resources, C++ RAII, Python context managers)
- **Global exception handler** — catch-all for unhandled errors

### Anti-Patterns
- Swallowing exceptions (empty catch blocks)
- Catching generic Exception/Throwable
- Using exceptions for control flow
- Returning null instead of empty collections
- Exception in constructors — partially constructed objects

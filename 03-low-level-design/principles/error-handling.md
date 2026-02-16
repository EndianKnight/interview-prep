# Error Handling Patterns

Robust error handling separates production-quality systems from fragile prototypes; mastering these patterns is essential for senior engineering interviews.

---

## 1. Exception Types

### Checked vs Unchecked Exceptions (Java)

| Aspect | Checked | Unchecked |
|--------|---------|-----------|
| Inherits from | `Exception` | `RuntimeException` |
| Compiler enforces | Yes (`throws` clause required) | No |
| When to use | Recoverable conditions (I/O, network) | Programming errors (null deref, bad args) |
| Examples | `IOException`, `SQLException` | `NullPointerException`, `IllegalArgumentException` |
| Caller obligation | Must catch or propagate | Optional to catch |

**Java**
```java
// Checked — caller MUST handle
public byte[] readFile(String path) throws IOException {
    return Files.readAllBytes(Path.of(path));
}

// Unchecked — programming error, fail fast
public void setAge(int age) {
    if (age < 0) throw new IllegalArgumentException("Age cannot be negative: " + age);
    this.age = age;
}
```

### Standard Exception Hierarchy (Java)

```
Throwable
├── Error                          // JVM-level, don't catch (OutOfMemoryError)
└── Exception                      // Checked
    ├── IOException
    ├── SQLException
    └── RuntimeException           // Unchecked
        ├── NullPointerException
        ├── IllegalArgumentException
        ├── IllegalStateException
        ├── UnsupportedOperationException
        └── IndexOutOfBoundsException
```

**Rule of thumb:** throw `IllegalArgumentException` for bad inputs, `IllegalStateException` for bad object state, `UnsupportedOperationException` for unimplemented methods.

### Custom Exception Hierarchies

Define a base exception per module or bounded context, then subclass for specific failures. This lets callers catch broadly or narrowly.

**Java**
```java
// Base exception for the payment domain
public class PaymentException extends RuntimeException {
    private final String errorCode;

    public PaymentException(String errorCode, String message) {
        super(message);
        this.errorCode = errorCode;
    }

    public PaymentException(String errorCode, String message, Throwable cause) {
        super(message, cause);
        this.errorCode = errorCode;
    }

    public String getErrorCode() { return errorCode; }
}

// Specific subtypes
public class InsufficientFundsException extends PaymentException {
    public InsufficientFundsException(double requested, double available) {
        super("PAY_INSUFFICIENT_FUNDS",
              String.format("Requested %.2f but only %.2f available", requested, available));
    }
}

public class PaymentGatewayException extends PaymentException {
    public PaymentGatewayException(String gateway, Throwable cause) {
        super("PAY_GATEWAY_ERROR", "Gateway " + gateway + " failed", cause);
    }
}
```

**C++**
```cpp
#include <stdexcept>
#include <string>

class PaymentException : public std::runtime_error {
    std::string error_code_;
public:
    PaymentException(std::string code, const std::string& msg)
        : std::runtime_error(msg), error_code_(std::move(code)) {}
    const std::string& error_code() const noexcept { return error_code_; }
};

class InsufficientFundsException : public PaymentException {
public:
    InsufficientFundsException(double requested, double available)
        : PaymentException("PAY_INSUFFICIENT_FUNDS",
            "Requested " + std::to_string(requested) +
            " but only " + std::to_string(available) + " available") {}
};
```

**Python**
```python
class PaymentException(Exception):
    def __init__(self, error_code: str, message: str):
        super().__init__(message)
        self.error_code = error_code

class InsufficientFundsException(PaymentException):
    def __init__(self, requested: float, available: float):
        super().__init__(
            "PAY_INSUFFICIENT_FUNDS",
            f"Requested {requested:.2f} but only {available:.2f} available"
        )

class PaymentGatewayException(PaymentException):
    def __init__(self, gateway: str, cause: Exception | None = None):
        super().__init__("PAY_GATEWAY_ERROR", f"Gateway {gateway} failed")
        self.__cause__ = cause
```

### C++ Exception Specifics

```
std::exception
├── std::logic_error               // Programming errors
│   ├── std::invalid_argument
│   ├── std::out_of_range
│   └── std::domain_error
└── std::runtime_error             // External failures
    ├── std::overflow_error
    ├── std::underflow_error
    └── std::system_error
```

**`noexcept` — the contract that a function will not throw:**

```cpp
// noexcept enables move optimizations and is critical for
// move constructors, swap, and destructors
void swap(MyClass& other) noexcept {
    std::swap(data_, other.data_);
}

// Conditional noexcept — propagate from inner types
template <typename T>
void wrapper(T& a, T& b) noexcept(noexcept(a.swap(b))) {
    a.swap(b);
}

// If a noexcept function throws, std::terminate() is called — no unwinding
```

**Key rule:** destructors are implicitly `noexcept`. Never throw from a destructor.

---

## 2. Error Handling Strategies

### Strategy Comparison

| Strategy | Best For | Pros | Cons |
|----------|----------|------|------|
| Exceptions (throw/catch) | Unexpected failures, Java/C++/Python | Clean happy path, stack trace | Performance cost, invisible control flow |
| Error codes / return values | C, Go, performance-critical code | Explicit, no runtime cost | Easy to ignore, clutters return types |
| Result/Either types | Rust, functional programming | Type-safe, composable, compiler-enforced | Verbose without language support |
| Optional/Nullable | Absent values (not errors) | Eliminates null checks | No error information |

### Exceptions (throw/catch)

The dominant strategy in Java, C++, and Python. Throw at the point of failure; catch at the level that can meaningfully recover.

**Java**
```java
public User findUser(String id) {
    User user = repository.findById(id);
    if (user == null) {
        throw new UserNotFoundException(id);  // throw at detection point
    }
    return user;
}

// Catch at the layer that can respond (e.g., controller)
@GetMapping("/users/{id}")
public ResponseEntity<User> getUser(@PathVariable String id) {
    try {
        return ResponseEntity.ok(userService.findUser(id));
    } catch (UserNotFoundException e) {
        return ResponseEntity.notFound().build();
    }
    // Don't catch generic Exception here — let it propagate to global handler
}
```

### Error Codes / Return Values (Go-Style)

Go returns `(value, error)` tuples. You can simulate this pattern in other languages.

**Java** (simulated with a sealed result type)
```java
public record Result<T>(T value, String error) {
    public static <T> Result<T> ok(T value) { return new Result<>(value, null); }
    public static <T> Result<T> fail(String error) { return new Result<>(null, error); }
    public boolean isOk() { return error == null; }
}

// Usage
Result<User> result = findUser("123");
if (!result.isOk()) {
    log.error(result.error());
    return;
}
User user = result.value();
```

**C++**
```cpp
#include <expected>  // C++23

std::expected<User, std::string> find_user(const std::string& id) {
    auto user = repository.find_by_id(id);
    if (!user) {
        return std::unexpected("User not found: " + id);
    }
    return *user;
}

// Usage
auto result = find_user("123");
if (!result) {
    std::cerr << result.error() << "\n";
    return;
}
User user = result.value();
```

**Python**
```python
from typing import TypeVar, Generic
from dataclasses import dataclass

T = TypeVar("T")

@dataclass
class Result(Generic[T]):
    value: T | None = None
    error: str | None = None

    @staticmethod
    def ok(value: T) -> "Result[T]":
        return Result(value=value)

    @staticmethod
    def fail(error: str) -> "Result[T]":
        return Result(error=error)

    @property
    def is_ok(self) -> bool:
        return self.error is None

# Usage
result = find_user("123")
if not result.is_ok:
    logger.error(result.error)
    return
user = result.value
```

### Optional / Nullable

Use `Optional` when the absence of a value is a **normal outcome**, not an error.

**Java**
```java
public Optional<User> findUser(String id) {
    return Optional.ofNullable(repository.findById(id));
}

// Caller — forced to handle the absent case
String name = findUser("123")
    .map(User::getName)
    .orElse("Unknown");
```

**C++**
```cpp
#include <optional>

std::optional<User> find_user(const std::string& id) {
    auto it = users_.find(id);
    if (it == users_.end()) return std::nullopt;
    return it->second;
}

// Caller
auto user = find_user("123");
std::string name = user.has_value() ? user->name() : "Unknown";
```

**Python**
```python
from typing import Optional

def find_user(user_id: str) -> Optional[User]:
    return user_repo.get(user_id)  # Returns None if absent

# Caller
user = find_user("123")
name = user.name if user else "Unknown"
```

### Choosing Strategies by Layer

| Layer | Recommended Strategy | Rationale |
|-------|---------------------|-----------|
| Domain / Business logic | Exceptions (custom hierarchy) | Clean business rules, no error plumbing |
| Repository / Data access | Exceptions or Optional | Optional for "not found", exceptions for failures |
| Service / Use case | Exceptions — translate and wrap | Translate low-level to domain exceptions |
| Controller / API boundary | Catch and convert to HTTP/gRPC codes | Global handler maps exception to response |
| Infrastructure / External calls | Result types or exceptions + retry | Need explicit failure handling |

---

## 3. Patterns

### Fail-Fast

Detect errors at the earliest possible point and throw immediately. Prevents corrupted state from propagating.

**Java**
```java
public class Order {
    private final List<LineItem> items;
    private final String customerId;

    public Order(String customerId, List<LineItem> items) {
        // Validate everything upfront — fail fast
        Objects.requireNonNull(customerId, "customerId must not be null");
        if (customerId.isBlank()) {
            throw new IllegalArgumentException("customerId must not be blank");
        }
        Objects.requireNonNull(items, "items must not be null");
        if (items.isEmpty()) {
            throw new IllegalArgumentException("Order must have at least one item");
        }
        this.customerId = customerId;
        this.items = List.copyOf(items);  // defensive copy + immutable
    }
}
```

**C++**
```cpp
class Order {
public:
    Order(std::string customer_id, std::vector<LineItem> items) {
        if (customer_id.empty()) {
            throw std::invalid_argument("customer_id must not be empty");
        }
        if (items.empty()) {
            throw std::invalid_argument("Order must have at least one item");
        }
        customer_id_ = std::move(customer_id);
        items_ = std::move(items);
    }
private:
    std::string customer_id_;
    std::vector<LineItem> items_;
};
```

**Python**
```python
class Order:
    def __init__(self, customer_id: str, items: list[LineItem]):
        if not customer_id or not customer_id.strip():
            raise ValueError("customer_id must not be blank")
        if not items:
            raise ValueError("Order must have at least one item")
        self.customer_id = customer_id
        self.items = list(items)  # defensive copy
```

### Fail-Safe (Graceful Degradation)

When the system can continue with reduced functionality, provide defaults or fallbacks instead of crashing.

**Java**
```java
public class RecommendationService {
    private final RecommendationEngine engine;
    private final List<Product> defaultRecommendations;

    public List<Product> getRecommendations(String userId) {
        try {
            return engine.recommend(userId);
        } catch (RecommendationEngineException e) {
            log.warn("Recommendation engine failed for user {}, returning defaults", userId, e);
            return defaultRecommendations;  // degrade gracefully
        }
    }
}
```

**C++**
```cpp
std::vector<Product> get_recommendations(const std::string& user_id) {
    try {
        return engine_.recommend(user_id);
    } catch (const std::exception& e) {
        spdlog::warn("Recommendation engine failed for {}: {}", user_id, e.what());
        return default_recommendations_;
    }
}
```

**Python**
```python
def get_recommendations(self, user_id: str) -> list[Product]:
    try:
        return self.engine.recommend(user_id)
    except RecommendationEngineError as e:
        logger.warning("Recommendation engine failed for %s: %s", user_id, e)
        return self.default_recommendations
```

### Retry with Exponential Backoff

For transient failures (network timeouts, rate limits), retry with increasing delays and jitter.

**Java**
```java
public <T> T retryWithBackoff(Supplier<T> operation, int maxRetries) {
    int attempt = 0;
    while (true) {
        try {
            return operation.get();
        } catch (TransientException e) {
            attempt++;
            if (attempt >= maxRetries) {
                throw new ServiceUnavailableException(
                    "Failed after " + maxRetries + " retries", e);
            }
            long delay = (long) Math.pow(2, attempt) * 100;       // exponential
            long jitter = ThreadLocalRandom.current().nextLong(delay / 2);  // jitter
            log.warn("Attempt {} failed, retrying in {}ms", attempt, delay + jitter);
            Thread.sleep(delay + jitter);
        }
    }
}
```

**C++**
```cpp
#include <chrono>
#include <thread>
#include <random>
#include <functional>

template <typename T>
T retry_with_backoff(std::function<T()> operation, int max_retries) {
    std::mt19937 rng(std::random_device{}());
    for (int attempt = 0; ; ++attempt) {
        try {
            return operation();
        } catch (const TransientException& e) {
            if (attempt + 1 >= max_retries) throw;
            auto delay = std::chrono::milliseconds(
                static_cast<long>(std::pow(2, attempt + 1)) * 100);
            std::uniform_int_distribution<long> dist(0, delay.count() / 2);
            auto jitter = std::chrono::milliseconds(dist(rng));
            std::this_thread::sleep_for(delay + jitter);
        }
    }
}
```

**Python**
```python
import time
import random

def retry_with_backoff(operation, max_retries: int = 3):
    for attempt in range(max_retries):
        try:
            return operation()
        except TransientError as e:
            if attempt + 1 == max_retries:
                raise ServiceUnavailableError(
                    f"Failed after {max_retries} retries") from e
            delay = (2 ** (attempt + 1)) * 0.1
            jitter = random.uniform(0, delay / 2)
            time.sleep(delay + jitter)
```

### Null Object Pattern

Replace `null` / `None` with a no-op implementation that conforms to the interface. Eliminates null checks throughout the codebase.

**Java**
```java
public interface Logger {
    void log(String message);
    void error(String message, Throwable t);
}

// Real implementation
public class FileLogger implements Logger {
    public void log(String message) { /* write to file */ }
    public void error(String message, Throwable t) { /* write error to file */ }
}

// Null Object — safe default, does nothing
public class NullLogger implements Logger {
    public void log(String message) { /* no-op */ }
    public void error(String message, Throwable t) { /* no-op */ }
}

// Usage — no null checks needed
public class OrderService {
    private final Logger logger;

    public OrderService(Logger logger) {
        this.logger = (logger != null) ? logger : new NullLogger();
    }

    public void process(Order order) {
        logger.log("Processing order " + order.getId());  // always safe
        // ...
    }
}
```

**C++**
```cpp
class Logger {
public:
    virtual ~Logger() = default;
    virtual void log(const std::string& msg) = 0;
};

class NullLogger : public Logger {
public:
    void log(const std::string&) override { /* no-op */ }
};

class OrderService {
    std::unique_ptr<Logger> logger_;
public:
    explicit OrderService(std::unique_ptr<Logger> logger = nullptr)
        : logger_(logger ? std::move(logger) : std::make_unique<NullLogger>()) {}

    void process(const Order& order) {
        logger_->log("Processing order " + order.id());  // always safe
    }
};
```

**Python**
```python
from abc import ABC, abstractmethod

class Logger(ABC):
    @abstractmethod
    def log(self, message: str) -> None: ...

class NullLogger(Logger):
    def log(self, message: str) -> None:
        pass  # no-op

class OrderService:
    def __init__(self, logger: Logger | None = None):
        self.logger = logger or NullLogger()

    def process(self, order: Order) -> None:
        self.logger.log(f"Processing order {order.id}")  # always safe
```

### Try-with-Resources / RAII / Context Managers

Guarantee cleanup of resources (files, connections, locks) regardless of exceptions.

**Java — try-with-resources**
```java
// Any class implementing AutoCloseable gets automatic cleanup
public class DatabaseConnection implements AutoCloseable {
    private final Connection conn;

    public DatabaseConnection(String url) throws SQLException {
        this.conn = DriverManager.getConnection(url);
    }

    @Override
    public void close() throws SQLException {
        conn.close();
    }
}

// Usage — close() called even if exception thrown
try (var db = new DatabaseConnection(url);
     var stmt = db.prepareStatement(sql)) {
    ResultSet rs = stmt.executeQuery();
    // process results
}  // both stmt and db are closed automatically, in reverse order
```

**C++ — RAII (Resource Acquisition Is Initialization)**
```cpp
// The RAII idiom: acquire in constructor, release in destructor
class FileHandle {
    FILE* fp_;
public:
    explicit FileHandle(const char* path, const char* mode)
        : fp_(std::fopen(path, mode)) {
        if (!fp_) throw std::runtime_error("Failed to open file");
    }
    ~FileHandle() {
        if (fp_) std::fclose(fp_);  // guaranteed cleanup
    }
    // Delete copy, allow move
    FileHandle(const FileHandle&) = delete;
    FileHandle& operator=(const FileHandle&) = delete;
    FileHandle(FileHandle&& other) noexcept : fp_(other.fp_) { other.fp_ = nullptr; }

    FILE* get() const noexcept { return fp_; }
};

// Usage — destructor runs when scope exits (normal or exception)
{
    FileHandle fh("data.txt", "r");
    // use fh.get() ...
}  // fh destroyed here, file closed
```

Standard library RAII wrappers: `std::unique_ptr`, `std::shared_ptr`, `std::lock_guard`, `std::fstream`.

**Python — context managers**
```python
class DatabaseConnection:
    def __init__(self, url: str):
        self.conn = connect(url)

    def __enter__(self):
        return self.conn

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.conn.close()
        return False  # don't suppress exceptions

# Usage
with DatabaseConnection(url) as conn:
    cursor = conn.execute(sql)
    # process results
# conn.close() called automatically

# Quick context manager via contextlib
from contextlib import contextmanager

@contextmanager
def open_db(url: str):
    conn = connect(url)
    try:
        yield conn
    finally:
        conn.close()
```

### Global Exception Handler

A catch-all at the application boundary that converts unhandled exceptions to appropriate responses, logs them, and prevents stack traces from leaking to users.

**Java (Spring Boot)**
```java
@RestControllerAdvice
public class GlobalExceptionHandler {

    @ExceptionHandler(UserNotFoundException.class)
    public ResponseEntity<ErrorResponse> handleNotFound(UserNotFoundException e) {
        return ResponseEntity.status(404)
            .body(new ErrorResponse(e.getErrorCode(), e.getMessage()));
    }

    @ExceptionHandler(IllegalArgumentException.class)
    public ResponseEntity<ErrorResponse> handleBadRequest(IllegalArgumentException e) {
        return ResponseEntity.badRequest()
            .body(new ErrorResponse("BAD_REQUEST", e.getMessage()));
    }

    @ExceptionHandler(Exception.class)
    public ResponseEntity<ErrorResponse> handleUnexpected(Exception e) {
        log.error("Unhandled exception", e);  // log full trace internally
        return ResponseEntity.status(500)
            .body(new ErrorResponse("INTERNAL_ERROR", "An unexpected error occurred"));
    }
}
```

**Python (FastAPI)**
```python
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI()

@app.exception_handler(UserNotFoundException)
async def user_not_found_handler(request: Request, exc: UserNotFoundException):
    return JSONResponse(status_code=404, content={"error": exc.error_code, "message": str(exc)})

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled exception: %s", exc, exc_info=True)
    return JSONResponse(status_code=500, content={"error": "INTERNAL_ERROR", "message": "An unexpected error occurred"})
```

---

## 4. Anti-Patterns

### Swallowing Exceptions

```java
// BAD — silently swallowed, debugging nightmare
try {
    processPayment(order);
} catch (Exception e) {
    // nothing here
}

// GOOD — at minimum log it; better: rethrow or handle
try {
    processPayment(order);
} catch (PaymentException e) {
    log.error("Payment failed for order {}", order.getId(), e);
    throw new OrderProcessingException("Payment step failed", e);
}
```

### Catching Generic Exception / Throwable

```java
// BAD — catches NullPointerException, OutOfMemoryError, everything
try {
    return userService.findUser(id);
} catch (Exception e) {
    return null;
}

// GOOD — catch the specific exception you can handle
try {
    return userService.findUser(id);
} catch (UserNotFoundException e) {
    return createGuestUser();
}
```

### Using Exceptions for Control Flow

```java
// BAD — exception as a goto; expensive and confusing
public boolean isInteger(String s) {
    try {
        Integer.parseInt(s);
        return true;
    } catch (NumberFormatException e) {
        return false;
    }
}

// GOOD — check before you leap, or use a dedicated method
public boolean isInteger(String s) {
    return s != null && s.matches("-?\\d+");
}
```

```python
# In Python, EAFP (Easier to Ask Forgiveness than Permission) is idiomatic.
# BUT don't abuse it for expected conditions in hot paths.

# Acceptable EAFP (Pythonic):
try:
    value = my_dict[key]
except KeyError:
    value = default

# Better when keys are often missing (cheaper):
value = my_dict.get(key, default)
```

### Returning Null Instead of Empty Collections

```java
// BAD — every caller must null-check
public List<Order> getOrders(String userId) {
    List<Order> orders = repository.findByUserId(userId);
    if (orders == null) return null;  // propagates null
    return orders;
}

// GOOD — return empty collection
public List<Order> getOrders(String userId) {
    List<Order> orders = repository.findByUserId(userId);
    return orders != null ? orders : List.of();  // never null
}
```

```cpp
// BAD
std::vector<Order>* get_orders(const std::string& id) {
    return nullptr;  // caller must check
}

// GOOD
std::vector<Order> get_orders(const std::string& id) {
    return {};  // empty vector, safe to iterate
}
```

```python
# BAD
def get_orders(user_id: str) -> list[Order] | None:
    return None

# GOOD
def get_orders(user_id: str) -> list[Order]:
    return []  # always iterable
```

### Exceptions in Constructors

If a constructor throws after acquiring resources, those resources leak because the destructor never runs (the object was never fully constructed).

```cpp
// BAD — if second allocation throws, first leaks
class BadResource {
    int* data1_;
    int* data2_;
public:
    BadResource() {
        data1_ = new int[1000];
        data2_ = new int[1000];  // if this throws, data1_ leaks
    }
    ~BadResource() { delete[] data1_; delete[] data2_; }
};

// GOOD — use RAII members; each cleans up independently
class GoodResource {
    std::unique_ptr<int[]> data1_;
    std::unique_ptr<int[]> data2_;
public:
    GoodResource()
        : data1_(std::make_unique<int[]>(1000)),
          data2_(std::make_unique<int[]>(1000)) {
        // if data2_ allocation throws, data1_ is automatically freed
    }
};
```

In Java, constructor exceptions are less dangerous (GC handles memory), but can leave partially constructed objects registered as listeners or in static collections. Prefer factory methods that validate first, then construct.

---

## Anti-Pattern Summary

| Anti-Pattern | Problem | Fix |
|-------------|---------|-----|
| Empty catch block | Silent failure | Log + rethrow or handle |
| Catching `Exception` / `Throwable` | Hides bugs | Catch specific types |
| Exceptions for control flow | Performance hit, confusing | Conditional checks or dedicated APIs |
| Returning `null` for collections | NullPointerException at caller | Return empty collection |
| Throwing in constructors (C++) | Resource leak | Use RAII members |

---

## Common Interview Questions

**Q1: When should you use checked vs unchecked exceptions in Java?**
Use checked exceptions for recoverable conditions the caller is expected to handle (file not found, network timeout). Use unchecked exceptions for programming errors that indicate bugs (null pointer, illegal argument). In practice, most modern Java codebases prefer unchecked exceptions to avoid polluting method signatures, catching and wrapping checked exceptions at module boundaries.

**Q2: How would you design an error handling strategy for a microservices system?**
Each service defines its own exception hierarchy rooted in a base domain exception. At the service boundary (controller), a global exception handler translates domain exceptions to HTTP/gRPC status codes. Between services, use structured error responses with error codes, human-readable messages, and correlation IDs. Implement retry with exponential backoff for transient failures, circuit breakers for persistent failures, and dead-letter queues for unprocessable messages.

**Q3: Explain RAII and why it matters for error handling in C++.**
RAII (Resource Acquisition Is Initialization) ties resource lifetime to object lifetime. Resources are acquired in the constructor and released in the destructor. Because C++ guarantees that destructors run when objects leave scope — even during stack unwinding from exceptions — RAII ensures no resource leaks. Standard examples include `unique_ptr`, `lock_guard`, and `fstream`. Without RAII, every function with resources would need manual try/catch cleanup blocks.

**Q4: What is the difference between fail-fast and fail-safe? When do you use each?**
Fail-fast detects errors immediately and aborts (e.g., precondition checks in constructors, `Objects.requireNonNull`). Use it for programming errors and invariant violations where continuing would produce corrupt state. Fail-safe detects errors but continues with degraded functionality (e.g., returning cached data when a service is down). Use it for non-critical features where availability is more important than correctness of that specific feature.

**Q5: Why is catching generic `Exception` considered bad practice?**
Catching `Exception` inadvertently catches programming errors like `NullPointerException` and `ClassCastException` that should crash the program so they get noticed and fixed. It also catches `InterruptedException`, breaking thread cancellation. The only place a generic catch is appropriate is at the top-level boundary (global exception handler) where you log the error and return a generic response to the client.

**Q6: How do Result/Either types compare to exceptions?**
Result types make errors explicit in the type signature, forcing callers to handle them at compile time. They compose well with `map`/`flatMap` for chaining operations. Exceptions keep the happy path clean but create invisible control flow and are easy to forget. Result types are preferable in functional codebases and when errors are expected outcomes (parsing, validation). Exceptions are preferable when errors are truly exceptional and the language ecosystem is built around them (Java, Python).

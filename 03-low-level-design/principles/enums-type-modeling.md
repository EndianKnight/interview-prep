# Enums, Constants & Type Modeling

Strong type modeling is one of the most impactful tools in low-level design -- it turns runtime bugs into compile-time errors and makes code self-documenting.

---

## 1. Enums

Enums represent a fixed set of named constants. In LLD interviews, they appear everywhere: order statuses, payment types, user roles, directions, and strategy selectors.

### Java Enums

Java enums are full-fledged classes. They can have fields, constructors, methods, and even implement interfaces.

**Basic enum with fields and methods**

```java
public enum OrderStatus {
    PENDING("Order placed, awaiting confirmation"),
    CONFIRMED("Payment verified"),
    SHIPPED("In transit"),
    DELIVERED("Received by customer"),
    CANCELLED("Order cancelled");

    private final String description;

    OrderStatus(String description) {
        this.description = description;
    }

    public String getDescription() {
        return description;
    }
}
```

**Enum implementing an interface**

```java
public interface Discountable {
    double applyDiscount(double price);
}

public enum MembershipTier implements Discountable {
    BRONZE {
        @Override
        public double applyDiscount(double price) { return price * 0.95; }
    },
    SILVER {
        @Override
        public double applyDiscount(double price) { return price * 0.90; }
    },
    GOLD {
        @Override
        public double applyDiscount(double price) { return price * 0.80; }
    };
}

// Usage -- polymorphism via enum
MembershipTier tier = MembershipTier.GOLD;
double finalPrice = tier.applyDiscount(100.0); // 80.0
```

**Enum with abstract methods (strategy via enum)**

```java
public enum SortStrategy {
    PRICE_ASC {
        @Override
        public int compare(Product a, Product b) {
            return Double.compare(a.getPrice(), b.getPrice());
        }
    },
    PRICE_DESC {
        @Override
        public int compare(Product a, Product b) {
            return Double.compare(b.getPrice(), a.getPrice());
        }
    },
    NAME {
        @Override
        public int compare(Product a, Product b) {
            return a.getName().compareTo(b.getName());
        }
    };

    public abstract int compare(Product a, Product b);

    public List<Product> sort(List<Product> products) {
        products.sort(this::compare);
        return products;
    }
}
```

**Key Java enum features:**
- Implicitly `final` and extend `java.lang.Enum`
- `values()` returns all constants; `valueOf(String)` parses by name
- `name()` and `ordinal()` are built-in (avoid relying on `ordinal()` for persistence)
- Can be used in `switch` statements and `EnumSet` / `EnumMap` for high-performance collections

### C++ Scoped Enums (`enum class`)

C++ offers both unscoped (`enum`) and scoped (`enum class`) enumerations. Always prefer `enum class` in modern C++.

```cpp
// Unscoped (legacy) -- pollutes enclosing namespace, implicitly converts to int
enum Color { RED, GREEN, BLUE }; // RED is visible without qualification

// Scoped (modern) -- type-safe, no implicit conversion
enum class OrderStatus : uint8_t {  // underlying type specified
    Pending,
    Confirmed,
    Shipped,
    Delivered,
    Cancelled
};

// Must qualify with scope
OrderStatus status = OrderStatus::Pending;

// No implicit conversion -- requires explicit cast
int raw = static_cast<int>(status);
```

| Feature | `enum` (unscoped) | `enum class` (scoped) |
|---------|-------------------|----------------------|
| Scope | Pollutes enclosing namespace | Names scoped to enum |
| Implicit conversion | Converts to `int` implicitly | No implicit conversion |
| Forward declaration | Only with underlying type | Always allowed |
| Type safety | Weak | Strong |

**Adding behavior to C++ enums (free functions)**

C++ enums cannot have member methods, so behavior is added via free functions or a companion class:

```cpp
#include <string>
#include <stdexcept>

enum class OrderStatus : uint8_t {
    Pending, Confirmed, Shipped, Delivered, Cancelled
};

std::string to_string(OrderStatus s) {
    switch (s) {
        case OrderStatus::Pending:   return "Pending";
        case OrderStatus::Confirmed: return "Confirmed";
        case OrderStatus::Shipped:   return "Shipped";
        case OrderStatus::Delivered: return "Delivered";
        case OrderStatus::Cancelled: return "Cancelled";
    }
    throw std::invalid_argument("Unknown OrderStatus");
}

bool can_cancel(OrderStatus s) {
    return s == OrderStatus::Pending || s == OrderStatus::Confirmed;
}
```

### Python Enums

Python provides enums through the `enum` module (Python 3.4+).

```python
from enum import Enum, IntEnum, auto

class OrderStatus(Enum):
    PENDING = auto()
    CONFIRMED = auto()
    SHIPPED = auto()
    DELIVERED = auto()
    CANCELLED = auto()

    @property
    def description(self) -> str:
        descriptions = {
            OrderStatus.PENDING: "Order placed, awaiting confirmation",
            OrderStatus.CONFIRMED: "Payment verified",
            OrderStatus.SHIPPED: "In transit",
            OrderStatus.DELIVERED: "Received by customer",
            OrderStatus.CANCELLED: "Order cancelled",
        }
        return descriptions[self]

# Access
status = OrderStatus.PENDING
print(status.name)   # "PENDING"
print(status.value)  # 1
```

**IntEnum for integer compatibility**

```python
from enum import IntEnum

class HttpStatus(IntEnum):
    OK = 200
    NOT_FOUND = 404
    INTERNAL_ERROR = 500

# IntEnum allows comparison with plain ints
if response_code == HttpStatus.NOT_FOUND:
    handle_404()
```

**Enum with behavior (strategy pattern)**

```python
from enum import Enum

class SortStrategy(Enum):
    PRICE_ASC = "price_asc"
    PRICE_DESC = "price_desc"
    NAME = "name"

    def sort(self, products: list) -> list:
        if self == SortStrategy.PRICE_ASC:
            return sorted(products, key=lambda p: p.price)
        elif self == SortStrategy.PRICE_DESC:
            return sorted(products, key=lambda p: p.price, reverse=True)
        elif self == SortStrategy.NAME:
            return sorted(products, key=lambda p: p.name)
```

| Feature | `Enum` | `IntEnum` | `Flag` |
|---------|--------|-----------|--------|
| Comparison with `int` | No | Yes | No |
| Bitwise operations | No | No | Yes |
| Identity guaranteed | Yes | Yes | Yes |
| Typical use | General constants | Status codes, levels | Permissions, bitfields |

### Enum as State Machine

Enums are a natural fit for modeling state transitions. Encoding valid transitions directly in the enum prevents illegal state changes.

**Java**

```java
public enum OrderStatus {
    PENDING {
        @Override
        public Set<OrderStatus> allowedTransitions() {
            return EnumSet.of(CONFIRMED, CANCELLED);
        }
    },
    CONFIRMED {
        @Override
        public Set<OrderStatus> allowedTransitions() {
            return EnumSet.of(SHIPPED, CANCELLED);
        }
    },
    SHIPPED {
        @Override
        public Set<OrderStatus> allowedTransitions() {
            return EnumSet.of(DELIVERED);
        }
    },
    DELIVERED {
        @Override
        public Set<OrderStatus> allowedTransitions() {
            return EnumSet.noneOf(OrderStatus.class);
        }
    },
    CANCELLED {
        @Override
        public Set<OrderStatus> allowedTransitions() {
            return EnumSet.noneOf(OrderStatus.class);
        }
    };

    public abstract Set<OrderStatus> allowedTransitions();

    public OrderStatus transitionTo(OrderStatus next) {
        if (!allowedTransitions().contains(next)) {
            throw new IllegalStateException(
                "Cannot transition from " + this + " to " + next);
        }
        return next;
    }
}
```

**C++**

```cpp
#include <unordered_set>
#include <stdexcept>
#include <string>

enum class OrderStatus { Pending, Confirmed, Shipped, Delivered, Cancelled };

std::unordered_set<OrderStatus> allowed_transitions(OrderStatus s) {
    switch (s) {
        case OrderStatus::Pending:   return {OrderStatus::Confirmed, OrderStatus::Cancelled};
        case OrderStatus::Confirmed: return {OrderStatus::Shipped, OrderStatus::Cancelled};
        case OrderStatus::Shipped:   return {OrderStatus::Delivered};
        default:                     return {};
    }
}

OrderStatus transition_to(OrderStatus current, OrderStatus next) {
    auto allowed = allowed_transitions(current);
    if (allowed.find(next) == allowed.end()) {
        throw std::invalid_argument("Invalid state transition");
    }
    return next;
}
```

**Python**

```python
from enum import Enum, auto

class OrderStatus(Enum):
    PENDING = auto()
    CONFIRMED = auto()
    SHIPPED = auto()
    DELIVERED = auto()
    CANCELLED = auto()

    @property
    def allowed_transitions(self) -> set:
        transitions = {
            OrderStatus.PENDING:   {OrderStatus.CONFIRMED, OrderStatus.CANCELLED},
            OrderStatus.CONFIRMED: {OrderStatus.SHIPPED, OrderStatus.CANCELLED},
            OrderStatus.SHIPPED:   {OrderStatus.DELIVERED},
            OrderStatus.DELIVERED: set(),
            OrderStatus.CANCELLED: set(),
        }
        return transitions[self]

    def transition_to(self, next_status: "OrderStatus") -> "OrderStatus":
        if next_status not in self.allowed_transitions:
            raise ValueError(f"Cannot transition from {self.name} to {next_status.name}")
        return next_status
```

---

## 2. Constants & Magic Values

### Why Magic Numbers and Strings Are Bad

```java
// BAD -- what does 3 mean? What is "PRO"?
if (user.getType() == 3 && user.getPlan().equals("PRO")) {
    applyDiscount(order, 0.15);
}

// GOOD -- self-documenting, refactor-safe, autocomplete-friendly
if (user.getType() == UserType.ENTERPRISE && user.getPlan() == Plan.PRO) {
    applyDiscount(order, Discounts.ENTERPRISE_PRO);
}
```

**Problems with magic values:**
- No discoverability -- you cannot autocomplete the number `3`
- No type safety -- nothing stops you from passing `42` where only `1-5` are valid
- Scattered duplication -- the same literal appears in dozens of files
- Silent bugs -- a typo in `"PROO"` compiles fine but fails at runtime

### Constants: Language-Specific Idioms

**Java: `static final`**

```java
public final class HttpConstants {
    private HttpConstants() {} // prevent instantiation

    public static final int DEFAULT_TIMEOUT_MS = 5000;
    public static final String AUTH_HEADER = "Authorization";
    public static final double MAX_RETRY_BACKOFF_SECONDS = 30.0;
}
```

**C++: `constexpr` and `inline constexpr`**

```cpp
// constexpr -- evaluated at compile time
namespace config {
    inline constexpr int kDefaultTimeoutMs = 5000;
    inline constexpr double kMaxRetryBackoff = 30.0;
    inline constexpr std::string_view kAuthHeader = "Authorization"; // C++17
}
```

**Python: Module-level UPPER_CASE**

```python
# config.py -- convention only, not enforced by the language
DEFAULT_TIMEOUT_MS = 5000
AUTH_HEADER = "Authorization"
MAX_RETRY_BACKOFF_SECONDS = 30.0
```

### Constants vs Enums

| Aspect | Constants | Enums |
|--------|-----------|-------|
| Type safety | Weak (just a value) | Strong (distinct type) |
| Grouping | Manual (class/namespace) | Inherent (enum type) |
| Iteration | Not possible | `values()` / `list(Enum)` |
| Behavior | None | Can have methods |
| Best for | Thresholds, config values | Fixed sets of named options |

**Rule of thumb:** If you have a fixed, finite set of related values that a variable can take, use an enum. If you have standalone numeric or string configuration values, use constants.

### Configuration vs Compile-Time Constants

| Property | Compile-time constant | Runtime configuration |
|----------|----------------------|----------------------|
| When resolved | Compilation / class loading | Application startup / dynamically |
| Change requires | Rebuild and redeploy | Config reload or restart |
| Examples | `Math.PI`, buffer sizes | DB connection strings, feature flags |
| Java | `static final` primitives/strings | Properties, env vars, config files |
| C++ | `constexpr` | Config file, CLI args |
| Python | Module-level constants | `os.environ`, YAML/JSON config |

---

## 3. Advanced Type Modeling

### Sealed Classes and Interfaces (Java 17+)

Sealed types restrict which classes can extend or implement them. This creates a **closed** hierarchy where the compiler knows every possible subtype.

```java
public sealed interface PaymentMethod
    permits CreditCard, BankTransfer, DigitalWallet {
}

public record CreditCard(String number, String expiry, String cvv)
    implements PaymentMethod {}

public record BankTransfer(String iban, String bic)
    implements PaymentMethod {}

public record DigitalWallet(String walletId, WalletProvider provider)
    implements PaymentMethod {}
```

**Why sealed types matter:**
- The compiler enforces exhaustive `switch` -- if you forget a case, it will not compile
- New subtypes require updating the `permits` list, making extension deliberate
- Combined with records, they give you concise algebraic data types in Java

### Algebraic Data Types (Sum Types + Product Types)

| Concept | Definition | Java Equivalent |
|---------|-----------|-----------------|
| **Product type** | A type that holds A *and* B (a struct/record) | `record Point(int x, int y)` |
| **Sum type** | A type that is A *or* B (a tagged union) | `sealed interface` with subclasses |
| **ADT** | Combination of sum and product types | Sealed interface + records |

**Modeling a result type:**

```java
public sealed interface Result<T>
    permits Result.Success, Result.Failure {

    record Success<T>(T value) implements Result<T> {}
    record Failure<T>(String error, Exception cause) implements Result<T> {}
}

// Usage
Result<Order> result = processOrder(request);
switch (result) {
    case Result.Success<Order> s -> ship(s.value());
    case Result.Failure<Order> f -> log.error(f.error(), f.cause());
}
```

**C++ equivalent using `std::variant`:**

```cpp
#include <variant>
#include <string>

struct Success { Order order; };
struct Failure { std::string error; };

using OrderResult = std::variant<Success, Failure>;

void handle(const OrderResult& result) {
    std::visit([](auto&& arg) {
        using T = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<T, Success>) {
            ship(arg.order);
        } else {
            log_error(arg.error);
        }
    }, result);
}
```

**Python equivalent using union types and dataclasses:**

```python
from dataclasses import dataclass
from typing import Union

@dataclass(frozen=True)
class Success:
    order: Order

@dataclass(frozen=True)
class Failure:
    error: str
    cause: Exception | None = None

OrderResult = Union[Success, Failure]  # or Success | Failure in 3.10+

def handle(result: OrderResult) -> None:
    match result:   # Python 3.10+ structural pattern matching
        case Success(order=order):
            ship(order)
        case Failure(error=error):
            log_error(error)
```

### Pattern Matching with Sealed Types

Pattern matching on sealed types gives you exhaustiveness checking -- the compiler guarantees you handle every case.

**Java 21+ pattern matching with switch:**

```java
public sealed interface Shape
    permits Circle, Rectangle, Triangle {}

public record Circle(double radius) implements Shape {}
public record Rectangle(double width, double height) implements Shape {}
public record Triangle(double base, double height) implements Shape {}

public double area(Shape shape) {
    return switch (shape) {
        case Circle c      -> Math.PI * c.radius() * c.radius();
        case Rectangle r   -> r.width() * r.height();
        case Triangle t    -> 0.5 * t.base() * t.height();
        // No default needed -- compiler knows these are all cases
    };
}
```

### Type-Safe IDs

A common source of bugs: passing a `long userId` where a `long orderId` is expected. The compiler cannot help because both are `long`. Wrapping IDs in distinct types eliminates this class of errors.

**Java**

```java
public record UserId(long value) {}
public record OrderId(long value) {}

// This signature makes it impossible to swap arguments
public Order findOrder(OrderId orderId, UserId userId) { ... }

// Compile error: incompatible types
OrderId oid = new OrderId(42);
UserId uid = new UserId(7);
findOrder(uid, oid);  // ERROR -- types are swapped
```

**C++**

```cpp
// Strong typedef using a tagged wrapper
template <typename Tag, typename T = long>
struct StrongId {
    T value;
    explicit StrongId(T v) : value(v) {}
    bool operator==(const StrongId& o) const { return value == o.value; }
};

struct UserTag {};
struct OrderTag {};

using UserId = StrongId<UserTag>;
using OrderId = StrongId<OrderTag>;

Order find_order(OrderId oid, UserId uid);

// Compile error -- UserId != OrderId
find_order(UserId{7}, OrderId{42});  // ERROR
```

**Python (runtime checking via NewType or dataclass)**

```python
from typing import NewType

UserId = NewType("UserId", int)
OrderId = NewType("OrderId", int)

def find_order(order_id: OrderId, user_id: UserId) -> Order:
    ...

# Type checkers (mypy, pyright) will flag this
oid = OrderId(42)
uid = UserId(7)
find_order(uid, oid)  # Type error caught by mypy
```

### Making Invalid States Unrepresentable

The principle: design your types so that the only values that can be constructed are valid values. Instead of validating at runtime, make the type system enforce your invariants.

**Bad: runtime validation scattered everywhere**

```java
public class Booking {
    private String status; // "pending", "confirmed", "checked_in"
    private LocalDateTime checkInTime; // only valid if status == "checked_in"
    private String roomNumber;         // only valid if status == "confirmed" or "checked_in"
}
// Nothing prevents: status = "pending" with a roomNumber set
```

**Good: distinct types per state**

```java
public sealed interface Booking
    permits PendingBooking, ConfirmedBooking, CheckedInBooking {

    String guestName();
    LocalDate date();
}

public record PendingBooking(String guestName, LocalDate date)
    implements Booking {}

public record ConfirmedBooking(String guestName, LocalDate date, String roomNumber)
    implements Booking {}

public record CheckedInBooking(String guestName, LocalDate date, String roomNumber,
                               LocalDateTime checkInTime)
    implements Booking {}
```

Now `PendingBooking` cannot have a `roomNumber`, and `ConfirmedBooking` cannot have a `checkInTime`. The type system enforces the invariant -- no validation needed.

---

## 4. Interview Application

### When to Use Enum vs Class Hierarchy vs Strategy Pattern

| Criterion | Enum | Sealed class hierarchy | Strategy pattern (interface) |
|-----------|------|----------------------|------------------------------|
| Set of options | Fixed, known at compile time | Fixed but each variant has different data | Open to extension at runtime |
| Behavior variation | Minor (1-2 methods) | Moderate (shared + variant-specific) | Significant, swappable at runtime |
| Data per variant | Same fields for all | Different fields per variant | Injected as dependency |
| Adding a new variant | Modify the enum (recompile) | Modify `permits` + add class | Add new class, no existing code changes |
| Typical LLD use | Status, type, direction | Payment methods, notifications, shapes | Pricing rules, sort orders, validators |

### Decision Flowchart for LLD Interviews

1. **Is the set of values fixed and small (< 15)?** Use an enum.
2. **Do different values carry different data fields?** Use sealed classes / ADTs.
3. **Do you need to add new variants without modifying existing code?** Use strategy pattern (Open/Closed Principle).
4. **Is it just a standalone numeric or string constant?** Use a named constant.
5. **Can you represent the concept so that invalid states are impossible to construct?** Do so -- the interviewer will notice.

### Practical Tips for LLD Interviews

- **Start with enums** for any finite set: `OrderStatus`, `PaymentType`, `UserRole`, `Direction`
- **Use enum state machines** to model workflows (order lifecycle, task management)
- **Wrap raw IDs** in value types when the method signature has two or more IDs of the same primitive type
- **Prefer sealed types** when modeling domain events, commands, or result types that carry variant-specific data
- **Mention tradeoffs**: "I am using an enum here because the set is closed and known. If we needed extensibility, I would switch to a strategy interface."

---

## Common Interview Questions

**Q1: Why should you avoid using `ordinal()` for persistence in Java enums?**
`ordinal()` is based on declaration order. Inserting a new constant in the middle shifts all subsequent ordinals, silently corrupting stored data. Always persist by `name()` or an explicit code field instead.

**Q2: What is the difference between `enum` and `enum class` in C++?**
`enum` is unscoped -- its members leak into the enclosing namespace and implicitly convert to `int`. `enum class` is scoped and type-safe: members must be qualified (`Color::Red`) and require explicit casts to convert to integers.

**Q3: How do you add behavior to an enum in Java vs C++?**
Java enums support fields, constructors, methods, and abstract methods directly. Each constant can override abstract methods, effectively implementing the strategy pattern. In C++, `enum class` has no methods, so behavior is added through companion free functions or a separate helper class.

**Q4: What are sealed classes and why do they matter in LLD?**
Sealed classes (Java 17+) restrict which classes can extend them. This creates a closed type hierarchy that the compiler can check exhaustively in `switch` statements. In LLD, they model domain concepts where the set of variants is fixed (payment methods, notification channels) and each variant carries different data.

**Q5: What does "making invalid states unrepresentable" mean?**
It means designing your types so that illegal combinations of data cannot be constructed. Instead of using a single class with nullable fields and a status string, you create separate types for each state. If a `PendingBooking` record has no `roomNumber` field, it is impossible to accidentally assign one -- the compiler enforces the invariant.

**Q6: When would you choose the strategy pattern over an enum with behavior?**
Choose strategy when: (a) new strategies must be added without modifying existing code (Open/Closed Principle), (b) strategies need dependencies injected (database, service calls), or (c) strategies are selected dynamically at runtime based on configuration. Use an enum when the set is small, fixed, and each variant only needs simple, self-contained logic.

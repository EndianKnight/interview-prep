# Immutability & Defensive Programming

Designing objects that cannot change after construction leads to safer, more predictable systems and eliminates entire categories of bugs.

---

## 1. Immutable Objects

An object is **immutable** if its observable state cannot be modified after construction. This requires:

1. All fields are set at construction time and never reassigned
2. No setter methods or mutating operations
3. No leaking of mutable internal references
4. The class itself cannot be subclassed (prevents mutable subclass overrides)

### Why Immutability Matters

| Benefit | Explanation |
|---------|-------------|
| **Thread safety** | No synchronization needed; immutable objects can be shared freely across threads |
| **Hashability** | Safe to use as map keys and set elements since hash code never changes |
| **Simpler reasoning** | No temporal coupling; an object means the same thing everywhere it appears |
| **Failure atomicity** | No partially-modified state if an operation fails |
| **Cache friendly** | Safe to cache and reuse without worrying about invalidation |

### Language Support Comparison

| Feature | Java | C++ | Python |
|---------|------|-----|--------|
| Immutable field | `final` | `const` member | No native keyword; convention `_` prefix |
| Compile-time constant | `static final` | `constexpr` | N/A (no compilation) |
| Immutable class | `record` (Java 16+) | custom (all `const` members) | `@dataclass(frozen=True)` |
| Immutable string | `String` (always immutable) | `const std::string&` | `str` (always immutable) |
| Immutable collection | `Collections.unmodifiableList()`, `List.of()` | `const std::vector<T>&` | `tuple`, `frozenset` |
| Prevent subclassing | `final class` | `final` class (C++11) | Not built-in (use `__init_subclass__`) |

---

### Java: Building an Immutable Class

```java
// All 5 rules of immutability applied
public final class Money {                          // 1. final class — cannot be subclassed
    private final BigDecimal amount;                // 2. final fields
    private final Currency currency;
    private final List<String> tags;                // mutable type — needs protection

    public Money(BigDecimal amount, Currency currency, List<String> tags) {
        // 4. Validate inputs (defensive programming)
        if (amount == null || currency == null) {
            throw new IllegalArgumentException("amount and currency must not be null");
        }
        this.amount = amount;                       // BigDecimal is itself immutable
        this.currency = currency;                   // Currency is itself immutable
        this.tags = List.copyOf(tags);              // 3. Defensive copy of mutable arg
    }

    // No setters — only getters                     // 5. No mutation methods
    public BigDecimal getAmount() { return amount; }
    public Currency getCurrency() { return currency; }
    public List<String> getTags() { return tags; }  // List.copyOf returns unmodifiable list

    // Return a NEW object for "mutations"
    public Money add(Money other) {
        if (!this.currency.equals(other.currency)) {
            throw new IllegalArgumentException("Currency mismatch");
        }
        return new Money(this.amount.add(other.amount), this.currency, this.tags);
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof Money m)) return false;
        return amount.equals(m.amount) && currency.equals(m.currency);
    }

    @Override
    public int hashCode() {
        return Objects.hash(amount, currency);
    }
}
```

**Java Records (Java 16+)** provide immutable value types with less boilerplate:

```java
public record Money(BigDecimal amount, Currency currency) {
    // Compact constructor for validation
    public Money {
        Objects.requireNonNull(amount, "amount must not be null");
        Objects.requireNonNull(currency, "currency must not be null");
    }

    public Money add(Money other) {
        if (!this.currency.equals(other.currency))
            throw new IllegalArgumentException("Currency mismatch");
        return new Money(this.amount.add(other.amount), this.currency);
    }
}
```

### C++: Immutability with `const` and `constexpr`

```cpp
class Money final {  // final prevents subclassing
public:
    Money(double amount, std::string currency)
        : amount_(amount), currency_(std::move(currency)) {
        if (amount < 0) {
            throw std::invalid_argument("amount must be non-negative");
        }
    }

    // const member functions — guarantee no state modification
    [[nodiscard]] double amount() const { return amount_; }
    [[nodiscard]] const std::string& currency() const { return currency_; }

    // Return new object for "mutations"
    [[nodiscard]] Money add(const Money& other) const {
        if (currency_ != other.currency_) {
            throw std::invalid_argument("Currency mismatch");
        }
        return Money(amount_ + other.amount_, currency_);
    }

    bool operator==(const Money& other) const = default;  // C++20

private:
    const double amount_;        // const members prevent reassignment
    const std::string currency_;
};

// constexpr: compile-time immutability
constexpr int MAX_RETRIES = 3;
constexpr double PI = 3.14159265358979;

// constexpr functions (C++14+): evaluated at compile time when possible
constexpr int factorial(int n) {
    return (n <= 1) ? 1 : n * factorial(n - 1);
}
static_assert(factorial(5) == 120);  // verified at compile time
```

### Python: Immutability Patterns

```python
from dataclasses import dataclass
from typing import Tuple

# Frozen dataclass — raises FrozenInstanceError on attribute assignment
@dataclass(frozen=True)
class Money:
    amount: float
    currency: str

    def __post_init__(self):
        if self.amount < 0:
            raise ValueError("amount must be non-negative")

    def add(self, other: "Money") -> "Money":
        if self.currency != other.currency:
            raise ValueError("Currency mismatch")
        return Money(self.amount + other.amount, self.currency)


m1 = Money(10.0, "USD")
m2 = Money(5.0, "USD")
m3 = m1.add(m2)         # Money(amount=15.0, currency='USD')
# m1.amount = 99         # FrozenInstanceError!

# Built-in immutable types
immutable_point = (3, 4)             # tuple
immutable_tags = frozenset({"a", "b"})  # frozenset

# Named tuples — lightweight immutable types
from collections import namedtuple
Point = namedtuple("Point", ["x", "y"])
p = Point(3, 4)
# p.x = 5  # AttributeError!
```

---

## 2. Value Objects vs Entities

In domain-driven design, distinguishing between value objects and entities is fundamental to correct modeling.

| Aspect | Value Object | Entity |
|--------|-------------|--------|
| **Identity** | Defined by its attributes | Defined by a unique ID |
| **Equality** | Two are equal if all attributes match | Two are equal only if IDs match |
| **Mutability** | Should be immutable | Typically mutable (state changes over time) |
| **Lifecycle** | Created and discarded freely | Tracked, persisted, has a lifecycle |
| **Examples** | Money, Address, DateRange, Color, Email | User, Order, BankAccount, Product |

### When to Use Which in LLD

- **Value Object**: When the concept has no meaningful identity beyond its data. You would never say "I want *that specific* $10 bill" — any $10 bill is the same.
- **Entity**: When you need to track something over time. "Order #12345" is a specific order whose state changes from PLACED to SHIPPED to DELIVERED.

### Java

```java
// VALUE OBJECT — defined by attributes, immutable
public record Address(String street, String city, String zipCode, String country) {
    public Address {
        Objects.requireNonNull(street);
        Objects.requireNonNull(city);
        Objects.requireNonNull(zipCode);
    }
    // equals/hashCode auto-generated from ALL fields by record
}

// ENTITY — defined by identity, mutable state
public class Order {
    private final UUID id;            // identity field — never changes
    private OrderStatus status;       // mutable state
    private final List<LineItem> items;
    private final Instant createdAt;

    public Order(UUID id, List<LineItem> items) {
        this.id = id;
        this.status = OrderStatus.PLACED;
        this.items = new ArrayList<>(items);  // defensive copy
        this.createdAt = Instant.now();
    }

    public void ship() {
        if (status != OrderStatus.PLACED) {
            throw new IllegalStateException("Can only ship a PLACED order");
        }
        this.status = OrderStatus.SHIPPED;
    }

    // Equality based on identity ONLY
    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof Order order)) return false;
        return id.equals(order.id);
    }

    @Override
    public int hashCode() { return id.hashCode(); }
}
```

### C++

```cpp
// VALUE OBJECT
struct Address {
    std::string street;
    std::string city;
    std::string zip_code;

    bool operator==(const Address& other) const = default;  // compare all fields
};

// ENTITY
class Order {
public:
    explicit Order(std::string id, std::vector<LineItem> items)
        : id_(std::move(id)),
          status_(OrderStatus::PLACED),
          items_(std::move(items)) {}

    void ship() {
        if (status_ != OrderStatus::PLACED)
            throw std::logic_error("Can only ship a PLACED order");
        status_ = OrderStatus::SHIPPED;
    }

    // Equality based on identity only
    bool operator==(const Order& other) const { return id_ == other.id_; }

private:
    const std::string id_;    // identity — never changes
    OrderStatus status_;      // mutable state
    std::vector<LineItem> items_;
};
```

### Python

```python
from dataclasses import dataclass, field
from uuid import UUID, uuid4
from enum import Enum

# VALUE OBJECT — frozen, equality on all fields (default for frozen dataclass)
@dataclass(frozen=True)
class Address:
    street: str
    city: str
    zip_code: str
    country: str = "US"

# ENTITY — equality on id only
class OrderStatus(Enum):
    PLACED = "PLACED"
    SHIPPED = "SHIPPED"
    DELIVERED = "DELIVERED"

@dataclass
class Order:
    id: UUID = field(default_factory=uuid4)
    status: OrderStatus = OrderStatus.PLACED
    items: list = field(default_factory=list)

    def ship(self):
        if self.status != OrderStatus.PLACED:
            raise ValueError("Can only ship a PLACED order")
        self.status = OrderStatus.SHIPPED

    def __eq__(self, other):
        if not isinstance(other, Order):
            return NotImplemented
        return self.id == other.id

    def __hash__(self):
        return hash(self.id)
```

---

## 3. Defensive Copying

Defensive copying prevents callers from mutating an object's internal state through references that escape.

### The Problem

```java
// BROKEN: internal state leaks via constructor arg and getter
public class Event {
    private final List<String> attendees;

    public Event(List<String> attendees) {
        this.attendees = attendees;          // DANGER: stores external reference
    }

    public List<String> getAttendees() {
        return attendees;                    // DANGER: caller can mutate internals
    }
}

// Exploit
List<String> names = new ArrayList<>(List.of("Alice", "Bob"));
Event event = new Event(names);
names.add("Eve");                           // modifies event's internal state!
event.getAttendees().clear();               // wipes out all attendees!
```

### The Fix: Copy on the Way In and on the Way Out

**Java**

```java
public class Event {
    private final List<String> attendees;

    public Event(List<String> attendees) {
        // Copy on the way IN
        this.attendees = List.copyOf(attendees);  // unmodifiable copy
    }

    public List<String> getAttendees() {
        // Already unmodifiable from List.copyOf, safe to return directly.
        // If using ArrayList internally, return Collections.unmodifiableList(attendees)
        // or new ArrayList<>(attendees) for a mutable copy.
        return attendees;
    }
}
```

**C++**

```cpp
class Event {
public:
    // Parameter is taken by value — caller's list is copied automatically
    explicit Event(std::vector<std::string> attendees)
        : attendees_(std::move(attendees)) {}

    // Return by const reference — caller cannot modify
    const std::vector<std::string>& attendees() const {
        return attendees_;
    }

    // Or return a copy if caller needs a mutable version
    std::vector<std::string> attendees_copy() const {
        return attendees_;
    }

private:
    std::vector<std::string> attendees_;
};
```

**Python**

```python
from dataclasses import dataclass, field

@dataclass
class Event:
    _attendees: list = field(default_factory=list)

    def __init__(self, attendees: list[str]):
        self._attendees = list(attendees)  # copy on the way in

    @property
    def attendees(self) -> tuple[str, ...]:
        return tuple(self._attendees)      # return immutable view

    def add_attendee(self, name: str):
        self._attendees.append(name)
```

### Unmodifiable Wrappers vs Deep Copies

| Strategy | What It Does | Pros | Cons |
|----------|-------------|------|------|
| **Unmodifiable wrapper** | `Collections.unmodifiableList(list)` | O(1), no allocation | Elements themselves still mutable; throws at runtime not compile time |
| **Shallow copy** | `new ArrayList<>(list)` | Isolates structural changes | Elements still shared — deep mutation leaks |
| **Deep copy** | Recursively clone each element | Full isolation | Expensive for large/deep graphs |
| **Immutable collection** | `List.of(...)`, `List.copyOf(...)` | Truly immutable, optimized | Java 9+ only; no null elements allowed |

**Rule of thumb**: Use `List.of()` / `List.copyOf()` in Java, `tuple` / `frozenset` in Python, and `const` references in C++. Resort to deep copies only when elements are mutable and you need full isolation.

---

## 4. Defensive Programming Techniques

Defensive programming ensures invalid data is caught immediately rather than corrupting state silently.

### Input Validation

**Java**

```java
public class User {
    private final String email;
    private final int age;

    public User(String email, int age) {
        // Null check
        this.email = Objects.requireNonNull(email, "email must not be null");

        // Format validation
        if (!email.contains("@")) {
            throw new IllegalArgumentException("Invalid email: " + email);
        }

        // Range check
        if (age < 0 || age > 150) {
            throw new IllegalArgumentException("Age out of range: " + age);
        }
        this.age = age;
    }
}
```

**C++**

```cpp
class User {
public:
    User(std::string email, int age) : email_(std::move(email)), age_(age) {
        if (email_.empty()) {
            throw std::invalid_argument("email must not be empty");
        }
        if (email_.find('@') == std::string::npos) {
            throw std::invalid_argument("Invalid email: " + email_);
        }
        if (age_ < 0 || age_ > 150) {
            throw std::out_of_range("Age out of range: " + std::to_string(age_));
        }
    }

private:
    std::string email_;
    int age_;
};
```

**Python**

```python
@dataclass(frozen=True)
class User:
    email: str
    age: int

    def __post_init__(self):
        if not self.email or "@" not in self.email:
            raise ValueError(f"Invalid email: {self.email}")
        if not (0 <= self.age <= 150):
            raise ValueError(f"Age out of range: {self.age}")
```

### Fail-Fast Principle

Detect errors at the earliest possible point. This minimizes the distance between cause and symptom.

```java
// BAD: silently ignores null, breaks later in some unrelated method
public void processOrder(Order order) {
    if (order == null) return;     // silent failure — caller never knows
    // ...
}

// GOOD: fail-fast — caller is immediately notified of the bug
public void processOrder(Order order) {
    Objects.requireNonNull(order, "order must not be null");
    // ...
}
```

### Making Invalid States Unrepresentable

Use the type system to make illegal states impossible to construct.

```java
// BAD: status can be any string, including invalid ones
public class Ticket {
    private String status;  // "open"? "OPEN"? "opne"? anything goes
}

// GOOD: only valid states exist
public enum TicketStatus { OPEN, IN_PROGRESS, RESOLVED, CLOSED }

public class Ticket {
    private TicketStatus status;  // compiler rejects invalid values
}
```

```java
// BAD: a discount percentage that could be negative or > 100
public void applyDiscount(int percent) { ... }

// GOOD: wrap in a validated type
public record Percentage(int value) {
    public Percentage {
        if (value < 0 || value > 100)
            throw new IllegalArgumentException("Percentage must be 0-100");
    }
}
public void applyDiscount(Percentage percent) { ... }  // impossible to pass invalid value
```

```python
# Python: NewType + validation
from dataclasses import dataclass

@dataclass(frozen=True)
class Percentage:
    value: int

    def __post_init__(self):
        if not (0 <= self.value <= 100):
            raise ValueError(f"Percentage must be 0-100, got {self.value}")

# Now applyDiscount(Percentage(-5)) raises immediately at construction
```

### Assertions vs Validation

| Aspect | Assertions | Validation |
|--------|-----------|------------|
| **Purpose** | Catch programmer errors (bugs) | Catch user/caller errors (bad input) |
| **When** | Development/testing | Always (including production) |
| **Disabled in prod?** | Often yes (Java `-ea`, C++ `NDEBUG`) | Never |
| **Example** | `assert list != null` | `if (list == null) throw ...` |
| **Failure means** | Bug in this code | Bad data from outside |

---

## 5. Thread Safety Through Immutability

### Why Immutable Objects Are Thread-Safe

A data race requires two conditions: (1) shared mutable state and (2) concurrent access without synchronization. Immutable objects eliminate condition (1) entirely.

```java
// MUTABLE — needs synchronization
public class MutableCounter {
    private int count = 0;

    public synchronized void increment() { count++; }       // lock needed
    public synchronized int getCount() { return count; }    // lock needed
}

// IMMUTABLE — no synchronization required
public record ImmutableCounter(int count) {
    public ImmutableCounter increment() {
        return new ImmutableCounter(count + 1);  // return new object
    }
    // Can be shared across any number of threads without locks
}
```

### Avoiding Synchronization Overhead

| Approach | Synchronization Needed? | Performance | Trade-off |
|----------|------------------------|-------------|-----------|
| Mutable + `synchronized` | Yes | Lock contention under high concurrency | Simple but slow |
| Mutable + `ConcurrentHashMap` | Partial (fine-grained) | Better than global lock | Complex API |
| Immutable + atomic swap | Only at swap point | Excellent read performance | Extra allocations |
| Fully immutable pipeline | None | Best for reads | Must create new objects for changes |

```java
// Atomic reference to immutable object — lock-free updates
private final AtomicReference<ImmutableConfig> config =
    new AtomicReference<>(new ImmutableConfig(defaults));

// Reader — no locking, always sees a consistent snapshot
ImmutableConfig current = config.get();

// Writer — atomic compare-and-swap
config.updateAndGet(c -> c.withTimeout(Duration.ofSeconds(30)));
```

### Persistent Data Structures

Persistent (or purely functional) data structures preserve previous versions when modified. They use **structural sharing** to avoid copying the entire structure.

```
Original list: A -> B -> C -> D

After "prepend X":
  New:      X -> A -> B -> C -> D
  Original:      A -> B -> C -> D    (unchanged, shares B->C->D)
```

**Real-world examples**:
- **Java**: Guava's `ImmutableList`, Vavr's `io.vavr.collection.List`
- **C++**: `immer` library for persistent vectors and maps
- **Python**: `pyrsistent` library (`pvector`, `pmap`)

```python
from pyrsistent import pvector, pmap

v1 = pvector([1, 2, 3])
v2 = v1.append(4)         # v1 is unchanged, v2 = [1, 2, 3, 4]
v3 = v1.set(0, 99)        # v1 is unchanged, v3 = [99, 2, 3]

m1 = pmap({"a": 1, "b": 2})
m2 = m1.set("c", 3)       # m1 is unchanged
```

---

## Common Interview Questions

**Q1: What are the rules for creating an immutable class in Java?**
Make the class `final`, all fields `private final`, no setters, defensive-copy mutable arguments in the constructor, return defensive copies (or unmodifiable views) from getters. Alternatively, use a `record` for simple cases.

**Q2: If I return `Collections.unmodifiableList(list)` from a getter, is my object fully immutable?**
Not necessarily. The unmodifiable wrapper prevents structural changes (add/remove), but if the elements themselves are mutable, callers can still mutate them. Also, if you kept a reference to the original `list`, code with that reference can still modify it, and changes will be visible through the wrapper. Use `List.copyOf()` or deep-copy to be safe.

**Q3: What is the difference between a value object and an entity?**
A value object has no identity and is defined entirely by its attributes (e.g., Money, Address). Two value objects with the same attributes are considered equal. An entity has a unique identity (e.g., a User ID) and equality is based on that identity alone, not its attributes. Value objects should be immutable; entities are typically mutable.

**Q4: When would you choose a deep copy over an unmodifiable wrapper?**
When the collection contains mutable elements that the caller should not be able to modify. For example, a `List<Date>` — an unmodifiable wrapper prevents `add`/`remove` but the caller can still call `date.setTime()` on the elements. A deep copy creates independent element instances. The trade-off is performance: deep copies are O(n) or worse, while wrappers are O(1).

**Q5: How does immutability help with concurrency?**
Immutable objects are inherently thread-safe because they cannot be modified after construction. No thread can observe a partially-updated state, so no locks or synchronization are needed for reads. Updates are done by creating new objects and swapping references atomically (e.g., using `AtomicReference`). This eliminates data races and deadlock risks.

**Q6: What is the fail-fast principle and why does it matter?**
Fail-fast means detecting and reporting errors immediately at the point they occur, rather than allowing them to propagate. For example, throwing `IllegalArgumentException` in a constructor when given invalid input, rather than storing the bad data and failing later during processing. This minimizes debugging time because the stack trace points directly to the root cause.

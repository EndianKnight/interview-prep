# API & Interface Design

Designing clean, intuitive, and evolvable APIs and interfaces is one of the most impactful skills a senior engineer can demonstrate in low-level design interviews.

---

## 1. Method Design

Good method design makes code self-documenting and reduces the cognitive burden on callers.

### Clear Naming: Verb + Noun

Method names should describe **what** the method does using a verb-noun pattern. Avoid vague names like `process()`, `handle()`, or `doStuff()`.

| Bad Name | Good Name | Why |
|----------|-----------|-----|
| `data()` | `fetchUserData()` | Verb clarifies action |
| `process()` | `validateOrder()` | Specific about what is processed |
| `run()` | `sendNotification()` | Describes exact behavior |
| `check()` | `isEligibleForDiscount()` | Boolean return conveyed by `is` prefix |

**Conventions by return type:**
- `getX()` / `findX()` — returns a value
- `isX()` / `hasX()` / `canX()` — returns boolean
- `createX()` / `buildX()` — factory/builder
- `toX()` — conversion (e.g., `toString()`, `toJson()`)

### Parameter Count: Keep It at 3 or Fewer

When a method needs more than three parameters, bundle them into a **parameter object**.

**Java**
```java
// Bad — too many parameters, easy to swap arguments
public Order createOrder(String customerId, String productId,
                         int quantity, double price,
                         String shippingAddress, String couponCode) { ... }

// Good — parameter object groups related data
public class OrderRequest {
    private final String customerId;
    private final String productId;
    private final int quantity;
    private final double price;
    private final String shippingAddress;
    private final String couponCode;
    // constructor, getters
}

public Order createOrder(OrderRequest request) { ... }
```

**C++**
```cpp
// Bad
Order createOrder(const std::string& customerId, const std::string& productId,
                  int quantity, double price,
                  const std::string& shippingAddress, const std::string& couponCode);

// Good
struct OrderRequest {
    std::string customerId;
    std::string productId;
    int quantity;
    double price;
    std::string shippingAddress;
    std::string couponCode;
};

Order createOrder(const OrderRequest& request);
```

**Python**
```python
# Bad
def create_order(customer_id, product_id, quantity, price, shipping_address, coupon_code):
    ...

# Good — use a dataclass as a parameter object
from dataclasses import dataclass
from typing import Optional

@dataclass
class OrderRequest:
    customer_id: str
    product_id: str
    quantity: int
    price: float
    shipping_address: str
    coupon_code: Optional[str] = None

def create_order(request: OrderRequest) -> Order:
    ...
```

### Return Types: Avoid Null

Returning `null` forces every caller to add null checks. Prefer `Optional` (Java), `std::optional` (C++), or `None` with explicit typing (Python).

| Scenario | Bad Return | Good Return |
|----------|-----------|-------------|
| Single item not found | `null` | `Optional<User>` / `std::optional<User>` |
| Collection with no results | `null` | Empty collection `List.of()` / `[]` |
| Operation success/failure | `null` on failure | `Result<T, E>` or throw exception |

**Java**
```java
// Bad — caller must null-check
public User findUser(String id) {
    return userMap.get(id);  // returns null if missing
}

// Good — forces caller to handle absence explicitly
public Optional<User> findUser(String id) {
    return Optional.ofNullable(userMap.get(id));
}

// Good — empty collection, never null
public List<Order> getOrdersByCustomer(String customerId) {
    List<Order> orders = orderRepository.findByCustomerId(customerId);
    return orders != null ? orders : Collections.emptyList();
}
```

**C++**
```cpp
// Bad
User* findUser(const std::string& id) {
    auto it = userMap.find(id);
    return (it != userMap.end()) ? &it->second : nullptr;
}

// Good — std::optional (C++17)
#include <optional>

std::optional<User> findUser(const std::string& id) {
    auto it = userMap.find(id);
    if (it != userMap.end()) {
        return it->second;
    }
    return std::nullopt;
}
```

**Python**
```python
from typing import Optional

# Bad — implicit None return
def find_user(user_id: str):
    return user_map.get(user_id)

# Good — explicit Optional type hint
def find_user(user_id: str) -> Optional[User]:
    return user_map.get(user_id)

# Good — return empty list, never None
def get_orders_by_customer(customer_id: str) -> list[Order]:
    orders = order_repository.find_by_customer_id(customer_id)
    return orders if orders else []
```

### Method Length: Single Level of Abstraction

Each method should operate at **one level of abstraction**. If a method mixes high-level orchestration with low-level details, extract the details into helper methods.

```java
// Bad — mixed levels of abstraction
public void processOrder(Order order) {
    // validation logic (low-level)
    if (order.getItems().isEmpty()) throw new IllegalArgumentException("No items");
    if (order.getTotal() < 0) throw new IllegalArgumentException("Negative total");

    // pricing logic (low-level)
    double discount = 0;
    if (order.getCoupon() != null) {
        discount = couponService.calculateDiscount(order.getCoupon());
    }
    order.applyDiscount(discount);

    // persistence (low-level)
    orderRepository.save(order);
    notificationService.sendConfirmation(order);
}

// Good — each call is at the same level of abstraction
public void processOrder(Order order) {
    validateOrder(order);
    applyDiscounts(order);
    saveAndNotify(order);
}
```

---

## 2. Interface Design Principles

### Program to Interfaces, Not Implementations

Depend on **abstractions** so that implementations can be swapped without changing client code.

**Java**
```java
// Bad — coupled to concrete implementation
public class OrderService {
    private MySqlOrderRepository repository = new MySqlOrderRepository();
}

// Good — depends on abstraction
public class OrderService {
    private final OrderRepository repository;

    public OrderService(OrderRepository repository) {
        this.repository = repository;
    }
}

public interface OrderRepository {
    void save(Order order);
    Optional<Order> findById(String id);
    List<Order> findByCustomerId(String customerId);
}
```

**C++**
```cpp
// Good — abstract base class as interface
class OrderRepository {
public:
    virtual ~OrderRepository() = default;
    virtual void save(const Order& order) = 0;
    virtual std::optional<Order> findById(const std::string& id) = 0;
    virtual std::vector<Order> findByCustomerId(const std::string& customerId) = 0;
};

class OrderService {
    std::unique_ptr<OrderRepository> repository_;
public:
    explicit OrderService(std::unique_ptr<OrderRepository> repo)
        : repository_(std::move(repo)) {}
};
```

**Python**
```python
from abc import ABC, abstractmethod

class OrderRepository(ABC):
    @abstractmethod
    def save(self, order: Order) -> None: ...

    @abstractmethod
    def find_by_id(self, order_id: str) -> Optional[Order]: ...

class OrderService:
    def __init__(self, repository: OrderRepository):
        self._repository = repository
```

### Thin Interfaces (Interface Segregation Principle)

Design **role-based** interfaces, not fat, entity-based ones. No client should be forced to depend on methods it does not use.

**Java**
```java
// Bad — fat interface forces implementors to handle unrelated methods
public interface SmartDevice {
    void turnOn();
    void turnOff();
    void print(Document doc);       // not all devices print
    void scan(Document doc);        // not all devices scan
    void fax(Document doc);         // not all devices fax
}

// Good — segregated, role-based interfaces
public interface Switchable {
    void turnOn();
    void turnOff();
}

public interface Printer {
    void print(Document doc);
}

public interface Scanner {
    void scan(Document doc);
}

// A multifunction device implements only what it supports
public class MultiFunctionPrinter implements Switchable, Printer, Scanner {
    // implement all three
}

public class SimplePrinter implements Switchable, Printer {
    // no scan or fax
}
```

### Default Methods (Java 8+)

Default methods allow you to **evolve interfaces** without breaking existing implementations. Use them sparingly for backward-compatible additions, not to turn interfaces into abstract classes.

```java
public interface OrderRepository {
    void save(Order order);
    Optional<Order> findById(String id);

    // Added in v2 — existing implementations are not broken
    default List<Order> findByStatus(OrderStatus status) {
        throw new UnsupportedOperationException("Not yet implemented");
    }

    // Utility default — genuinely belongs on the interface
    default boolean exists(String id) {
        return findById(id).isPresent();
    }
}
```

**When to use default methods:**
- Adding new methods to a widely-implemented interface
- Providing convenience methods derived from existing abstract methods
- Template method patterns within interfaces

**When NOT to use default methods:**
- Storing state (interfaces have no instance fields)
- Replacing abstract classes with complex shared logic

### Marker Interfaces vs Annotations

| Aspect | Marker Interface | Annotation |
|--------|-----------------|------------|
| Compile-time type checking | Yes (`if (obj instanceof Serializable)`) | No |
| Can carry metadata | No (no methods) | Yes (`@Retention`, `@Target`, values) |
| Limits applicability | Yes (only classes that implement it) | Yes (via `@Target`) |
| Examples | `Serializable`, `Cloneable` | `@Entity`, `@Deprecated`, `@Override` |
| Modern preference | Use when type checking is needed | Use for metadata / cross-cutting concerns |

```java
// Marker interface — enables compile-time type safety
public interface Auditable {}

public class AuditService {
    public void audit(Auditable entity) { // only accepts auditable types
        ...
    }
}

// Annotation — metadata, processed by framework
@Retention(RetentionPolicy.RUNTIME)
@Target(ElementType.TYPE)
public @interface Auditable {
    String tableName() default "";
}
```

---

## 3. Fluent Interfaces & Builder Pattern

### Method Chaining

Fluent interfaces return `this` (or a new instance) from each method, enabling a readable, declarative call chain.

**Java**
```java
public class QueryBuilder {
    private String table;
    private String whereClause;
    private String orderBy;
    private int limit;

    public QueryBuilder from(String table) {
        this.table = table;
        return this;
    }

    public QueryBuilder where(String clause) {
        this.whereClause = clause;
        return this;
    }

    public QueryBuilder orderBy(String column) {
        this.orderBy = column;
        return this;
    }

    public QueryBuilder limit(int limit) {
        this.limit = limit;
        return this;
    }

    public String build() {
        StringBuilder sb = new StringBuilder("SELECT * FROM " + table);
        if (whereClause != null) sb.append(" WHERE ").append(whereClause);
        if (orderBy != null) sb.append(" ORDER BY ").append(orderBy);
        if (limit > 0) sb.append(" LIMIT ").append(limit);
        return sb.toString();
    }
}

// Usage
String query = new QueryBuilder()
    .from("users")
    .where("age > 18")
    .orderBy("name")
    .limit(10)
    .build();
```

**C++**
```cpp
class QueryBuilder {
    std::string table_;
    std::string where_;
    std::string orderBy_;
    int limit_ = 0;

public:
    QueryBuilder& from(const std::string& table) {
        table_ = table;
        return *this;
    }

    QueryBuilder& where(const std::string& clause) {
        where_ = clause;
        return *this;
    }

    QueryBuilder& orderBy(const std::string& column) {
        orderBy_ = column;
        return *this;
    }

    QueryBuilder& limit(int n) {
        limit_ = n;
        return *this;
    }

    std::string build() const {
        std::string sql = "SELECT * FROM " + table_;
        if (!where_.empty()) sql += " WHERE " + where_;
        if (!orderBy_.empty()) sql += " ORDER BY " + orderBy_;
        if (limit_ > 0) sql += " LIMIT " + std::to_string(limit_);
        return sql;
    }
};

// Usage
auto query = QueryBuilder()
    .from("users")
    .where("age > 18")
    .orderBy("name")
    .limit(10)
    .build();
```

**Python**
```python
class QueryBuilder:
    def __init__(self):
        self._table = ""
        self._where = ""
        self._order_by = ""
        self._limit = 0

    def from_table(self, table: str) -> "QueryBuilder":
        self._table = table
        return self

    def where(self, clause: str) -> "QueryBuilder":
        self._where = clause
        return self

    def order_by(self, column: str) -> "QueryBuilder":
        self._order_by = column
        return self

    def limit(self, n: int) -> "QueryBuilder":
        self._limit = n
        return self

    def build(self) -> str:
        sql = f"SELECT * FROM {self._table}"
        if self._where:
            sql += f" WHERE {self._where}"
        if self._order_by:
            sql += f" ORDER BY {self._order_by}"
        if self._limit > 0:
            sql += f" LIMIT {self._limit}"
        return sql

# Usage
query = (QueryBuilder()
    .from_table("users")
    .where("age > 18")
    .order_by("name")
    .limit(10)
    .build())
```

### When Fluent Is Good vs Misleading

| Good Use Cases | Problematic Use Cases |
|---------------|----------------------|
| Builders / configuration objects | Domain objects with real business logic |
| Query DSLs | Methods with important side effects |
| Test data setup | When call order matters but is not enforced |
| Assertions (`assertThat(x).isNotNull().isEqualTo(y)`) | When return type changes mid-chain |

**Key rule:** Fluent APIs should be used for **configuration and construction**, not for orchestrating stateful business operations where the return value of each step matters.

### Builder Pattern for Complex Construction

The Builder pattern separates construction of a complex object from its representation, enforcing required fields and providing compile-time safety.

**Java**
```java
public class User {
    private final String id;          // required
    private final String name;        // required
    private final String email;       // optional
    private final int age;            // optional

    private User(Builder builder) {
        this.id = builder.id;
        this.name = builder.name;
        this.email = builder.email;
        this.age = builder.age;
    }

    public static class Builder {
        private final String id;      // required — set in constructor
        private final String name;    // required — set in constructor
        private String email = "";
        private int age = 0;

        public Builder(String id, String name) {
            this.id = id;
            this.name = name;
        }

        public Builder email(String email) {
            this.email = email;
            return this;
        }

        public Builder age(int age) {
            this.age = age;
            return this;
        }

        public User build() {
            return new User(this);
        }
    }
}

// Usage
User user = new User.Builder("u123", "Alice")
    .email("alice@example.com")
    .age(30)
    .build();
```

**C++**
```cpp
class User {
    std::string id_;
    std::string name_;
    std::string email_;
    int age_;

    User(std::string id, std::string name, std::string email, int age)
        : id_(std::move(id)), name_(std::move(name)),
          email_(std::move(email)), age_(age) {}

public:
    class Builder {
        std::string id_;
        std::string name_;
        std::string email_;
        int age_ = 0;

    public:
        Builder(std::string id, std::string name)
            : id_(std::move(id)), name_(std::move(name)) {}

        Builder& email(std::string email) {
            email_ = std::move(email);
            return *this;
        }

        Builder& age(int age) {
            age_ = age;
            return *this;
        }

        User build() {
            return User(id_, name_, email_, age_);
        }
    };
};

// Usage
auto user = User::Builder("u123", "Alice")
    .email("alice@example.com")
    .age(30)
    .build();
```

**Python**
```python
from dataclasses import dataclass, field

# Python approach: dataclass with defaults is often sufficient
@dataclass
class User:
    id: str
    name: str
    email: str = ""
    age: int = 0

# For more complex cases, use a builder
class UserBuilder:
    def __init__(self, user_id: str, name: str):
        self._id = user_id
        self._name = name
        self._email = ""
        self._age = 0

    def email(self, email: str) -> "UserBuilder":
        self._email = email
        return self

    def age(self, age: int) -> "UserBuilder":
        self._age = age
        return self

    def build(self) -> User:
        return User(id=self._id, name=self._name,
                    email=self._email, age=self._age)

# Usage
user = UserBuilder("u123", "Alice").email("alice@example.com").age(30).build()
```

---

## 4. Law of Demeter (Principle of Least Knowledge)

An object should only talk to its **immediate friends** — not to strangers accessed through a chain of calls.

### The Rule

A method `M` of object `O` should only call methods on:
1. `O` itself
2. `M`'s parameters
3. Objects created within `M`
4. `O`'s direct fields

### Train Wreck Anti-Pattern

```java
// Bad — "train wreck" violates Law of Demeter
String city = order.getCustomer().getAddress().getCity();

// The caller must know the internal structure of Order, Customer, AND Address.
// If Address changes its API, code that has nothing to do with Address breaks.
```

### Solutions

**1. Delegate Methods (tell, don't ask)**

**Java**
```java
// Good — Order exposes a delegate method
public class Order {
    private Customer customer;

    public String getShippingCity() {
        return customer.getShippingCity();
    }
}

public class Customer {
    private Address address;

    public String getShippingCity() {
        return address.getCity();
    }
}

// Caller only talks to its immediate friend
String city = order.getShippingCity();
```

**C++**
```cpp
class Order {
    Customer customer_;
public:
    std::string getShippingCity() const {
        return customer_.getShippingCity();
    }
};

class Customer {
    Address address_;
public:
    std::string getShippingCity() const {
        return address_.getCity();
    }
};

// Clean usage
std::string city = order.getShippingCity();
```

**Python**
```python
class Order:
    def __init__(self, customer: Customer):
        self._customer = customer

    def get_shipping_city(self) -> str:
        return self._customer.get_shipping_city()

class Customer:
    def __init__(self, address: Address):
        self._address = address

    def get_shipping_city(self) -> str:
        return self._address.city

# Clean usage
city = order.get_shipping_city()
```

**2. Tell, Don't Ask**

Instead of interrogating an object's internal state and making decisions externally, **tell** the object what to do and let it decide how.

```java
// Bad — asking for data and making decisions externally
if (account.getBalance() >= amount) {
    account.setBalance(account.getBalance() - amount);
}

// Good — tell the object what you want
account.withdraw(amount);  // Account handles validation internally
```

### When It Is OK to Break the Law of Demeter

- **Data structures / DTOs** — Plain data holders with no behavior (`order.address.city` in a DTO is acceptable).
- **Fluent APIs** — Method chains that return the same builder are not train wrecks because you are always talking to the same object.
- **Stream/LINQ pipelines** — `list.stream().filter(...).map(...).collect(...)` is a fluent transformation pipeline, not a navigation chain.

---

## 5. Design by Contract

Design by Contract (DbC), introduced by Bertrand Meyer, specifies precise obligations between a method and its callers.

### The Three Components

| Component | Who Is Responsible | What It Means |
|-----------|-------------------|---------------|
| **Precondition** | Caller | What must be true before calling the method |
| **Postcondition** | Method | What the method guarantees after execution |
| **Invariant** | Class | What must always be true about the object's state |

### Preconditions

**Java**
```java
public class BankAccount {
    private double balance;

    /**
     * @pre amount > 0 && amount <= balance
     * @post balance == old(balance) - amount
     */
    public void withdraw(double amount) {
        // Validate precondition
        if (amount <= 0) {
            throw new IllegalArgumentException("Amount must be positive");
        }
        if (amount > balance) {
            throw new IllegalArgumentException("Insufficient funds");
        }
        balance -= amount;
    }
}
```

**C++**
```cpp
class BankAccount {
    double balance_;
public:
    void withdraw(double amount) {
        // Precondition checks
        assert(amount > 0 && "Amount must be positive");
        assert(amount <= balance_ && "Insufficient funds");
        // Or throw for production code:
        if (amount <= 0 || amount > balance_) {
            throw std::invalid_argument("Invalid withdrawal amount");
        }
        balance_ -= amount;
    }
};
```

**Python**
```python
class BankAccount:
    def __init__(self, balance: float):
        self._balance = balance

    def withdraw(self, amount: float) -> None:
        """Withdraw funds.

        Pre: amount > 0 and amount <= balance
        Post: balance == old_balance - amount
        """
        if amount <= 0:
            raise ValueError("Amount must be positive")
        if amount > self._balance:
            raise ValueError("Insufficient funds")
        self._balance -= amount
```

### Postconditions and Invariants

```java
public class SortedList<T extends Comparable<T>> {
    private final List<T> items = new ArrayList<>();

    // Invariant: items is always sorted in ascending order

    public void add(T item) {
        // Precondition: item is not null
        Objects.requireNonNull(item, "Item must not be null");

        int index = Collections.binarySearch(items, item);
        if (index < 0) index = -(index + 1);
        items.add(index, item);

        // Postcondition: list still sorted (assert in debug)
        assert isSorted() : "Invariant violated: list is no longer sorted";
    }

    private boolean isSorted() {
        for (int i = 1; i < items.size(); i++) {
            if (items.get(i).compareTo(items.get(i - 1)) < 0) return false;
        }
        return true;
    }
}
```

### Defensive Programming vs Contract-Based Trust

| Aspect | Defensive Programming | Design by Contract |
|--------|----------------------|-------------------|
| Philosophy | Trust nobody, validate everything | Clearly define responsibilities |
| Where checks live | Inside every method, redundantly | At module/API boundaries |
| Performance | More overhead from repeated checks | Less overhead internally |
| Error discovery | Masks bugs by handling bad input silently | Fails fast, bug is in the caller |
| Best for | Public APIs, untrusted input | Internal code, team conventions |

**Practical guideline:** Use **defensive programming** at system boundaries (public APIs, user input, network data) and **Design by Contract** within trusted internal code.

---

## 6. Versioning & Evolution

### Backward Compatibility

When evolving an interface, some changes are safe and others are breaking.

| Change Type | Backward Compatible? | Reason |
|------------|---------------------|--------|
| Add new method with default implementation | Yes | Existing implementations unaffected |
| Add new method without default | **No** | All implementations must add it |
| Add optional parameter with default value | Yes | Existing callers still work |
| Change method signature (parameters/return type) | **No** | Callers break |
| Remove a method | **No** | Callers that use it break |
| Widen parameter type (`String` to `Object`) | Yes | Existing callers still match |
| Narrow return type (`Object` to `String`) | Yes | Existing callers get a more specific type |

### Deprecated Methods

Always provide a migration path when deprecating functionality.

**Java**
```java
public interface PaymentProcessor {

    /**
     * @deprecated Use {@link #processPayment(PaymentRequest)} instead.
     *             Will be removed in v3.0.
     */
    @Deprecated(since = "2.0", forRemoval = true)
    default boolean charge(String cardNumber, double amount) {
        return processPayment(new PaymentRequest(cardNumber, amount)).isSuccess();
    }

    PaymentResult processPayment(PaymentRequest request);
}
```

**C++**
```cpp
class PaymentProcessor {
public:
    // Deprecated: use processPayment(PaymentRequest) instead
    [[deprecated("Use processPayment(PaymentRequest) instead. Will be removed in v3.0")]]
    virtual bool charge(const std::string& cardNumber, double amount) {
        return processPayment(PaymentRequest{cardNumber, amount}).isSuccess();
    }

    virtual PaymentResult processPayment(const PaymentRequest& request) = 0;

    virtual ~PaymentProcessor() = default;
};
```

**Python**
```python
import warnings

class PaymentProcessor:
    def charge(self, card_number: str, amount: float) -> bool:
        """Deprecated: Use process_payment() instead. Will be removed in v3.0."""
        warnings.warn(
            "charge() is deprecated, use process_payment() instead",
            DeprecationWarning,
            stacklevel=2
        )
        result = self.process_payment(PaymentRequest(card_number, amount))
        return result.is_success

    def process_payment(self, request: PaymentRequest) -> PaymentResult:
        raise NotImplementedError
```

### Sealed Interfaces (Java 17+)

Sealed interfaces restrict which classes can implement them, giving the API designer full control over the type hierarchy. This enables exhaustive pattern matching and prevents unauthorized extensions.

```java
// Only these three classes can implement Shape
public sealed interface Shape permits Circle, Rectangle, Triangle {
    double area();
}

public record Circle(double radius) implements Shape {
    public double area() { return Math.PI * radius * radius; }
}

public record Rectangle(double width, double height) implements Shape {
    public double area() { return width * height; }
}

public record Triangle(double base, double height) implements Shape {
    public double area() { return 0.5 * base * height; }
}

// Exhaustive pattern matching (Java 21+)
public String describe(Shape shape) {
    return switch (shape) {
        case Circle c    -> "Circle with radius " + c.radius();
        case Rectangle r -> "Rectangle " + r.width() + "x" + r.height();
        case Triangle t  -> "Triangle with base " + t.base();
        // No default needed — compiler knows all cases are covered
    };
}
```

**When to seal an interface:**
- You own all implementations and want exhaustive handling (e.g., AST nodes, state machines, command types)
- You want to prevent external extension for safety or correctness reasons

**When NOT to seal:**
- The interface is a public extension point (e.g., plugin APIs, repository interfaces)

---

## Common Interview Questions

**Q1: Why should method parameters be limited to three or fewer?**

More parameters increase cognitive load, make call sites error-prone (especially with same-typed parameters that can be swapped silently), and make the method harder to test. When more data is needed, group related parameters into a parameter object, which also provides a natural place for validation and defaults.

**Q2: What is the Law of Demeter and how do you fix violations?**

The Law of Demeter states that an object should only call methods on its immediate collaborators, not on objects obtained by navigating through them (e.g., `a.getB().getC().doThing()`). Fix violations by creating delegate methods on intermediate objects so the caller only talks to its direct dependency. This reduces coupling and limits the blast radius of internal changes.

**Q3: When would you choose a marker interface over an annotation?**

Use a marker interface when you need **compile-time type safety** — for example, restricting a method parameter to only accept types that implement the marker (`void audit(Auditable entity)`). Use annotations when you need to attach **metadata** that is processed by frameworks at runtime (e.g., `@Entity`, `@Cacheable`) or when the marker needs to carry configuration values.

**Q4: How do you evolve a widely-used interface without breaking existing implementations?**

In Java, add new methods with `default` implementations so existing classes compile without changes. Follow semantic versioning. Deprecate old methods with `@Deprecated(forRemoval = true)` and clear migration documentation. Provide bridge implementations in the default methods that delegate to the new API. Remove deprecated methods only in major version bumps.

**Q5: What is Design by Contract and where should you apply defensive checks vs contract-based trust?**

Design by Contract assigns explicit responsibilities: the caller must satisfy preconditions, and the method guarantees postconditions and preserves class invariants. Apply defensive programming (validate everything) at **public API boundaries** where input is untrusted. Within internal/trusted code, rely on contracts — use assertions for preconditions so bugs surface immediately rather than being silently handled.

**Q6: When is a fluent interface appropriate, and when is it a bad idea?**

Fluent interfaces are ideal for **construction and configuration** scenarios — builders, query DSLs, test fixtures, and assertion libraries — where the chain reads declaratively and order does not matter. They are a bad idea when methods have **significant side effects**, when the return type changes mid-chain confusingly, or when call order is critical but the API does not enforce it, because errors surface at runtime rather than compile time.

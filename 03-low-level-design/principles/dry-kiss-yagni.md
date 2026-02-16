# DRY, KISS, YAGNI

Three foundational software design principles that guide day-to-day engineering decisions and come up frequently in senior-level design interviews.

---

## DRY — Don't Repeat Yourself

> "Every piece of knowledge must have a single, unambiguous, authoritative representation within a system." — Andy Hunt & Dave Thomas, *The Pragmatic Programmer*

### Code Duplication vs Knowledge Duplication

These are fundamentally different, and confusing them is a common mistake.

| Type | Description | Example |
|------|-------------|---------|
| **Code duplication** | Identical lines of code in multiple places | Two methods that both format a date string the same way |
| **Knowledge duplication** | The same business rule or domain concept encoded in multiple places | Validation logic for "valid email" scattered across controller, service, and frontend |
| **Accidental similarity** | Code that looks identical but represents different concepts | Two functions that both multiply by 1.1, but one is tax and the other is a tip — these should NOT be merged |

The key insight: **DRY is about knowledge, not code**. Two pieces of code can look identical but represent different domain concepts. Merging them creates a false coupling — when one concept changes, the other breaks.

### Rule of Three

Don't abstract on the first or second occurrence. Wait until you see the pattern three times before extracting.

1. **First time** — just write it.
2. **Second time** — note the duplication, but tolerate it.
3. **Third time** — now refactor. The pattern is confirmed.

This prevents premature abstraction while still catching real duplication.

### DRY Violation and Fix

**Violation — duplicated discount logic:**

**Java**
```java
// BAD: discount logic duplicated in two places
public class OrderService {
    public double calculateOnlinePrice(Item item) {
        double price = item.getBasePrice();
        if (item.getCategory() == Category.ELECTRONICS) {
            price *= 0.9;  // 10% off electronics
        }
        if (price > 100) {
            price -= 10;   // $10 off orders over $100
        }
        return price;
    }

    public double calculateInStorePrice(Item item) {
        double price = item.getBasePrice();
        if (item.getCategory() == Category.ELECTRONICS) {
            price *= 0.9;  // 10% off electronics — same rule!
        }
        if (price > 100) {
            price -= 10;   // $10 off orders over $100 — same rule!
        }
        return price + 2.0; // in-store surcharge
    }
}
```

**Fix — extract method:**

**Java**
```java
// GOOD: single source of truth for discount logic
public class OrderService {
    private double applyStandardDiscounts(Item item) {
        double price = item.getBasePrice();
        if (item.getCategory() == Category.ELECTRONICS) {
            price *= 0.9;
        }
        if (price > 100) {
            price -= 10;
        }
        return price;
    }

    public double calculateOnlinePrice(Item item) {
        return applyStandardDiscounts(item);
    }

    public double calculateInStorePrice(Item item) {
        return applyStandardDiscounts(item) + 2.0;
    }
}
```

**C++**
```cpp
// GOOD: extract shared logic into a private method
class OrderService {
private:
    double applyStandardDiscounts(const Item& item) {
        double price = item.getBasePrice();
        if (item.getCategory() == Category::ELECTRONICS) {
            price *= 0.9;
        }
        if (price > 100) {
            price -= 10;
        }
        return price;
    }

public:
    double calculateOnlinePrice(const Item& item) {
        return applyStandardDiscounts(item);
    }

    double calculateInStorePrice(const Item& item) {
        return applyStandardDiscounts(item) + 2.0;
    }
};
```

**Python**
```python
# GOOD: extract shared logic into a helper
class OrderService:
    def _apply_standard_discounts(self, item: Item) -> float:
        price = item.base_price
        if item.category == Category.ELECTRONICS:
            price *= 0.9
        if price > 100:
            price -= 10
        return price

    def calculate_online_price(self, item: Item) -> float:
        return self._apply_standard_discounts(item)

    def calculate_in_store_price(self, item: Item) -> float:
        return self._apply_standard_discounts(item) + 2.0
```

### DRY Techniques

| Technique | When to Use | Example |
|-----------|-------------|---------|
| **Extract Method** | Same logic in multiple methods of one class | Pull common calculation into a private method |
| **Extract Class** | Shared logic across multiple classes | Create a `DiscountCalculator` used by `OrderService` and `InvoiceService` |
| **Template Method Pattern** | Subclasses share an algorithm skeleton but differ in steps | Abstract base class defines the flow; subclasses override specific steps |
| **Utility / Helper** | Stateless, cross-cutting operations (formatting, validation) | `DateUtils.format(date, pattern)` |
| **Configuration / Constants** | Magic numbers or strings repeated everywhere | `MAX_RETRY_COUNT = 3` defined once |

---

## KISS — Keep It Simple, Stupid

> "Simplicity is the ultimate sophistication." — Leonardo da Vinci

The simplest solution that works is almost always the best. Complexity is a cost that compounds: it slows development, increases bugs, and makes onboarding harder.

### Signs of Over-Engineering

| Smell | Description |
|-------|-------------|
| **Premature abstraction** | Creating interfaces, factories, and strategy patterns for a single concrete implementation |
| **Speculative generality** | Adding extension points "in case we need them later" |
| **Gold plating** | Adding features nobody asked for because they seem cool |
| **Indirection overload** | 7 layers of abstraction to save a string to a database |
| **Framework-itis** | Building an internal framework when a simple function would do |

### Over-Engineered vs Simple

**Over-engineered — a factory-strategy-observer monstrosity to send a notification:**

**Java**
```java
// BAD: massive over-engineering for a single notification type
public interface NotificationStrategy {
    void send(String message, String recipient);
}

public class EmailNotificationStrategy implements NotificationStrategy {
    public void send(String message, String recipient) {
        // send email
    }
}

public interface NotificationFactory {
    NotificationStrategy create(String type);
}

public class DefaultNotificationFactory implements NotificationFactory {
    public NotificationStrategy create(String type) {
        if ("email".equals(type)) return new EmailNotificationStrategy();
        throw new IllegalArgumentException("Unknown type: " + type);
    }
}

public class NotificationService {
    private final NotificationFactory factory;

    public NotificationService(NotificationFactory factory) {
        this.factory = factory;
    }

    public void notify(String type, String message, String recipient) {
        NotificationStrategy strategy = factory.create(type);
        strategy.send(message, recipient);
    }
}
// 4 classes, 1 interface, to send an email.
```

**Simple — just send the email:**

**Java**
```java
// GOOD: simple and direct — only one notification type exists today
public class NotificationService {
    public void sendEmail(String message, String recipient) {
        // send email
    }
}
// When you actually need SMS and push, THEN add the abstraction.
```

**C++**
```cpp
// BAD: over-engineered
class NotificationStrategy {
public:
    virtual ~NotificationStrategy() = default;
    virtual void send(const std::string& msg, const std::string& to) = 0;
};

class EmailStrategy : public NotificationStrategy {
public:
    void send(const std::string& msg, const std::string& to) override {
        // send email
    }
};

// ... factory, service, etc. — 4+ classes for one email sender

// GOOD: simple and direct
class NotificationService {
public:
    void sendEmail(const std::string& message, const std::string& recipient) {
        // send email
    }
};
```

**Python**
```python
# BAD: over-engineered
class NotificationStrategy(ABC):
    @abstractmethod
    def send(self, message: str, recipient: str) -> None: ...

class EmailNotificationStrategy(NotificationStrategy):
    def send(self, message: str, recipient: str) -> None:
        ...  # send email

class NotificationFactory:
    def create(self, type_: str) -> NotificationStrategy:
        if type_ == "email":
            return EmailNotificationStrategy()
        raise ValueError(f"Unknown type: {type_}")

# GOOD: simple and direct
class NotificationService:
    def send_email(self, message: str, recipient: str) -> None:
        ...  # send email
```

### Simplicity in API Design

Good APIs follow KISS naturally:

```java
// BAD: caller needs to know too much
report.generate(true, false, null, 3, "pdf", true);

// GOOD: clear, intention-revealing methods
report.generatePdf();
report.generateCsv();

// Or if configuration is needed, use a builder:
Report.builder()
    .format(Format.PDF)
    .includeCharts(true)
    .pageLimit(3)
    .build();
```

**Rules of thumb for simple APIs:**
- Fewer parameters (aim for 0-3)
- Method names that describe what they do, not how
- Sensible defaults — the common case should require no configuration
- Fail fast with clear error messages

---

## YAGNI — You Aren't Gonna Need It

> "Always implement things when you actually need them, never when you just foresee that you need them." — Ron Jeffries

### The Cost of Unnecessary Code

Every line of code you write has ongoing costs:

| Cost | Description |
|------|-------------|
| **Development time** | Time spent building something nobody uses |
| **Maintenance burden** | Every feature must be tested, documented, and updated |
| **Cognitive load** | More code means more to understand for every developer |
| **Bug surface area** | Code that exists can have bugs, even if it is never called |
| **Decision paralysis** | Premature abstractions force future developers to understand unused extension points |

Studies consistently show that a large percentage of planned features are never used. Building them is pure waste.

### YAGNI Violation and Fix

**Violation — building for a future that may never come:**

**Java**
```java
// BAD: supporting 5 databases when you only use PostgreSQL
public interface DatabaseDriver {
    Connection connect(String url);
    void disconnect();
}

public class PostgresDriver implements DatabaseDriver { /* ... */ }
public class MySQLDriver implements DatabaseDriver { /* ... */ }
public class OracleDriver implements DatabaseDriver { /* ... */ }
public class SQLiteDriver implements DatabaseDriver { /* ... */ }
public class MongoDriver implements DatabaseDriver { /* ... */ }

public class DatabaseDriverFactory {
    public static DatabaseDriver create(String type) {
        return switch (type) {
            case "postgres" -> new PostgresDriver();
            case "mysql"    -> new MySQLDriver();
            case "oracle"   -> new OracleDriver();
            case "sqlite"   -> new SQLiteDriver();
            case "mongo"    -> new MongoDriver();
            default -> throw new IllegalArgumentException("Unknown: " + type);
        };
    }
}
```

**Fix — build only what you need today:**

**Java**
```java
// GOOD: you use PostgreSQL. Just use PostgreSQL.
public class PostgresDatabase {
    private final String url;

    public PostgresDatabase(String url) {
        this.url = url;
    }

    public Connection connect() {
        return DriverManager.getConnection(url);
    }
}
// If you ever need MySQL, add abstraction THEN.
```

**C++**
```cpp
// BAD: abstract factory for databases you'll never use
class DatabaseDriver {
public:
    virtual ~DatabaseDriver() = default;
    virtual void connect(const std::string& url) = 0;
};
class PostgresDriver : public DatabaseDriver { /* ... */ };
class MySQLDriver : public DatabaseDriver { /* ... */ };
class OracleDriver : public DatabaseDriver { /* ... */ };
// ... more drivers nobody asked for

// GOOD: just use what you need
class PostgresDatabase {
    std::string url_;
public:
    explicit PostgresDatabase(const std::string& url) : url_(url) {}
    void connect() { /* connect to postgres */ }
};
```

**Python**
```python
# BAD: abstract driver system for databases you don't use
class DatabaseDriver(ABC):
    @abstractmethod
    def connect(self, url: str) -> None: ...

class PostgresDriver(DatabaseDriver): ...
class MySQLDriver(DatabaseDriver): ...
class OracleDriver(DatabaseDriver): ...
class SQLiteDriver(DatabaseDriver): ...

# GOOD: just use what you need
class PostgresDatabase:
    def __init__(self, url: str):
        self.url = url

    def connect(self):
        return psycopg2.connect(self.url)
```

### YAGNI vs Planning Ahead — Where to Draw the Line

YAGNI does **not** mean "never think about the future." It means "don't **build** for the future until the future arrives."

| DO (Good Planning) | DON'T (YAGNI Violation) |
|---------------------|------------------------|
| Design clean interfaces so abstractions are easy to add later | Build the abstraction layer now for hypothetical future use |
| Choose a database that can scale to your projected needs | Build a custom database sharding framework before you have 1000 users |
| Write modular code that is easy to refactor | Build plugin systems, configuration DSLs, or extensibility hooks "just in case" |
| Use environment variables for deployment config | Build a feature-flag system before you have any features to flag |
| Keep coupling low so components can be swapped | Pre-build adapters for services you might integrate with someday |

**The litmus test:** Is someone asking for this right now? Is there a concrete user story or requirement? If the answer is no, don't build it.

---

## How They Work Together

The three principles are complementary but can create tension:

```
DRY  → eliminate redundancy    → "Don't say the same thing twice"
KISS → keep solutions simple   → "Don't make it complicated"
YAGNI → prevent over-building  → "Don't build what you don't need"
```

### Synergies

- **KISS + YAGNI** reinforce each other: not building unneeded features (YAGNI) naturally keeps the system simpler (KISS).
- **DRY + KISS**: eliminating duplication often makes code simpler because there is one place to look for each concept.
- All three push toward **lean, focused codebases** that are easy to understand and change.

### Tensions

The most common tension is **DRY vs KISS**:

**Java**
```java
// Two handlers with similar but not identical logic
public class OrderHandler {
    public void handle(Order order) {
        validate(order);
        log("Processing order: " + order.getId());
        save(order);
        notifyCustomer(order);
    }
}

public class ReturnHandler {
    public void handle(Return ret) {
        validate(ret);
        log("Processing return: " + ret.getId());
        save(ret);
        refundCustomer(ret);   // <-- different from above
    }
}
```

A DRY purist might extract a template method:

```java
// DRY but now more complex — is it worth it?
public abstract class TransactionHandler<T extends Transaction> {
    public final void handle(T transaction) {
        validate(transaction);
        log("Processing " + getType() + ": " + transaction.getId());
        save(transaction);
        postProcess(transaction);
    }

    protected abstract String getType();
    protected abstract void postProcess(T transaction);
}
```

**When DRY hurts KISS:** If the two handlers evolve independently (different validation rules, different logging needs), the shared abstraction becomes a straitjacket. Every change requires checking whether it breaks the other handler.

### Decision Framework

| Situation | Recommended Principle |
|-----------|-----------------------|
| Same business rule in 3+ places | **DRY** — extract it |
| Similar-looking code, different reasons to change | **KISS** — tolerate the duplication |
| Tempted to add an abstraction layer "for the future" | **YAGNI** — wait for a real need |
| Two classes doing the same thing, growing together | **DRY** — merge them |
| Two classes doing the same thing, growing apart | **KISS** — keep them separate |

**The golden rule:** When in doubt, prefer the simpler solution. You can always refactor later when requirements are clear. You cannot easily undo an over-engineered abstraction that has spread through the codebase.

---

## Common Interview Questions

**1. What is DRY and why is it important?**

DRY (Don't Repeat Yourself) states that every piece of knowledge should have a single, authoritative representation in the system. It is important because duplicated logic means bugs must be fixed in multiple places, and changes to business rules require hunting down every copy. DRY reduces maintenance costs and the risk of inconsistencies.

**2. Is all code duplication a DRY violation?**

No. DRY is about **knowledge** duplication, not **code** duplication. If two pieces of code look identical but represent different domain concepts (e.g., tax calculation and tip calculation both happen to multiply by the same factor), merging them would create a false coupling. When one concept changes, the other would break. The Rule of Three also suggests tolerating duplication until a pattern is confirmed.

**3. Give an example of a KISS violation and how you would fix it.**

A common KISS violation is building a factory + strategy + observer pattern to handle a single concrete implementation. For instance, creating `NotificationStrategy`, `NotificationFactory`, and `NotificationService` when the system only sends emails. The fix is a single `NotificationService` class with a `sendEmail()` method. Add abstractions only when a second notification channel is actually needed.

**4. How do you decide whether something is YAGNI or legitimate planning ahead?**

The key question is: "Is there a concrete requirement for this right now?" Good planning means writing clean, modular code that is easy to extend later. YAGNI violations are building the extension itself before it is needed. For example, using dependency injection (good planning) vs building a plugin system nobody has asked for (YAGNI). If you cannot point to a user story or a ticket, do not build it.

**5. What happens when DRY and KISS conflict?**

DRY can push toward abstractions (extract base class, template method) that make code harder to follow, violating KISS. The resolution depends on context: if the duplicated code changes together for the same reasons, DRY wins — extract it. If the code looks similar but evolves independently, KISS wins — keep it separate and tolerate the duplication. Coupling two unrelated concepts to avoid duplication is worse than the duplication itself.

**6. How would you apply these principles in a system design interview?**

In system design, DRY appears as shared libraries or services (e.g., a single auth service rather than every microservice implementing its own auth). KISS means choosing proven, simple architectures over clever custom solutions (e.g., use a message queue instead of building your own pub/sub). YAGNI means starting with a monolith or simple architecture and adding complexity (caching, sharding, microservices) only when scale demands it. All three principles push toward incremental complexity driven by real needs.

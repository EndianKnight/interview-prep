# SOLID Principles

The five foundational design principles for writing maintainable, extensible, and testable object-oriented software -- coined by Robert C. Martin (Uncle Bob).

---

## Overview

| Letter | Principle | One-Liner |
|--------|-----------|-----------|
| **S** | Single Responsibility | A class should have only one reason to change |
| **O** | Open/Closed | Open for extension, closed for modification |
| **L** | Liskov Substitution | Subtypes must be substitutable for their base types |
| **I** | Interface Segregation | No client should depend on methods it does not use |
| **D** | Dependency Inversion | Depend on abstractions, not concretions |

---

## S -- Single Responsibility Principle (SRP)

> "A class should have only one reason to change." -- Robert C. Martin

A class (or module) should be responsible to **one and only one actor**. When a class handles multiple concerns, a change to one concern risks breaking the other.

### Violation: Class Handles Validation AND Persistence

**Java**
```java
// BAD: UserService validates AND persists -- two reasons to change
public class UserService {
    public boolean validate(String email) {
        return email != null && email.contains("@");
    }

    public void saveToDatabase(User user) {
        Connection conn = DriverManager.getConnection("jdbc:mysql://localhost/db");
        PreparedStatement stmt = conn.prepareStatement("INSERT INTO users VALUES (?, ?)");
        stmt.setString(1, user.getName());
        stmt.setString(2, user.getEmail());
        stmt.executeUpdate();
    }
}
```

**C++**
```cpp
// BAD: UserService handles validation AND persistence
class UserService {
public:
    bool validate(const std::string& email) {
        return !email.empty() && email.find('@') != std::string::npos;
    }

    void saveToDatabase(const User& user) {
        // Direct DB logic here -- tightly coupled
        auto conn = mysql_init(nullptr);
        mysql_real_connect(conn, "localhost", "root", "", "db", 0, nullptr, 0);
        std::string query = "INSERT INTO users VALUES ('" + user.name + "')";
        mysql_query(conn, query.c_str());
    }
};
```

**Python**
```python
# BAD: UserService handles validation AND persistence
class UserService:
    def validate(self, email: str) -> bool:
        return email is not None and "@" in email

    def save_to_database(self, user: "User") -> None:
        import sqlite3
        conn = sqlite3.connect("users.db")
        conn.execute("INSERT INTO users VALUES (?, ?)", (user.name, user.email))
        conn.commit()
```

### Fix: Split Into Focused Classes

**Java**
```java
// GOOD: Each class has a single responsibility
public class EmailValidator {
    public boolean isValid(String email) {
        return email != null && email.contains("@");
    }
}

public class UserRepository {
    public void save(User user) {
        Connection conn = DriverManager.getConnection("jdbc:mysql://localhost/db");
        PreparedStatement stmt = conn.prepareStatement("INSERT INTO users VALUES (?, ?)");
        stmt.setString(1, user.getName());
        stmt.setString(2, user.getEmail());
        stmt.executeUpdate();
    }
}

public class UserService {
    private final EmailValidator validator;
    private final UserRepository repository;

    public UserService(EmailValidator validator, UserRepository repository) {
        this.validator = validator;
        this.repository = repository;
    }

    public void registerUser(User user) {
        if (!validator.isValid(user.getEmail())) {
            throw new IllegalArgumentException("Invalid email");
        }
        repository.save(user);
    }
}
```

**C++**
```cpp
// GOOD: Separate validator, repository, and service
class EmailValidator {
public:
    bool isValid(const std::string& email) const {
        return !email.empty() && email.find('@') != std::string::npos;
    }
};

class UserRepository {
public:
    void save(const User& user) {
        // Database logic isolated here
    }
};

class UserService {
    EmailValidator validator_;
    UserRepository repository_;
public:
    UserService(EmailValidator validator, UserRepository repository)
        : validator_(std::move(validator)), repository_(std::move(repository)) {}

    void registerUser(const User& user) {
        if (!validator_.isValid(user.email)) {
            throw std::invalid_argument("Invalid email");
        }
        repository_.save(user);
    }
};
```

**Python**
```python
# GOOD: Each class has one job
class EmailValidator:
    def is_valid(self, email: str) -> bool:
        return email is not None and "@" in email

class UserRepository:
    def save(self, user: "User") -> None:
        import sqlite3
        conn = sqlite3.connect("users.db")
        conn.execute("INSERT INTO users VALUES (?, ?)", (user.name, user.email))
        conn.commit()

class UserService:
    def __init__(self, validator: EmailValidator, repository: UserRepository):
        self.validator = validator
        self.repository = repository

    def register_user(self, user: "User") -> None:
        if not self.validator.is_valid(user.email):
            raise ValueError("Invalid email")
        self.repository.save(user)
```

### SRP Summary

| Aspect | Violation | Fix |
|--------|-----------|-----|
| Class count | 1 "God class" | Multiple focused classes |
| Reason to change | Multiple (validation rules, DB schema) | One per class |
| Testability | Hard -- must mock DB to test validation | Easy -- test validator in isolation |
| Reusability | Low -- can't reuse validation without DB code | High -- validator usable anywhere |

---

## O -- Open/Closed Principle (OCP)

> "Software entities should be open for extension but closed for modification." -- Bertrand Meyer

You should be able to add new behavior **without changing existing code**. This is typically achieved through polymorphism, the strategy pattern, or dependency injection.

### Violation: Growing Switch/If-Else for New Types

**Java**
```java
// BAD: Adding a new shape requires modifying this method
public class AreaCalculator {
    public double calculateArea(Object shape) {
        if (shape instanceof Circle c) {
            return Math.PI * c.getRadius() * c.getRadius();
        } else if (shape instanceof Rectangle r) {
            return r.getWidth() * r.getHeight();
        }
        // Every new shape = modify this class
        throw new IllegalArgumentException("Unknown shape");
    }
}
```

**C++**
```cpp
// BAD: Must modify function for every new shape
double calculateArea(const Shape& shape) {
    if (shape.type == "circle") {
        return M_PI * shape.radius * shape.radius;
    } else if (shape.type == "rectangle") {
        return shape.width * shape.height;
    }
    throw std::invalid_argument("Unknown shape");
}
```

**Python**
```python
# BAD: if-elif chain grows with every new shape
def calculate_area(shape) -> float:
    if isinstance(shape, Circle):
        return math.pi * shape.radius ** 2
    elif isinstance(shape, Rectangle):
        return shape.width * shape.height
    # Must modify this function for Triangle, Hexagon, etc.
    raise ValueError("Unknown shape")
```

### Fix: Use Polymorphism

**Java**
```java
// GOOD: New shapes extend without modifying existing code
public interface Shape {
    double area();
}

public class Circle implements Shape {
    private final double radius;
    public Circle(double radius) { this.radius = radius; }

    @Override
    public double area() {
        return Math.PI * radius * radius;
    }
}

public class Rectangle implements Shape {
    private final double width, height;
    public Rectangle(double width, double height) {
        this.width = width;
        this.height = height;
    }

    @Override
    public double area() {
        return width * height;
    }
}

// Adding Triangle requires ZERO changes to existing classes
public class Triangle implements Shape {
    private final double base, height;
    public Triangle(double base, double height) {
        this.base = base;
        this.height = height;
    }

    @Override
    public double area() {
        return 0.5 * base * height;
    }
}

// Calculator works with any Shape -- closed for modification
public class AreaCalculator {
    public double totalArea(List<Shape> shapes) {
        return shapes.stream().mapToDouble(Shape::area).sum();
    }
}
```

**C++**
```cpp
// GOOD: Virtual dispatch -- extend by adding new subclasses
class Shape {
public:
    virtual ~Shape() = default;
    virtual double area() const = 0;
};

class Circle : public Shape {
    double radius_;
public:
    explicit Circle(double r) : radius_(r) {}
    double area() const override { return M_PI * radius_ * radius_; }
};

class Rectangle : public Shape {
    double width_, height_;
public:
    Rectangle(double w, double h) : width_(w), height_(h) {}
    double area() const override { return width_ * height_; }
};

// Adding a new shape -- no existing code changes
class Triangle : public Shape {
    double base_, height_;
public:
    Triangle(double b, double h) : base_(b), height_(h) {}
    double area() const override { return 0.5 * base_ * height_; }
};

double totalArea(const std::vector<std::unique_ptr<Shape>>& shapes) {
    double sum = 0;
    for (const auto& s : shapes) sum += s->area();
    return sum;
}
```

**Python**
```python
# GOOD: Polymorphism via abstract base class
from abc import ABC, abstractmethod
import math

class Shape(ABC):
    @abstractmethod
    def area(self) -> float: ...

class Circle(Shape):
    def __init__(self, radius: float):
        self.radius = radius
    def area(self) -> float:
        return math.pi * self.radius ** 2

class Rectangle(Shape):
    def __init__(self, width: float, height: float):
        self.width = width
        self.height = height
    def area(self) -> float:
        return self.width * self.height

# New shape -- no changes to existing code
class Triangle(Shape):
    def __init__(self, base: float, height: float):
        self.base = base
        self.height = height
    def area(self) -> float:
        return 0.5 * self.base * self.height

def total_area(shapes: list[Shape]) -> float:
    return sum(s.area() for s in shapes)
```

### OCP Summary

| Aspect | Violation | Fix |
|--------|-----------|-----|
| Adding new type | Modify existing if/switch | Add a new class |
| Risk of regression | High -- touching shared code | Low -- existing code untouched |
| Pattern used | Procedural type checking | Polymorphism / Strategy |
| Testing | Must re-test entire function | Only test the new class |

---

## L -- Liskov Substitution Principle (LSP)

> "If S is a subtype of T, then objects of type T may be replaced with objects of type S without altering the correctness of the program." -- Barbara Liskov

Subtypes must honor the **contract** of their base type. Inheritance should model "is-a" in terms of behavior, not just taxonomy.

### Contract Rules

| Rule | Meaning |
|------|---------|
| **Preconditions** | A subtype must not strengthen preconditions (must not demand more) |
| **Postconditions** | A subtype must not weaken postconditions (must deliver at least as much) |
| **Invariants** | A subtype must preserve all invariants of the base type |
| **History constraint** | A subtype must not allow state changes the base type would not allow |

### Classic Violation: Square Extending Rectangle

**Java**
```java
// BAD: Square breaks Rectangle's behavioral contract
public class Rectangle {
    protected int width;
    protected int height;

    public void setWidth(int w)  { this.width = w; }
    public void setHeight(int h) { this.height = h; }
    public int getWidth()  { return width; }
    public int getHeight() { return height; }
    public int area() { return width * height; }
}

public class Square extends Rectangle {
    // Must keep width == height, so override both setters
    @Override
    public void setWidth(int w) {
        this.width = w;
        this.height = w;  // Side effect -- violates Rectangle's contract
    }

    @Override
    public void setHeight(int h) {
        this.width = h;
        this.height = h;
    }
}

// Client code that breaks with Square
public void resize(Rectangle r) {
    r.setWidth(5);
    r.setHeight(10);
    assert r.area() == 50;  // FAILS for Square -- area is 100!
}
```

**C++**
```cpp
// BAD: Square violates Rectangle's behavioral contract
class Rectangle {
protected:
    int width_, height_;
public:
    virtual void setWidth(int w)  { width_ = w; }
    virtual void setHeight(int h) { height_ = h; }
    int area() const { return width_ * height_; }
};

class Square : public Rectangle {
public:
    void setWidth(int w) override  { width_ = w; height_ = w; }
    void setHeight(int h) override { width_ = h; height_ = h; }
};

void resize(Rectangle& r) {
    r.setWidth(5);
    r.setHeight(10);
    assert(r.area() == 50);  // FAILS for Square!
}
```

**Python**
```python
# BAD: Square breaks Rectangle's contract
class Rectangle:
    def __init__(self, width: int, height: int):
        self._width = width
        self._height = height

    @property
    def width(self) -> int:
        return self._width

    @width.setter
    def width(self, value: int):
        self._width = value

    @property
    def height(self) -> int:
        return self._height

    @height.setter
    def height(self, value: int):
        self._height = value

    def area(self) -> int:
        return self._width * self._height

class Square(Rectangle):
    def __init__(self, side: int):
        super().__init__(side, side)

    @Rectangle.width.setter
    def width(self, value: int):
        self._width = value
        self._height = value  # Side effect

    @Rectangle.height.setter
    def height(self, value: int):
        self._width = value
        self._height = value

def resize(r: Rectangle):
    r.width = 5
    r.height = 10
    assert r.area() == 50  # FAILS for Square -- area is 100!
```

### Fix: Use a Common Abstraction

**Java**
```java
// GOOD: Separate types with a common interface
public interface Shape {
    int area();
}

public class Rectangle implements Shape {
    private final int width, height;
    public Rectangle(int w, int h) { this.width = w; this.height = h; }
    public int area() { return width * height; }
}

public class Square implements Shape {
    private final int side;
    public Square(int s) { this.side = s; }
    public int area() { return side * side; }
}
```

**C++**
```cpp
// GOOD: No inheritance relationship between Square and Rectangle
class Shape {
public:
    virtual ~Shape() = default;
    virtual int area() const = 0;
};

class Rectangle : public Shape {
    int width_, height_;
public:
    Rectangle(int w, int h) : width_(w), height_(h) {}
    int area() const override { return width_ * height_; }
};

class Square : public Shape {
    int side_;
public:
    explicit Square(int s) : side_(s) {}
    int area() const override { return side_ * side_; }
};
```

**Python**
```python
# GOOD: Immutable value objects, no problematic inheritance
from abc import ABC, abstractmethod

class Shape(ABC):
    @abstractmethod
    def area(self) -> int: ...

class Rectangle(Shape):
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
    def area(self) -> int:
        return self.width * self.height

class Square(Shape):
    def __init__(self, side: int):
        self.side = side
    def area(self) -> int:
        return self.side * self.side
```

### LSP Summary

| Aspect | Violation | Fix |
|--------|-----------|-----|
| Hierarchy | Square extends Rectangle | Both implement Shape |
| Contract | `setWidth`/`setHeight` have side effects in Square | Immutable or independent types |
| Substitutability | Client code breaks with Square | Any Shape works correctly |
| Root cause | Inheritance based on "is-a" taxonomy, not behavior | Inheritance based on behavioral contract |

---

## I -- Interface Segregation Principle (ISP)

> "No client should be forced to depend on methods it does not use." -- Robert C. Martin

Fat interfaces force implementors to provide stubs or throw exceptions for irrelevant methods. Split them into **thin, role-specific** interfaces so each client depends only on what it needs.

### Violation: Fat Worker Interface

**Java**
```java
// BAD: Robot doesn't eat or sleep, but must implement these
public interface Worker {
    void work();
    void eat();
    void sleep();
}

public class HumanWorker implements Worker {
    public void work()  { System.out.println("Working"); }
    public void eat()   { System.out.println("Eating"); }
    public void sleep() { System.out.println("Sleeping"); }
}

public class RobotWorker implements Worker {
    public void work()  { System.out.println("Working"); }
    public void eat()   { /* Not applicable! */ throw new UnsupportedOperationException(); }
    public void sleep() { /* Not applicable! */ throw new UnsupportedOperationException(); }
}
```

**C++**
```cpp
// BAD: Robot forced to implement eat() and sleep()
class Worker {
public:
    virtual ~Worker() = default;
    virtual void work() = 0;
    virtual void eat() = 0;
    virtual void sleep() = 0;
};

class RobotWorker : public Worker {
public:
    void work() override  { std::cout << "Working\n"; }
    void eat() override   { throw std::logic_error("Robots don't eat"); }
    void sleep() override { throw std::logic_error("Robots don't sleep"); }
};
```

**Python**
```python
# BAD: Robot must implement eat() and sleep()
from abc import ABC, abstractmethod

class Worker(ABC):
    @abstractmethod
    def work(self): ...
    @abstractmethod
    def eat(self): ...
    @abstractmethod
    def sleep(self): ...

class RobotWorker(Worker):
    def work(self):  print("Working")
    def eat(self):   raise NotImplementedError("Robots don't eat")
    def sleep(self): raise NotImplementedError("Robots don't sleep")
```

### Fix: Thin, Role-Specific Interfaces

**Java**
```java
// GOOD: Segregated interfaces -- each client depends only on what it needs
public interface Workable {
    void work();
}

public interface Feedable {
    void eat();
}

public interface Sleepable {
    void sleep();
}

public class HumanWorker implements Workable, Feedable, Sleepable {
    public void work()  { System.out.println("Working"); }
    public void eat()   { System.out.println("Eating"); }
    public void sleep() { System.out.println("Sleeping"); }
}

public class RobotWorker implements Workable {
    public void work() { System.out.println("Working"); }
    // No eat() or sleep() -- not needed, not implemented
}

// Client code depends only on what it uses
public class WorkScheduler {
    public void assign(Workable worker) {
        worker.work();  // Works for both Human and Robot
    }
}
```

**C++**
```cpp
// GOOD: Small, focused interfaces
class Workable {
public:
    virtual ~Workable() = default;
    virtual void work() = 0;
};

class Feedable {
public:
    virtual ~Feedable() = default;
    virtual void eat() = 0;
};

class Sleepable {
public:
    virtual ~Sleepable() = default;
    virtual void sleep() = 0;
};

class HumanWorker : public Workable, public Feedable, public Sleepable {
public:
    void work() override  { std::cout << "Working\n"; }
    void eat() override   { std::cout << "Eating\n"; }
    void sleep() override { std::cout << "Sleeping\n"; }
};

class RobotWorker : public Workable {
public:
    void work() override { std::cout << "Working\n"; }
};
```

**Python**
```python
# GOOD: Segregated ABCs
from abc import ABC, abstractmethod

class Workable(ABC):
    @abstractmethod
    def work(self): ...

class Feedable(ABC):
    @abstractmethod
    def eat(self): ...

class Sleepable(ABC):
    @abstractmethod
    def sleep(self): ...

class HumanWorker(Workable, Feedable, Sleepable):
    def work(self):  print("Working")
    def eat(self):   print("Eating")
    def sleep(self): print("Sleeping")

class RobotWorker(Workable):
    def work(self): print("Working")
    # Clean -- no forced empty methods
```

### ISP Summary

| Aspect | Violation | Fix |
|--------|-----------|-----|
| Interface size | 1 fat interface with N methods | N small interfaces, 1 method each |
| Implementors | Forced to stub/throw for unused methods | Only implement what applies |
| Coupling | Client depends on entire interface | Client depends on the slice it uses |
| Adding a concern | Modify the fat interface (breaks all implementors) | Add a new interface (no impact) |

---

## D -- Dependency Inversion Principle (DIP)

> "High-level modules should not depend on low-level modules. Both should depend on abstractions." -- Robert C. Martin

The principle has two parts:
1. High-level modules should not import anything from low-level modules -- both should depend on abstractions.
2. Abstractions should not depend on details. Details should depend on abstractions.

### Dependency Injection Types

| Type | Mechanism | When to Use |
|------|-----------|-------------|
| **Constructor injection** | Pass dependencies via constructor | Default choice -- immutable, explicit |
| **Setter injection** | Set dependencies via setter methods | Optional dependencies, reconfiguration |
| **Interface injection** | Implement an injector interface | Rare -- used in some frameworks |

### Violation: High-Level Depends on Concrete Low-Level

**Java**
```java
// BAD: NotificationService directly depends on concrete EmailSender
public class EmailSender {
    public void send(String message) {
        // SMTP logic
        System.out.println("Email: " + message);
    }
}

public class NotificationService {
    private final EmailSender emailSender = new EmailSender();  // Hardcoded dependency

    public void notifyUser(String message) {
        emailSender.send(message);  // Can't switch to SMS without modifying this class
    }
}
```

**C++**
```cpp
// BAD: High-level directly instantiates low-level
class EmailSender {
public:
    void send(const std::string& msg) {
        std::cout << "Email: " << msg << "\n";
    }
};

class NotificationService {
    EmailSender sender_;  // Concrete dependency -- can't swap
public:
    void notifyUser(const std::string& msg) {
        sender_.send(msg);
    }
};
```

**Python**
```python
# BAD: NotificationService tightly coupled to EmailSender
class EmailSender:
    def send(self, message: str) -> None:
        print(f"Email: {message}")

class NotificationService:
    def __init__(self):
        self.sender = EmailSender()  # Hardcoded -- can't test or swap

    def notify_user(self, message: str) -> None:
        self.sender.send(message)
```

### Fix: Depend on Abstractions + Constructor Injection

**Java**
```java
// GOOD: Both high-level and low-level depend on an abstraction
public interface MessageSender {
    void send(String message);
}

public class EmailSender implements MessageSender {
    @Override
    public void send(String message) {
        System.out.println("Email: " + message);
    }
}

public class SmsSender implements MessageSender {
    @Override
    public void send(String message) {
        System.out.println("SMS: " + message);
    }
}

public class NotificationService {
    private final MessageSender sender;  // Depends on abstraction

    public NotificationService(MessageSender sender) {  // Constructor injection
        this.sender = sender;
    }

    public void notifyUser(String message) {
        sender.send(message);
    }
}

// Usage -- swap implementations freely
NotificationService emailNotifier = new NotificationService(new EmailSender());
NotificationService smsNotifier   = new NotificationService(new SmsSender());
```

**C++**
```cpp
// GOOD: Depend on abstract interface, inject via constructor
class MessageSender {
public:
    virtual ~MessageSender() = default;
    virtual void send(const std::string& msg) = 0;
};

class EmailSender : public MessageSender {
public:
    void send(const std::string& msg) override {
        std::cout << "Email: " << msg << "\n";
    }
};

class SmsSender : public MessageSender {
public:
    void send(const std::string& msg) override {
        std::cout << "SMS: " << msg << "\n";
    }
};

class NotificationService {
    std::unique_ptr<MessageSender> sender_;
public:
    explicit NotificationService(std::unique_ptr<MessageSender> sender)
        : sender_(std::move(sender)) {}

    void notifyUser(const std::string& msg) {
        sender_->send(msg);
    }
};

// Usage
auto service = NotificationService(std::make_unique<SmsSender>());
service.notifyUser("Hello!");
```

**Python**
```python
# GOOD: Depend on abstraction, inject via constructor
from abc import ABC, abstractmethod

class MessageSender(ABC):
    @abstractmethod
    def send(self, message: str) -> None: ...

class EmailSender(MessageSender):
    def send(self, message: str) -> None:
        print(f"Email: {message}")

class SmsSender(MessageSender):
    def send(self, message: str) -> None:
        print(f"SMS: {message}")

class NotificationService:
    def __init__(self, sender: MessageSender):  # Constructor injection
        self.sender = sender

    def notify_user(self, message: str) -> None:
        self.sender.send(message)

# Usage -- easily testable and swappable
service = NotificationService(SmsSender())
service.notify_user("Hello!")
```

### IoC Containers

In real-world applications, **Inversion of Control (IoC) containers** automate dependency wiring:

| Language | Popular IoC / DI Frameworks |
|----------|-----------------------------|
| Java | Spring (most common), Guice, Dagger |
| C++ | Boost.DI, fruit |
| Python | dependency-injector, injector |

**Spring example (Java):**
```java
@Service
public class NotificationService {
    private final MessageSender sender;

    @Autowired  // Spring resolves the concrete implementation
    public NotificationService(MessageSender sender) {
        this.sender = sender;
    }
}

@Component
public class EmailSender implements MessageSender {
    public void send(String message) { /* ... */ }
}
```

### DIP Summary

| Aspect | Violation | Fix |
|--------|-----------|-----|
| Dependency direction | High-level -> concrete low-level | Both -> abstraction |
| Swappability | Must modify high-level to change impl | Inject any implementation |
| Testability | Hard -- can't mock concrete class easily | Easy -- inject a mock/stub |
| Framework support | Manual wiring | IoC container automates it |

---

## How SOLID Principles Relate to Each Other

| Relationship | Explanation |
|-------------|-------------|
| SRP + ISP | SRP focuses classes; ISP focuses interfaces. Both fight bloat. |
| OCP + DIP | DIP enables OCP -- depending on abstractions lets you extend via new implementations. |
| LSP + OCP | If subtypes violate LSP, clients add type-checks, violating OCP. |
| ISP + DIP | Thin interfaces make it easier to depend on the right abstraction. |

---

## Common Interview Questions

**1. Explain the Single Responsibility Principle with an example.**

A class should have only one reason to change. For example, a `UserService` that both validates emails and writes to a database has two reasons to change (validation rules, DB schema). Split it into `EmailValidator` and `UserRepository` so each class has a single, focused responsibility. This improves testability and reduces the blast radius of changes.

**2. How does the Open/Closed Principle reduce risk in production systems?**

OCP means adding new features by writing new code (new classes/modules) rather than modifying existing, tested code. For example, adding a new payment method as a new `PaymentProcessor` implementation requires zero changes to the payment orchestration logic. This eliminates regression risk in code paths that are already in production.

**3. Why is Square extending Rectangle a violation of LSP? How would you fix it?**

Rectangle's contract implies that `setWidth` and `setHeight` are independent operations. Square must keep width == height, so `setWidth` silently changes height (and vice versa). Any client code calling `setWidth(5); setHeight(10)` and expecting area == 50 will break. The fix is to avoid the inheritance relationship entirely -- make both implement a `Shape` interface with an `area()` method, or use immutable value objects.

**4. What is the difference between Dependency Inversion and Dependency Injection?**

Dependency Inversion is a **design principle** -- high-level modules should depend on abstractions, not concretions. Dependency Injection is a **technique** (constructor, setter, or interface injection) used to achieve that principle. DI is one way to implement DIP. IoC containers like Spring automate DI at scale.

**5. Give an example of Interface Segregation violation in a real codebase.**

A common violation is a `Repository` interface with `findAll()`, `findById()`, `save()`, `update()`, `delete()`, and `bulkImport()`. A read-only reporting service is forced to depend on mutating methods it never uses. Fix: split into `ReadableRepository` and `WritableRepository` (and optionally `BulkRepository`), so each client depends only on the slice it needs.

**6. Which SOLID principle is most closely related to testability, and why?**

The Dependency Inversion Principle. When a class depends on an abstract interface rather than a concrete implementation, you can inject test doubles (mocks, stubs, fakes) during unit testing. Without DIP, testing a `NotificationService` requires a real SMTP server or database connection. With DIP, you inject a `MockMessageSender` and test the logic in isolation.

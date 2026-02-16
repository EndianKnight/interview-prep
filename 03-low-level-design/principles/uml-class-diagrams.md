# UML & Class Diagrams

The universal visual language for communicating object-oriented designs in interviews and design documents.

---

## 1. Class Diagrams

A class diagram shows the structure of a system by depicting classes, their attributes, methods, and the relationships between them. Each class is drawn as a box divided into three compartments.

### The Class Box

```
┌─────────────────────────┐
│      <<stereotype>>     │  ← optional stereotype
│       ClassName         │  ← class name (bold, centered)
├─────────────────────────┤
│ - id: int               │  ← attributes (fields)
│ - name: String          │
│ # status: Status        │
├─────────────────────────┤
│ + getName(): String     │  ← methods (operations)
│ + setName(n: String)    │
│ - validate(): boolean   │
└─────────────────────────┘
```

### Visibility Modifiers

| Symbol | Visibility | Meaning |
|--------|-----------|---------|
| `+` | Public | Accessible from anywhere |
| `-` | Private | Accessible only within the class |
| `#` | Protected | Accessible within the class and subclasses |
| `~` | Package | Accessible within the same package (Java) |

### Special Notations

| Notation | UML Convention | Meaning |
|----------|---------------|---------|
| Underline | `+getInstance(): Singleton` | Static member (class-level, not instance-level) |
| *Italic* | `*draw(): void*` | Abstract method (no implementation) |
| *ClassName* | *Shape* | Abstract class (cannot be instantiated) |
| `<<interface>>` | `<<interface>> Comparable` | Interface (all methods abstract) |

### Mermaid: Basic Class

```mermaid
classDiagram
    class User {
        -int id
        -String name
        -String email
        +getId() int
        +getName() String
        +setName(name: String) void
        +toString() String
    }
```

### Mermaid: Abstract Class and Interface

```mermaid
classDiagram
    class Shape {
        <<abstract>>
        #double x
        #double y
        +getPosition() Point
        +area() double*
        +perimeter() double*
    }

    class Drawable {
        <<interface>>
        +draw(canvas: Canvas) void
        +resize(factor: double) void
    }

    class Circle {
        -double radius
        +area() double
        +perimeter() double
        +draw(canvas: Canvas) void
        +resize(factor: double) void
    }

    Shape <|-- Circle : extends
    Drawable <|.. Circle : implements
```

---

## 2. Relationships

Relationships are the backbone of class diagrams. Each type conveys a different strength of coupling between classes.

### Relationship Summary Table

| Relationship | Arrow | Strength | Meaning | Lifetime Coupling |
|-------------|-------|----------|---------|-------------------|
| Dependency | `..>` dashed arrow | Weakest | "uses temporarily" | None |
| Association | `-->` solid arrow | Weak | "uses" / "knows about" | Independent |
| Aggregation | `o--` hollow diamond | Medium | "has-a" (shared) | Parts outlive whole |
| Composition | `*--` filled diamond | Strong | "owns" (exclusive) | Parts die with whole |
| Inheritance | `<\|--` hollow triangle | Strongest | "is-a" | Permanent |
| Realization | `<\|..` dashed triangle | Strongest | "implements" | Permanent |

### Dependency ("depends on")

The weakest relationship. Class A uses class B temporarily, typically as a method parameter, local variable, or return type. A change in B may affect A.

```mermaid
classDiagram
    class OrderService {
        +processOrder(order: Order) void
    }
    class EmailService {
        +sendConfirmation(to: String) void
    }
    OrderService ..> EmailService : uses
```

**Example:** `OrderService` calls `EmailService.sendConfirmation()` but does not store a reference to it.

### Association ("uses" / "knows about")

A structural relationship where one class holds a reference to another. Both objects have independent lifecycles.

```mermaid
classDiagram
    class Student {
        -String name
        +enroll(course: Course) void
    }
    class Course {
        -String title
        +getStudents() List~Student~
    }
    Student "0..*" --> "0..*" Course : enrolls in
```

**Example:** A `Student` enrolls in many `Courses`, and a `Course` has many `Students`. Deleting a student does not delete the course.

### Aggregation ("has-a", shared ownership)

A special form of association. The whole contains parts, but the parts can exist independently. Represented by a hollow diamond on the "whole" side.

```mermaid
classDiagram
    class Department {
        -String name
        +addEmployee(e: Employee) void
    }
    class Employee {
        -String name
        -String role
    }
    Department o-- "1..*" Employee : has
```

**Example:** A `Department` has `Employees`, but if the department is dissolved, the employees still exist (they can be reassigned).

### Composition ("owns", exclusive ownership)

The strongest form of association. The whole exclusively owns its parts. When the whole is destroyed, all parts are destroyed too. Represented by a filled diamond.

```mermaid
classDiagram
    class House {
        -String address
        +getRooms() List~Room~
    }
    class Room {
        -double area
        -String type
    }
    House *-- "1..*" Room : contains
```

**Example:** A `House` contains `Rooms`. Destroying the house destroys its rooms -- a room cannot exist without its house.

### Inheritance ("is-a")

A generalization relationship. The subclass inherits all attributes and methods from the superclass and can add or override behavior.

```mermaid
classDiagram
    class Vehicle {
        <<abstract>>
        #String make
        #int year
        +start() void
        +stop() void
    }
    class Car {
        -int numDoors
        +start() void
    }
    class Truck {
        -double payloadCapacity
        +start() void
    }
    Vehicle <|-- Car
    Vehicle <|-- Truck
```

### Realization / Implementation ("implements")

A class provides a concrete implementation of an interface. Drawn with a dashed line and hollow triangle.

```mermaid
classDiagram
    class Comparable {
        <<interface>>
        +compareTo(other: Object) int
    }
    class Serializable {
        <<interface>>
        +serialize() byte[]
    }
    class Product {
        -String name
        -double price
        +compareTo(other: Object) int
        +serialize() byte[]
    }
    Comparable <|.. Product
    Serializable <|.. Product
```

### Multiplicity Notation

Multiplicity labels on relationship lines indicate how many instances participate.

| Notation | Meaning |
|----------|---------|
| `1` | Exactly one |
| `0..1` | Zero or one (optional) |
| `*` or `0..*` | Zero or more |
| `1..*` | One or more |
| `3..5` | Specific range |

```mermaid
classDiagram
    class Order {
        -int orderId
        -Date orderDate
    }
    class LineItem {
        -int quantity
        -double unitPrice
    }
    class Customer {
        -String name
    }
    Customer "1" --> "0..*" Order : places
    Order *-- "1..*" LineItem : contains
```

---

## 3. Sequence Diagrams

Sequence diagrams show how objects interact over time. They are essential for illustrating flows like "user places an order" or "borrower checks out a book."

### Core Elements

| Element | Description | Notation |
|---------|-------------|----------|
| **Actor** | External entity (user, system) | Stick figure or labeled box |
| **Lifeline** | Vertical dashed line extending from an object | Represents the object's existence over time |
| **Activation bar** | Thin rectangle on a lifeline | Period when the object is actively processing |
| **Synchronous message** | Solid arrow with filled head (`->>`) | Caller waits for response |
| **Asynchronous message** | Solid arrow with open head (`-)`) | Caller does not wait |
| **Return message** | Dashed arrow (`-->>`) | Response back to caller |
| **Self-message** | Arrow looping back to same lifeline | Object calls its own method |

### Basic Sequence Diagram

```mermaid
sequenceDiagram
    actor User
    participant UI as Web UI
    participant API as API Server
    participant DB as Database

    User->>UI: Click "Login"
    UI->>API: POST /login (credentials)
    API->>DB: SELECT user WHERE email = ?
    DB-->>API: User record
    API-->>API: Validate password hash
    API-->>UI: 200 OK + JWT token
    UI-->>User: Show dashboard
```

### Fragments: Loops, Conditionals, Optional

Fragments model control flow within sequence diagrams.

**alt / else** -- conditional branching (like if/else):

```mermaid
sequenceDiagram
    actor User
    participant Auth as AuthService
    participant DB as Database

    User->>Auth: login(email, password)
    Auth->>DB: findByEmail(email)
    DB-->>Auth: User record

    alt Valid credentials
        Auth-->>User: 200 OK + token
    else Invalid credentials
        Auth-->>User: 401 Unauthorized
    end
```

**opt** -- optional block (executes if condition is true):

```mermaid
sequenceDiagram
    participant OrderSvc as OrderService
    participant Inventory as InventoryService
    participant Notifier as NotificationService

    OrderSvc->>Inventory: reserveStock(itemId, qty)
    Inventory-->>OrderSvc: reserved

    opt Customer opted into SMS alerts
        OrderSvc->>Notifier: sendSMS(phone, message)
        Notifier-->>OrderSvc: sent
    end
```

**loop** -- repeated execution:

```mermaid
sequenceDiagram
    participant Client
    participant Server
    participant Cache

    Client->>Server: getItems(category)

    loop For each item in category
        Server->>Cache: lookup(itemId)
        alt Cache hit
            Cache-->>Server: cachedItem
        else Cache miss
            Server->>Server: fetchFromDB(itemId)
            Server->>Cache: store(itemId, item)
        end
    end

    Server-->>Client: List of items
```

---

## 4. Complete Example: Library Management System

Putting it all together with a realistic interview-style example.

### Class Diagram

```mermaid
classDiagram
    class Library {
        -String name
        -String address
        -List~Book~ catalog
        +searchByTitle(title: String) List~Book~
        +searchByAuthor(author: String) List~Book~
        +registerMember(member: Member) void
    }

    class Book {
        -String isbn
        -String title
        -String author
        -int totalCopies
        -int availableCopies
        +isAvailable() boolean
        +checkout() void
        +returnCopy() void
    }

    class Member {
        -String memberId
        -String name
        -String email
        -List~Loan~ activeLoans
        +borrowBook(book: Book) Loan
        +returnBook(loan: Loan) void
        +getActiveLoans() List~Loan~
    }

    class Loan {
        -String loanId
        -Date borrowDate
        -Date dueDate
        -Date returnDate
        -LoanStatus status
        +isOverdue() boolean
        +markReturned() void
    }

    class Librarian {
        -String employeeId
        +issueBook(member: Member, book: Book) Loan
        +acceptReturn(loan: Loan) void
        +collectFine(member: Member, amount: double) void
    }

    class LoanStatus {
        <<enumeration>>
        ACTIVE
        RETURNED
        OVERDUE
    }

    Library *-- "0..*" Book : contains
    Library o-- "0..*" Member : has members
    Member "1" --> "0..*" Loan : has
    Loan "1" --> "1" Book : for
    Librarian --> Library : manages
    Loan --> LoanStatus : has status
```

### Sequence Diagram: Borrow Book Flow

```mermaid
sequenceDiagram
    actor Borrower as Member
    participant Lib as Librarian
    participant Sys as LibrarySystem
    participant BookObj as Book
    participant LoanObj as Loan

    Borrower->>Lib: Request to borrow "Clean Code"
    Lib->>Sys: issueBook(member, bookTitle)
    Sys->>Sys: validateMembership(member)

    alt Member not valid
        Sys-->>Lib: Error: Invalid membership
        Lib-->>Borrower: Cannot issue book
    else Member valid
        Sys->>BookObj: isAvailable()
        BookObj-->>Sys: availability status

        alt Book not available
            Sys-->>Lib: Error: No copies available
            Lib-->>Borrower: Book unavailable, added to waitlist
        else Book available
            Sys->>BookObj: checkout()
            BookObj->>BookObj: availableCopies--
            Sys->>LoanObj: create(member, book, today, dueDate)
            LoanObj-->>Sys: new Loan record
            Sys-->>Lib: Loan confirmation
            Lib-->>Borrower: Book issued, due on [date]
        end
    end
```

---

## 5. Interview Tips

### Drawing on a Whiteboard

1. **Start with nouns** -- identify the key entities from the problem statement. Each noun is a potential class (e.g., "Library", "Book", "Member", "Loan").
2. **Draw relationships first** -- sketch boxes with just class names and connect them with lines. This shows you understand the domain before diving into details.
3. **Add attributes second** -- write the most important 3-5 fields per class. Skip obvious getters/setters.
4. **Add methods as you discuss use cases** -- when the interviewer asks "how does a member borrow a book?", add `borrowBook()` to the `Member` class and walk through the sequence.
5. **Label multiplicity** -- always annotate `1`, `*`, `0..1` on relationship lines. This shows you think about edge cases.

### Common Mistakes to Avoid

| Mistake | Why It Matters |
|---------|---------------|
| Using inheritance when composition fits better | Leads to rigid hierarchies; interviewers look for this |
| Missing multiplicity labels | Shows incomplete thinking about the domain |
| Putting every method in one "God class" | Violates Single Responsibility; spread behavior across classes |
| Confusing aggregation and composition | Know the lifecycle difference -- interviewers will probe this |
| Drawing sequence diagrams without return arrows | Incomplete communication flow; always show responses |

### Aggregation vs. Composition Decision Guide

Ask yourself: **"If I delete the parent, should the children be deleted too?"**

- **Yes** --> Composition (filled diamond). Example: `Order *-- LineItem`
- **No** --> Aggregation (hollow diamond). Example: `Team o-- Player`

---

## Common Interview Questions

**Q1: What is the difference between aggregation and composition?**
Both represent "has-a" relationships. **Aggregation** (hollow diamond) is shared ownership -- the child can exist independently of the parent (e.g., a `Department` has `Employees`, but employees survive if the department is dissolved). **Composition** (filled diamond) is exclusive ownership -- the child's lifecycle is bound to the parent (e.g., a `House` has `Rooms`, and rooms are destroyed when the house is demolished).

**Q2: When would you use an association instead of a dependency?**
Use **association** when one class holds a persistent reference to another (a field/member variable). Use **dependency** when the usage is transient -- the other class appears only as a method parameter, local variable, or return type. Association implies a structural relationship; dependency implies only a temporary usage.

**Q3: How do you decide what goes into a class diagram during an interview?**
Start with the core entities (nouns from the problem). Draw 4-6 key classes with their relationships first. Add 3-5 essential attributes per class and the methods that directly support the use cases the interviewer asks about. Avoid modeling utility classes, exceptions, or framework details unless specifically asked.

**Q4: How do you represent an interface vs. an abstract class in UML?**
An **interface** is labeled with the `<<interface>>` stereotype and has no concrete method implementations. Classes connect to it with a dashed line and hollow triangle (realization). An **abstract class** has its name in *italics* (or labeled `<<abstract>>`), can contain both abstract and concrete methods, and subclasses connect with a solid line and hollow triangle (inheritance).

**Q5: What does multiplicity "0..*" vs "1..*" convey in a design?**
`0..*` means the relationship is optional -- there can be zero or more associated objects (e.g., a new `Customer` may have zero `Orders`). `1..*` means at least one is required (e.g., an `Order` must have at least one `LineItem`). This distinction communicates validation constraints and affects your code's null-checking and error-handling logic.

**Q6: When should you draw a sequence diagram vs. a class diagram in an interview?**
Draw a **class diagram** when the question is about system structure -- "Design a parking lot" or "Model an e-commerce system." Draw a **sequence diagram** when the question is about behavior or flow -- "Walk me through what happens when a user places an order." In practice, start with the class diagram to establish entities, then use sequence diagrams to illustrate specific use cases the interviewer wants to explore.

# Low-Level Design (LLD)

A guide to object-oriented design and low-level system design interviews for senior engineers.

## How to Approach LLD Interviews

```
1. Clarify Requirements (3-5 min)
   ├── Identify actors / users
   ├── Core use cases
   └── Constraints & assumptions

2. Class Design (10-15 min)
   ├── Identify key entities / classes
   ├── Define relationships (is-a, has-a)
   ├── Apply SOLID principles
   └── Draw class diagrams

3. API / Interface Design (5 min)
   ├── Public methods & contracts
   └── Error handling strategy

4. Walkthrough (5 min)
   ├── Trace through a use case
   ├── Identify extensibility points
   └── Discuss design patterns used
```

## Principles

| Topic | Guide |
|-------|-------|
| OOP Fundamentals | [oop-fundamentals.md](principles/oop-fundamentals.md) |
| SOLID Principles | [solid.md](principles/solid.md) |
| DRY, KISS, YAGNI | [dry-kiss-yagni.md](principles/dry-kiss-yagni.md) |
| Coupling & Cohesion | [coupling-cohesion.md](principles/coupling-cohesion.md) |
| Composition vs Inheritance | [composition-vs-inheritance.md](principles/composition-vs-inheritance.md) |
| UML & Class Diagrams | [uml-class-diagrams.md](principles/uml-class-diagrams.md) |
| Concurrency Basics | [concurrency-basics.md](principles/concurrency-basics.md) |

## Case Studies

Ordered by interview frequency (🔴 = must-know, 🟡 = high priority, 🟢 = nice to have):

| Priority | Problem | Key Patterns | Guide |
|----------|---------|-------------|-------|
| 🔴 | Parking Lot | Strategy, Factory | [parking-lot.md](case-studies/parking-lot.md) |
| 🔴 | Elevator System | State, Strategy, Observer | [elevator-system.md](case-studies/elevator-system.md) |
| 🔴 | Vending Machine | State, Strategy | [vending-machine.md](case-studies/vending-machine.md) |
| 🔴 | ATM System | State, Chain of Responsibility | [atm-system.md](case-studies/atm-system.md) |
| 🔴 | Tic-Tac-Toe | Clean OOP, Strategy | [tic-tac-toe.md](case-studies/tic-tac-toe.md) |
| 🔴 | Movie Ticket Booking | Observer, Concurrency | [movie-ticket-booking.md](case-studies/movie-ticket-booking.md) |
| 🟡 | Library Management | Factory, Observer | [library-management.md](case-studies/library-management.md) |
| 🟡 | Snake and Ladder | Strategy, Factory | [snake-and-ladder.md](case-studies/snake-and-ladder.md) |
| 🟡 | Logger System | Singleton, Chain of Resp. | [logger-system.md](case-studies/logger-system.md) |
| 🟡 | Chess Game | Inheritance, Strategy | [chess-game.md](case-studies/chess-game.md) |
| 🟡 | Hotel Booking System | State, Observer | [hotel-booking.md](case-studies/hotel-booking.md) |
| 🟡 | Food Delivery App | Strategy, Observer | [food-delivery-app.md](case-studies/food-delivery-app.md) |
| 🟢 | Shopping Cart (Amazon) | Strategy, Decorator | [shopping-cart.md](case-studies/shopping-cart.md) |
| 🟢 | Splitwise (Expense Sharing) | Observer, Graph | [splitwise.md](case-studies/splitwise.md) |
| 🟢 | Car Rental System | State, Strategy | [car-rental.md](case-studies/car-rental.md) |
| 🟢 | Online Auction System | Observer, Strategy, State | [online-auction.md](case-studies/online-auction.md) |

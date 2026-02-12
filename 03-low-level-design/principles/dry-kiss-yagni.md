# DRY, KISS, YAGNI

> TODO: Detailed guide with examples in Java/C++/Python

## Topics to Cover

### DRY — Don't Repeat Yourself
- Every piece of knowledge should have a single representation
- Code duplication vs knowledge duplication — they're different
- Rule of Three — duplicate twice before abstracting
- Techniques: extract method, extract class, template method pattern

### KISS — Keep It Simple, Stupid
- The simplest solution that works is usually the best
- Signs of over-engineering: premature abstraction, speculative generality
- Simplicity in API design — fewer parameters, clear naming

### YAGNI — You Aren't Gonna Need It
- Don't build features until they're actually needed
- Cost of unnecessary code: maintenance burden, cognitive load, bugs
- YAGNI vs planning ahead — where to draw the line

### How They Work Together
- DRY eliminates redundancy, KISS keeps solutions simple, YAGNI prevents over-building
- Tension: DRY can make code more complex (violating KISS)

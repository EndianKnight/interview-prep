# DSA in Java

Java implementations, Collections framework tips, and language-specific patterns for coding interviews.

## Java Interview Essentials

### Collections Cheat Sheet

| Collection | Use Case | Key Operations |
|-----------|----------|---------------|
| `ArrayList` | Dynamic array | `add`, `get`, `set`, `remove` |
| `LinkedList` | Doubly-linked list / deque | `addFirst/Last`, `pollFirst/Last` |
| `HashMap` | Hash map | `put`, `get`, `containsKey` |
| `TreeMap` | Ordered map (Red-Black) | `floorKey`, `ceilingKey` |
| `HashSet` | Hash set | `add`, `contains`, `remove` |
| `TreeSet` | Ordered set | `floor`, `ceiling`, `first`, `last` |
| `PriorityQueue` | Min-heap | `offer`, `poll`, `peek` |
| `ArrayDeque` | Stack / Queue | `push`, `pop`, `offer`, `poll` |

### Tips

- `PriorityQueue` is a **min-heap** by default; use `Collections.reverseOrder()` for max-heap
- Prefer `ArrayDeque` over `Stack` for stack operations
- Use `Map.getOrDefault()` and `Map.merge()` for cleaner frequency counting
- `StringBuilder` for string concatenation in loops (not `+`)
- Use `Arrays.sort()` with custom comparators via lambdas

## Solutions

> Solutions organized by topic will be added here.

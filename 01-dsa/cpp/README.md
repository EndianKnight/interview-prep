# DSA in C++

C++ implementations, STL tips, and language-specific patterns for coding interviews.

## C++ Interview Essentials

### STL Containers Cheat Sheet

| Container | Use Case | Key Operations |
|-----------|----------|---------------|
| `vector` | Dynamic array | `push_back`, `pop_back`, `[]` |
| `deque` | Double-ended queue | `push_front/back`, `pop_front/back` |
| `unordered_map` | Hash map | `[]`, `find`, `count` |
| `map` | Ordered map (BST) | `lower_bound`, `upper_bound` |
| `unordered_set` | Hash set | `insert`, `find`, `count` |
| `set` | Ordered set | `lower_bound`, `upper_bound` |
| `priority_queue` | Max-heap | `push`, `pop`, `top` |
| `stack` | LIFO | `push`, `pop`, `top` |
| `queue` | FIFO | `push`, `pop`, `front` |

### Tips

- Use `auto` generously to keep code concise
- Prefer `unordered_map` over `map` for O(1) average lookups
- Use `reserve()` on vectors when size is known upfront
- Remember: `priority_queue` is a **max-heap** by default; use `greater<int>` for min-heap
- Use structured bindings (`auto [key, val] : map`) for cleaner iteration (C++17)

## Solutions

> Solutions organized by topic will be added here.

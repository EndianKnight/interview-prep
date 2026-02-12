# DSA in Python

Python implementations, standard library tips, and Pythonic patterns for coding interviews.

## Python Interview Essentials

### Standard Library Cheat Sheet

| Module / Type | Use Case | Key Operations |
|--------------|----------|---------------|
| `list` | Dynamic array | `append`, `pop`, slicing |
| `collections.deque` | Double-ended queue | `append`, `appendleft`, `popleft` |
| `dict` | Hash map | `[]`, `get`, `setdefault` |
| `collections.defaultdict` | Hash map with defaults | Auto-initializing keys |
| `collections.Counter` | Frequency counting | `most_common`, arithmetic ops |
| `set` | Hash set | `add`, `discard`, set ops |
| `heapq` | Min-heap (on lists) | `heappush`, `heappop`, `nlargest` |
| `sortedcontainers.SortedList` | Ordered list | `add`, `bisect_left`, `irange` |
| `bisect` | Binary search on sorted lists | `bisect_left`, `insort` |

### Tips

- `heapq` is a **min-heap**; negate values for max-heap behavior
- Use `collections.Counter` for frequency problems — supports subtraction & intersection
- Slicing creates a copy — O(k) not O(1)
- Tuple packing/unpacking for clean swap: `a, b = b, a`
- `@lru_cache` (or `@cache` in 3.9+) for memoization
- `math.inf` and `-math.inf` for sentinel values
- Use `enumerate()` instead of manual index tracking

## Solutions

> Solutions organized by topic will be added here.

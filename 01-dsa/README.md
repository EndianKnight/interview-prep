# Data Structures & Algorithms

A pattern-based DSA refresher for senior engineers. Each topic guide covers the core concept, Big-O complexities, common patterns, pitfalls, and links to practice problems.

## How to Use

1. **Skim the topic guides** in [`topics/`](topics/) to refresh concepts
2. **Practice in your interview language** using the solutions in [`cpp/`](cpp/), [`java/`](java/), or [`python/`](python/)
3. **Focus on weak areas** — don't grind everything linearly

## Language Refreshers

Quick-reference guides for your interview language, covering **every data structure**, standard library algorithms, lambda functions, custom comparators, and class/object usage:

| Language | Guide | Highlights |
|----------|-------|------------|
| C++ | [cpp/README.md](cpp/README.md) | STL containers (`vector`, `map`, `set`, `priority_queue`, `bitset`, `multiset`), `<algorithm>` & `<numeric>` (40+ functions), lambda captures, custom comparators for sort/PQ/set, structs with operator overloading & custom hash |
| Java | [java/README.md](java/README.md) | Collections (`ArrayList`, `HashMap`, `TreeMap`, `PriorityQueue`, `ArrayDeque`, `LinkedHashMap`, `BitSet`), functional interfaces & method references, `Comparator.comparing` chains, `record` types, `equals`/`hashCode` contract, Streams & Collectors |
| Python | [python/README.md](python/README.md) | Built-ins (`list`, `dict`, `set`, `tuple`, `frozenset`), `collections` (`Counter`, `defaultdict`, `deque`, `OrderedDict`), `heapq`, `bisect`, `itertools`, `functools` (`lru_cache`, `cmp_to_key`), `@dataclass`, `@total_ordering`, `sortedcontainers` |

## Topics Index

| # | Topic | Key Patterns |
|---|-------|-------------|
| 1 | [Arrays & Strings](topics/arrays-and-strings.md) | Two pointers, prefix sums, kadane's |
| 2 | [Linked Lists](topics/linked-lists.md) | Fast/slow pointers, reversal |
| 3 | [Stacks](topics/stacks.md) | Monotonic stack, matching, expression eval, min/max |
| 4 | [Queues & Deques](topics/queues.md) | BFS, monotonic deque, sliding window max |
| 5 | [Trees & Graphs](topics/trees-and-graphs.md) | DFS, BFS, topological sort |
| 6 | [Heaps & Priority Queues](topics/heaps-and-priority-queues.md) | Top-K, merge K sorted |
| 7 | [Hash Maps & Sets](topics/hash-maps-and-sets.md) | Frequency count, grouping |
| 8 | [Sorting & Searching](topics/sorting-and-searching.md) | Binary search variations |
| 9 | [Recursion & Backtracking](topics/recursion-and-backtracking.md) | Permutations, subsets, N-queens |
| 10 | [Dynamic Programming](topics/dynamic-programming.md) | Memoization, tabulation, state machines |
| 11 | [Greedy Algorithms](topics/greedy-algorithms.md) | Interval scheduling, Huffman |
| 12 | [Sliding Window & Two Pointers](topics/sliding-window-two-pointers.md) | Fixed/variable window |
| 13 | [Bit Manipulation](topics/bit-manipulation.md) | XOR tricks, bit masks |
| 14 | [Intervals](topics/intervals.md) | Merge, insert, overlap |
| 15 | [Tries & Advanced](topics/tries-and-advanced.md) | Prefix trees, segment trees |

## Complexity Cheat Sheet

| Structure | Access | Search | Insert | Delete |
|-----------|--------|--------|--------|--------|
| Array | O(1) | O(n) | O(n) | O(n) |
| Linked List | O(n) | O(n) | O(1) | O(1) |
| Hash Table | — | O(1)* | O(1)* | O(1)* |
| BST | O(log n)* | O(log n)* | O(log n)* | O(log n)* |
| Heap | — | O(n) | O(log n) | O(log n) |

\* amortized / average case

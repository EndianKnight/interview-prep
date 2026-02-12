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
| 1 | [Arrays & Strings](topics/01-arrays-and-strings.md) | Two pointers, prefix sums, kadane's |
| 2 | [Linked Lists](topics/02-linked-lists.md) | Fast/slow pointers, reversal |
| 3 | [Stacks](topics/03-stacks.md) | Monotonic stack, matching, expression eval, min/max |
| 4 | [Queues & Deques](topics/04-queues.md) | BFS, monotonic deque, sliding window max |
| 5 | [Trees](topics/05-trees.md) | DFS traversals, BFS level-order, BST, LCA, construction |
| 6 | [Graphs](topics/06-graphs.md) | DFS/BFS, topological sort, Union-Find, Dijkstra, MST, SCC |
| 7 | [Heaps & Priority Queues](topics/07-heaps-and-priority-queues.md) | Top-K, merge K sorted |
| 8 | [Hash Maps & Sets](topics/08-hash-maps-and-sets.md) | Frequency count, grouping |
| 9 | [Sorting & Searching](topics/09-sorting-and-searching.md) | Binary search variations |
| 10 | [Recursion & Backtracking](topics/10-recursion-and-backtracking.md) | Permutations, subsets, N-queens |
| 11 | [Dynamic Programming](topics/11-dynamic-programming.md) | Memoization, tabulation, state machines |
| 12 | [Greedy Algorithms](topics/12-greedy-algorithms.md) | Interval scheduling, Huffman |
| 13 | [Sliding Window & Two Pointers](topics/13-sliding-window-two-pointers.md) | Fixed/variable window |
| 14 | [Bit Manipulation](topics/14-bit-manipulation.md) | XOR tricks, bit masks |
| 15 | [Intervals](topics/15-intervals.md) | Merge, insert, overlap |
| 16 | [Tries & Advanced](topics/16-tries-and-advanced.md) | Prefix trees, segment trees |
| 17 | [Math & Number Theory](topics/17-math-and-number-theory.md) | Primes, sieve, GCD, modular arithmetic, combinatorics |
| 18 | [Matrix Patterns](topics/18-matrix-patterns.md) | Spiral, rotate, search sorted 2D, grid DP |
| 19 | [Design Data Structures](topics/19-design-data-structures.md) | LRU/LFU Cache, RandomizedSet, Min Stack, iterators |

## Complexity Cheat Sheet

| Structure | Access | Search | Insert | Delete |
|-----------|--------|--------|--------|--------|
| Array | O(1) | O(n) | O(n) | O(n) |
| Linked List | O(n) | O(n) | O(1) | O(1) |
| Hash Table | — | O(1)* | O(1)* | O(1)* |
| BST | O(log n)* | O(log n)* | O(log n)* | O(log n)* |
| Heap | — | O(n) | O(log n) | O(log n) |

\* amortized / average case

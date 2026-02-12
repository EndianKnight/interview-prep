# Data Structures & Algorithms

A pattern-based DSA refresher for senior engineers. Each topic guide covers the core concept, Big-O complexities, common patterns, pitfalls, and links to practice problems.

## How to Use

1. **Skim the topic guides** in [`topics/`](topics/) to refresh concepts
2. **Practice in your interview language** using the solutions in [`cpp/`](cpp/), [`java/`](java/), or [`python/`](python/)
3. **Focus on weak areas** — don't grind everything linearly

## Topics Index

| # | Topic | Key Patterns |
|---|-------|-------------|
| 1 | [Arrays & Strings](topics/arrays-and-strings.md) | Two pointers, prefix sums, kadane's |
| 2 | [Linked Lists](topics/linked-lists.md) | Fast/slow pointers, reversal |
| 3 | [Stacks & Queues](topics/stacks-and-queues.md) | Monotonic stack, BFS |
| 4 | [Trees & Graphs](topics/trees-and-graphs.md) | DFS, BFS, topological sort |
| 5 | [Heaps & Priority Queues](topics/heaps-and-priority-queues.md) | Top-K, merge K sorted |
| 6 | [Hash Maps & Sets](topics/hash-maps-and-sets.md) | Frequency count, grouping |
| 7 | [Sorting & Searching](topics/sorting-and-searching.md) | Binary search variations |
| 8 | [Recursion & Backtracking](topics/recursion-and-backtracking.md) | Permutations, subsets, N-queens |
| 9 | [Dynamic Programming](topics/dynamic-programming.md) | Memoization, tabulation, state machines |
| 10 | [Greedy Algorithms](topics/greedy-algorithms.md) | Interval scheduling, Huffman |
| 11 | [Sliding Window & Two Pointers](topics/sliding-window-two-pointers.md) | Fixed/variable window |
| 12 | [Bit Manipulation](topics/bit-manipulation.md) | XOR tricks, bit masks |
| 13 | [Intervals](topics/intervals.md) | Merge, insert, overlap |
| 14 | [Tries & Advanced](topics/tries-and-advanced.md) | Prefix trees, segment trees |

## Complexity Cheat Sheet

| Structure | Access | Search | Insert | Delete |
|-----------|--------|--------|--------|--------|
| Array | O(1) | O(n) | O(n) | O(n) |
| Linked List | O(n) | O(n) | O(1) | O(1) |
| Hash Table | — | O(1)* | O(1)* | O(1)* |
| BST | O(log n)* | O(log n)* | O(log n)* | O(log n)* |
| Heap | — | O(n) | O(log n) | O(log n) |

\* amortized / average case

# DSA in Python — Complete Interview Refresher

A comprehensive Python refresher for coding interviews, covering built-in data structures, standard library, idioms, and performance tips.

---

## Table of Contents
- [Data Types & Built-ins](#data-types--built-ins)
- [Data Structures Deep Dive](#data-structures-deep-dive)
- [String Operations](#string-operations)
- [Standard Library Essentials](#standard-library-essentials)
- [Common Interview Idioms](#common-interview-idioms)
- [Python-Specific Gotchas](#python-specific-gotchas)

---

## Data Types & Built-ins

```python
# Integers — arbitrary precision (no overflow!)
x = 2 ** 100        # works fine
int("42")            # str → int
str(42)              # int → str

# Floats
float('inf'), float('-inf')    # infinity sentinels
math.inf, -math.inf            # alternative

# Booleans
True, False
bool(0) == False; bool(1) == True; bool("") == False; bool([]) == False

# None
x = None
if x is None: ...    # always use 'is', not '=='
```

---

## Data Structures Deep Dive

### `list` — Dynamic Array

```python
lst = []                      # empty
lst = [0] * n                 # n zeros
lst = [i for i in range(n)]   # list comprehension
grid = [[0] * cols for _ in range(rows)]  # 2D grid

# Operations
lst.append(x)          # O(1) amortized
lst.pop()              # O(1) — remove last
lst.pop(i)             # O(n) — remove at index
lst.insert(i, x)       # O(n) — insert at index
lst[i]                 # O(1) — access
lst[-1]                # last element
lst[-2]                # second to last
len(lst)               # O(1)

# Slicing — creates a COPY, O(k) where k = slice length
lst[start:end]         # [start, end)
lst[::2]               # every 2nd element
lst[::-1]              # reversed copy

# Sorting
lst.sort()                        # in-place, stable (Timsort)
lst.sort(key=lambda x: x[1])     # custom key
lst.sort(key=lambda x: (-x[0], x[1]))  # multi-criteria
sorted(lst)                       # returns new sorted list
sorted(lst, reverse=True)         # descending

# Other
lst.index(x)           # O(n) — first occurrence
x in lst               # O(n) — membership check
lst.count(x)           # O(n)
lst.reverse()          # in-place reverse
lst.extend(other)      # append all elements from other
lst + other            # concatenate (creates new list)
```

### `dict` — Hash Map

```python
d = {}
d = {'a': 1, 'b': 2}
d = dict(zip(keys, values))

# Operations
d[key] = val                  # insert/update
d[key]                        # access — KeyError if missing!
d.get(key, default)           # safe access — returns default if missing
key in d                      # O(1) avg — check existence
del d[key]                    # remove
d.pop(key, default)           # remove and return
len(d)                        # count of entries

# Iteration (insertion-ordered since Python 3.7)
for key in d:                 # iterate keys
for key, val in d.items():    # iterate key-value pairs
for val in d.values():        # iterate values

# Comprehension
d = {k: v for k, v in pairs}
d = {x: x**2 for x in range(10)}

# Merge (Python 3.9+)
merged = d1 | d2
```

### `set` — Hash Set

```python
s = set()
s = {1, 2, 3}
s = set(lst)                   # deduplicate a list

# Operations
s.add(x)              # O(1) avg
s.discard(x)          # O(1) avg — no error if missing
s.remove(x)           # O(1) avg — KeyError if missing
x in s                # O(1) avg
len(s)

# Set operations
s1 & s2               # intersection
s1 | s2               # union
s1 - s2               # difference
s1 ^ s2               # symmetric difference
s1.issubset(s2)       # s1 ⊆ s2
s1.issuperset(s2)     # s1 ⊇ s2
```

### `tuple` — Immutable Sequence

```python
t = (1, 2, 3)
t = 1, 2, 3           # parentheses optional
a, b, c = t           # unpacking
t[0]                   # access
len(t)

# Tuples are hashable → can be dict keys or set elements
seen = set()
seen.add((row, col))

# Comparison is lexicographic
(1, 2) < (1, 3)       # True
(1, 2) < (2, 0)       # True
```

---

## String Operations

```python
s = "hello"
len(s)
s[i]                   # O(1) access
s[1:4]                 # "ell" — slicing
s[::-1]                # reversed string
s + " world"           # concatenation — O(n+m), creates new string!

# Methods
s.find("ll")           # 2 (index), -1 if not found
s.index("ll")          # 2 (index), ValueError if not found
s.count("l")           # 2
s.startswith("he")     # True
s.endswith("lo")       # True
s.replace("l", "r")    # "herro" — returns new string
s.split()              # ["hello"] — split by whitespace
s.split(",")           # split by delimiter
",".join(["a","b","c"])# "a,b,c"
s.strip()              # remove leading/trailing whitespace
s.lower(); s.upper()
s.isdigit(); s.isalpha(); s.isalnum()

# Character operations
ord('a')   # 97
chr(97)    # 'a'
ord(c) - ord('a')  # 0-25 index

# Building strings efficiently
parts = []
for ...:
    parts.append(chunk)
result = "".join(parts)  # O(n) total — MUCH better than += in loop

# f-strings
f"value is {x}, computed: {x*2}"
```

---

## Standard Library Essentials

### `collections`

```python
from collections import defaultdict, Counter, deque, OrderedDict

# defaultdict — auto-initializing dict
graph = defaultdict(list)
graph[node].append(neighbor)  # no KeyError
freq = defaultdict(int)
freq[x] += 1

# Counter — frequency counting
counts = Counter("abracadabra")   # Counter({'a': 5, 'b': 2, ...})
counts.most_common(3)             # top 3
counts[x]                         # frequency of x (0 if missing)
counts.update(more_items)         # add counts
counts1 + counts2                 # combine
counts1 - counts2                 # subtract (keeps positive)
counts1 & counts2                 # intersection (min)

# deque — doubly-ended queue (O(1) both ends)
dq = deque()
dq.append(x)          # right
dq.appendleft(x)      # left
dq.pop()               # right
dq.popleft()           # left — THIS is why you use deque, not list
dq[0]; dq[-1]          # peek
deque(iterable, maxlen=k)  # fixed-size window
```

### `heapq` — Min-Heap

```python
import heapq

heap = []
heapq.heappush(heap, x)     # O(log n)
heapq.heappop(heap)          # O(log n) — returns smallest
heap[0]                      # O(1) — peek smallest
heapq.heapify(lst)           # O(n) — convert list to heap in-place
heapq.nlargest(k, lst)       # top-k largest
heapq.nsmallest(k, lst)      # top-k smallest

# Max-heap trick: negate values
heapq.heappush(heap, -x)
max_val = -heapq.heappop(heap)

# With tuples (compared lexicographically — handy for priority)
heapq.heappush(heap, (priority, data))

# Merge k sorted iterables
merged = list(heapq.merge(sorted1, sorted2, sorted3))
```

### `bisect` — Binary Search

```python
from bisect import bisect_left, bisect_right, insort

# bisect_left: first position where x could be inserted (leftmost)
i = bisect_left(sorted_arr, x)   # first >= x

# bisect_right: last position where x could be inserted (rightmost)
j = bisect_right(sorted_arr, x)  # first > x

# insort: insert and maintain sorted order
insort(sorted_arr, x)            # O(n) due to shift

# Count occurrences in sorted array
count = bisect_right(arr, x) - bisect_left(arr, x)

# Check if x exists in sorted array
def binary_search(arr, x):
    i = bisect_left(arr, x)
    return i < len(arr) and arr[i] == x
```

### `itertools`

```python
from itertools import permutations, combinations, product, accumulate, chain

list(permutations([1,2,3]))         # all orderings
list(permutations([1,2,3], 2))      # length-2 orderings
list(combinations([1,2,3], 2))       # choose 2
list(product([0,1], repeat=3))       # cartesian product: 000,001,...,111
list(accumulate([1,2,3,4]))          # prefix sums: [1,3,6,10]
list(chain([1,2], [3,4]))            # flatten: [1,2,3,4]
```

### `functools`

```python
from functools import lru_cache, cache, reduce

# Memoization for DP
@lru_cache(maxsize=None)   # or @cache (Python 3.9+)
def dp(i, j):
    ...

# Reduce
from operator import xor
reduce(xor, nums)  # XOR all elements
```

### `math`

```python
import math

math.inf, -math.inf
math.ceil(x)           # ceiling
math.floor(x)          # floor
math.sqrt(x)
math.gcd(a, b)         # GCD
math.lcm(a, b)         # LCM (Python 3.9+)
math.log(x), math.log2(x), math.log10(x)
math.isqrt(n)          # integer square root (Python 3.8+)
math.comb(n, k)        # n choose k (Python 3.8+)
math.factorial(n)
```

---

## Common Interview Idioms

### Enumerate and Zip

```python
for i, val in enumerate(lst):       # index + value
for i, val in enumerate(lst, 1):    # start from 1

for a, b in zip(lst1, lst2):        # parallel iteration
for a, b, c in zip(l1, l2, l3):     # three lists
```

### List Comprehensions

```python
squares = [x**2 for x in range(10)]
evens = [x for x in lst if x % 2 == 0]
flat = [x for row in grid for x in row]   # flatten 2D
```

### Dictionary Patterns

```python
# Frequency counter (manual)
freq = {}
for x in nums:
    freq[x] = freq.get(x, 0) + 1

# Invert a dict
inv = {v: k for k, v in d.items()}

# Group by
from collections import defaultdict
groups = defaultdict(list)
for item in items:
    groups[key_func(item)].append(item)
```

### Swap

```python
a, b = b, a           # no temp variable needed
```

### Infinity

```python
float('inf')           # positive infinity
float('-inf')          # negative infinity
math.inf               # alternative
```

### Min/Max with Default

```python
min(lst, default=float('inf'))
max(lst, default=float('-inf'))
```

---

## Python-Specific Gotchas

| Gotcha | Solution |
|--------|----------|
| `list.pop(0)` is O(n) | Use `collections.deque.popleft()` for O(1) |
| Mutable default arguments | `def f(lst=None): lst = lst or []` |
| `[[0]*n]*m` shares rows | Use `[[0]*n for _ in range(m)]` |
| `is` vs `==` | `is` checks identity; `==` checks equality. Use `is` only for `None` |
| Integer division `//` floors | `-7 // 2 == -4` (floors toward -∞). Use `int(-7/2)` for truncation toward 0 |
| Slicing creates copies | `lst[:]` is O(n), not O(1) |
| `set` and `dict` keys must be hashable | Lists are unhashable — convert to `tuple` |
| Recursion limit (default 1000) | `sys.setrecursionlimit(10000)` or use iterative approach |
| `lru_cache` requires hashable args | Can't cache functions with list arguments — use tuples |
| Global variables in nested functions | Use `nonlocal` keyword to modify enclosing scope variable |
| `sort()` is in-place (returns None) | `sorted()` returns a new list |

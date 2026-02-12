# DSA in Python — Complete Interview Refresher

A comprehensive Python refresher for coding interviews, covering built-in data structures, standard library, lambdas, comparators, classes, and performance tips.

---

## Table of Contents
- [Data Types & Built-ins](#data-types--built-ins)
- [Data Structures Deep Dive](#data-structures-deep-dive)
- [String Operations](#string-operations)
- [Lambda Functions](#lambda-functions)
- [Custom Comparators & Sorting](#custom-comparators--sorting)
- [Classes & Objects](#classes--objects)
- [Standard Library Essentials](#standard-library-essentials)
- [Built-in Functions](#built-in-functions)
- [Common Interview Idioms](#common-interview-idioms)
- [Python-Specific Gotchas](#python-specific-gotchas)

---

## Data Types & Built-ins

```python
# Integers — arbitrary precision (no overflow!)
x = 2 ** 100        # works fine
int("42")            # str → int
str(42)              # int → str
bin(10)              # '0b1010'
oct(10)              # '0o12'
hex(255)             # '0xff'
int('1010', 2)       # binary str → int: 10
int('ff', 16)        # hex str → int: 255

# Floats
float('inf'), float('-inf')    # infinity sentinels
float('nan')                   # not a number
import math
math.inf, -math.inf            # alternative infinity
math.isnan(x); math.isinf(x)  # check special values

# Booleans
True, False
bool(0) == False; bool(1) == True; bool("") == False; bool([]) == False

# None
x = None
if x is None: ...    # always use 'is', not '=='

# Type checking
isinstance(x, int)           # True if x is int
isinstance(x, (int, float))  # True if int or float
type(x) == int               # exact type check (less preferred)
```

---

## Data Structures Deep Dive

### `list` — Dynamic Array

```python
lst = []                      # empty
lst = [0] * n                 # n zeros
lst = [i for i in range(n)]   # list comprehension
grid = [[0] * cols for _ in range(rows)]  # 2D grid (correct way)

# Operations
lst.append(x)          # O(1) amortized
lst.pop()              # O(1) — remove last
lst.pop(i)             # O(n) — remove at index, returns removed
lst.insert(i, x)       # O(n) — insert at index
lst[i]                 # O(1) — access
lst[-1]                # last element
lst[-2]                # second to last
len(lst)               # O(1)
lst.extend(other)      # O(k) — append all elements from other
lst += other           # same as extend

# Slicing — creates a COPY, O(k) where k = slice length
lst[start:end]         # [start, end)
lst[start:]            # from start to end
lst[:end]              # from beginning to end
lst[::2]               # every 2nd element
lst[::-1]              # reversed copy
lst[i:j] = [new_vals]  # replace slice (can change size!)

# Sorting
lst.sort()                        # in-place, stable (Timsort), returns None!
lst.sort(key=lambda x: x[1])     # custom key
lst.sort(key=lambda x: (-x[0], x[1]))  # multi-criteria
sorted(lst)                       # returns new sorted list
sorted(lst, reverse=True)         # descending

# Searching
lst.index(x)           # O(n) — first occurrence, ValueError if absent
lst.index(x, start)    # search from start position
x in lst               # O(n) — membership check
lst.count(x)           # O(n) — count occurrences

# Other
lst.reverse()          # in-place reverse, returns None
lst.copy()             # shallow copy
lst.clear()            # remove all elements
lst + other            # concatenate (creates new list)
lst * 3                # repeat
del lst[i]             # remove by index
del lst[i:j]           # remove slice
```

### `dict` — Hash Map

```python
d = {}
d = {'a': 1, 'b': 2}
d = dict(zip(keys, values))
d = dict.fromkeys(['a', 'b', 'c'], 0)  # all values = 0

# Operations
d[key] = val                  # insert/update
d[key]                        # access — KeyError if missing!
d.get(key, default)           # safe access — returns default if missing
key in d                      # O(1) avg — check existence
del d[key]                    # remove — KeyError if missing
d.pop(key, default)           # remove and return — default if missing
d.setdefault(key, default)    # get value, set to default if missing
len(d)                        # count of entries

# Iteration (insertion-ordered since Python 3.7)
for key in d:                 # iterate keys
for key, val in d.items():    # iterate key-value pairs
for val in d.values():        # iterate values
for key in d.keys():          # iterate keys (explicit)

# Comprehension
d = {k: v for k, v in pairs}
d = {x: x**2 for x in range(10)}
d = {k: v for k, v in d.items() if v > 0}  # filter

# Merge (Python 3.9+)
merged = d1 | d2              # d2 values override d1
d1 |= d2                      # update d1 with d2

# Other
d.update(other_dict)           # merge other into d
d.copy()                       # shallow copy
list(d.keys())                 # list of keys
list(d.values())               # list of values
```

### `set` — Hash Set

```python
s = set()
s = {1, 2, 3}
s = set(lst)                   # deduplicate a list
s = set("hello")               # {'h', 'e', 'l', 'o'}

# Operations
s.add(x)              # O(1) avg
s.discard(x)          # O(1) avg — no error if missing
s.remove(x)           # O(1) avg — KeyError if missing
s.pop()               # remove and return arbitrary element
x in s                # O(1) avg
len(s)
s.clear()
s.copy()

# Set operations — return new sets
s1 & s2               # intersection
s1 | s2               # union
s1 - s2               # difference (in s1 but not s2)
s1 ^ s2               # symmetric difference

# In-place set operations
s1 &= s2              # intersection_update
s1 |= s2              # update (union)
s1 -= s2              # difference_update
s1 ^= s2              # symmetric_difference_update

# Subset / superset
s1.issubset(s2)       # s1 ⊆ s2   (or s1 <= s2)
s1.issuperset(s2)     # s1 ⊇ s2   (or s1 >= s2)
s1.isdisjoint(s2)     # no common elements
s1 < s2               # proper subset
```

### `frozenset` — Immutable Set

```python
fs = frozenset([1, 2, 3])
# Hashable — can be used as dict key or set element
seen = set()
seen.add(frozenset({1, 2}))  # set of sets
```

### `tuple` — Immutable Sequence

```python
t = (1, 2, 3)
t = 1, 2, 3           # parentheses optional
a, b, c = t           # unpacking
a, *rest = t           # a=1, rest=[2, 3]
*rest, c = t           # rest=[1, 2], c=3
t[0]                   # access
len(t)
t + (4, 5)             # concatenation
t * 2                  # repetition
t.count(x)             # count occurrences
t.index(x)             # first index

# Tuples are hashable → can be dict keys or set elements
seen = set()
seen.add((row, col))

# Comparison is lexicographic
(1, 2) < (1, 3)       # True
(1, 2) < (2, 0)       # True
# Great for multi-criteria sorting!
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
s.find("ll", start)    # search from position
s.index("ll")          # 2 (index), ValueError if not found
s.rfind("l")           # search from right
s.count("l")           # 2
s.startswith("he")     # True
s.endswith("lo")       # True
s.startswith(("he", "wo"))  # True if starts with any (tuple)
s.replace("l", "r")    # "herro" — returns new string
s.replace("l", "r", 1) # "herlo" — replace only first
s.split()              # ["hello"] — split by whitespace
s.split(",")           # split by delimiter
s.split(",", maxsplit=1) # split at most once
s.rsplit(",", maxsplit=1) # split from right
",".join(["a","b","c"])# "a,b,c"
s.strip()              # remove leading/trailing whitespace
s.lstrip(); s.rstrip() # left/right strip
s.strip("aeiou")       # strip specific characters
s.lower(); s.upper(); s.title(); s.capitalize()
s.swapcase()           # swap upper/lower
s.isdigit(); s.isalpha(); s.isalnum(); s.isspace()
s.isupper(); s.islower()
s.zfill(5)             # pad with zeros: "00042"
s.center(10); s.ljust(10); s.rjust(10) # padding

# Character operations
ord('a')   # 97
chr(97)    # 'a'
ord(c) - ord('a')  # 0-25 index

# Building strings efficiently
parts = []
for ...:
    parts.append(chunk)
result = "".join(parts)  # O(n) total — MUCH better than += in loop

# f-strings (Python 3.6+)
f"value is {x}, computed: {x*2}"
f"{x:05d}"            # zero-padded: "00042"
f"{x:.2f}"            # 2 decimal places: "3.14"
f"{x:>10}"            # right-aligned, width 10
f"{x:,}"              # thousands separator: "1,000,000"
f"{x:b}"              # binary: "1010"
```

---

## Lambda Functions

```python
# Basic syntax
square = lambda x: x * x
square(5)  # 25

# Multiple arguments
add = lambda a, b: a + b
add(3, 4)  # 7

# With default arguments
greet = lambda name="World": f"Hello, {name}!"

# Limitation: single expression only (no statements, no assignment)
# Use a named function for anything complex

# ===== Lambda with Built-in Functions =====

# sorted / sort
sorted(lst, key=lambda x: x[1])          # sort by second element
sorted(lst, key=lambda x: -x)            # descending
sorted(lst, key=lambda x: (x[0], -x[1])) # multi-criteria

# map — apply function to each element
list(map(lambda x: x ** 2, [1, 2, 3]))   # [1, 4, 9]
list(map(str, [1, 2, 3]))                # ['1', '2', '3']
list(map(lambda a, b: a + b, [1,2], [3,4])) # [4, 6] — multiple iterables

# filter — keep elements where function returns True
list(filter(lambda x: x > 0, [-1, 2, -3, 4]))  # [2, 4]
list(filter(None, [0, 1, '', 'a', [], [1]]))    # truthy values: [1, 'a', [1]]

# min / max with key
min(lst, key=lambda x: abs(x))           # closest to zero
max(d, key=d.get)                        # key with highest value

# reduce
from functools import reduce
reduce(lambda a, b: a + b, [1, 2, 3, 4])  # 10 (sum)
reduce(lambda a, b: a * b, [1, 2, 3, 4])  # 24 (product)

# ===== Lambda as Argument =====
# Use anywhere a function is expected
from collections import defaultdict
dd = defaultdict(lambda: [0, 0])  # default value is [0, 0]
```

---

## Custom Comparators & Sorting

### Key Functions (Pythonic Way)

```python
# Single criterion
sorted(words, key=len)                     # by length
sorted(words, key=str.lower)               # case-insensitive

# Descending
sorted(nums, reverse=True)
sorted(words, key=lambda w: -len(w))       # or negate for numeric

# Multi-criteria using tuples
students = [("Alice", 90), ("Bob", 85), ("Charlie", 90)]
sorted(students, key=lambda s: (-s[1], s[0]))
# Sort by grade descending, then name ascending

# Negate trick for descending (numbers only)
# Ascending:  key=lambda x: x[0]
# Descending: key=lambda x: -x[0]
# Mixed:      key=lambda x: (-x[1], x[0])  → desc by [1], asc by [0]
```

### `functools.cmp_to_key` — Old-Style Comparators

```python
from functools import cmp_to_key

# When you need a full comparator (like Java's Comparator)
# Compare function: return negative/0/positive
def compare(a, b):
    # Custom logic that's hard to express as a key
    if len(a) != len(b):
        return len(a) - len(b)  # shorter first
    return (a > b) - (a < b)    # then alphabetical

sorted(words, key=cmp_to_key(compare))

# Practical example: largest number problem
def largest_num_cmp(a, b):
    if a + b > b + a: return -1  # a should come first
    elif a + b < b + a: return 1
    return 0

nums_str = [str(n) for n in nums]
result = "".join(sorted(nums_str, key=cmp_to_key(largest_num_cmp)))
```

### Sorting Complex Objects

```python
# Sort by attribute
from operator import attrgetter, itemgetter

# itemgetter — for dicts and sequences
sorted(dicts, key=itemgetter('age'))
sorted(tuples, key=itemgetter(1))           # by second element
sorted(tuples, key=itemgetter(1, 0))        # by second, then first

# attrgetter — for objects
sorted(objects, key=attrgetter('name'))
sorted(objects, key=attrgetter('score', 'name'))

# These are faster than equivalent lambdas
```

### Maintaining Sorted Order

```python
from bisect import bisect_left, bisect_right, insort, insort_left, insort_right

# bisect_left: first position where x could be inserted
i = bisect_left(sorted_arr, x)   # first >= x

# bisect_right: position after all existing x values
j = bisect_right(sorted_arr, x)  # first > x

# insort: insert maintaining sorted order
insort(sorted_arr, x)            # O(n) due to shift, O(log n) search

# Count occurrences in sorted array
count = bisect_right(arr, x) - bisect_left(arr, x)

# Check existence in sorted array
def binary_search(arr, x):
    i = bisect_left(arr, x)
    return i < len(arr) and arr[i] == x

# bisect with key (Python 3.10+)
bisect_left(arr, x, key=lambda item: item[0])
```

---

## Classes & Objects

### Basic Class

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

# Usage
head = ListNode(1, ListNode(2, ListNode(3)))
root = TreeNode(1, TreeNode(2), TreeNode(3))
```

### Magic Methods for Containers & Sorting

```python
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    # String representation
    def __repr__(self):
        return f"Point({self.x}, {self.y})"

    # Equality (needed for use in set/dict)
    def __eq__(self, other):
        return isinstance(other, Point) and self.x == other.x and self.y == other.y

    # Hash (needed for use in set/dict — MUST be consistent with __eq__)
    def __hash__(self):
        return hash((self.x, self.y))

    # Less-than (needed for sorting, heapq, min/max)
    def __lt__(self, other):
        if self.x != other.x:
            return self.x < other.x
        return self.y < other.y

# Now works with:
points = [Point(3, 1), Point(1, 2), Point(1, 1)]
sorted(points)           # [Point(1, 1), Point(1, 2), Point(3, 1)]
heapq.heappush(heap, Point(1, 2))  # min-heap by (x, y)
{Point(1, 2), Point(3, 4)}         # set of points
{Point(1, 2): "origin"}            # dict key
```

### `@total_ordering` — Generate All Comparison Methods

```python
from functools import total_ordering

@total_ordering
class Student:
    def __init__(self, name, grade):
        self.name = name
        self.grade = grade

    def __eq__(self, other):
        return self.grade == other.grade

    def __lt__(self, other):
        return self.grade < other.grade

    # __le__, __gt__, __ge__ are auto-generated from __eq__ and __lt__
```

### `@dataclass` — Auto-Generated Methods (Python 3.7+)

```python
from dataclasses import dataclass, field

@dataclass
class Point:
    x: int
    y: int

# Auto-generates: __init__, __repr__, __eq__
p = Point(1, 2)
p.x, p.y        # attribute access
p == Point(1, 2) # True

# Ordered (for sorting/comparison)
@dataclass(order=True)
class Student:
    grade: int       # compared first
    name: str        # compared second

# Frozen (immutable, hashable — works in sets/dicts)
@dataclass(frozen=True)
class Point:
    x: int
    y: int

seen = {Point(1, 2), Point(3, 4)}  # works because frozen=True

# Default values and custom fields
@dataclass
class Graph:
    nodes: int
    edges: list = field(default_factory=list)  # mutable default
    visited: set = field(default_factory=set, repr=False)

# Custom sort key with dataclass
@dataclass(order=True)
class Task:
    sort_priority: tuple = field(init=False, repr=False)
    priority: int
    name: str

    def __post_init__(self):
        self.sort_priority = (-self.priority, self.name)  # desc priority, asc name
```

### `namedtuple` — Lightweight Immutable Class

```python
from collections import namedtuple

Point = namedtuple('Point', ['x', 'y'])
p = Point(1, 2)
p.x, p.y       # 1, 2
p[0], p[1]      # 1, 2 — also index-accessible
x, y = p        # unpacking

# Hashable, comparable (lexicographic), immutable
# Lighter weight than @dataclass(frozen=True)
```

---

## Standard Library Essentials

### `collections`

```python
from collections import defaultdict, Counter, deque, OrderedDict, ChainMap

# ===== defaultdict =====
graph = defaultdict(list)
graph[node].append(neighbor)  # no KeyError
freq = defaultdict(int)
freq[x] += 1
groups = defaultdict(set)
matrix = defaultdict(lambda: defaultdict(int))  # 2D default dict

# ===== Counter =====
counts = Counter("abracadabra")   # Counter({'a': 5, 'b': 2, ...})
counts = Counter([1, 1, 2, 3])    # Counter({1: 2, 2: 1, 3: 1})
counts.most_common(3)             # top 3 [(element, count), ...]
counts.most_common()              # all, sorted by frequency desc
counts[x]                         # frequency of x (0 if missing, no KeyError)
counts.total()                    # sum of all counts (Python 3.10+)
counts.update(more_items)         # add counts
counts.subtract(other)            # subtract counts
counts1 + counts2                 # combine (sum counts)
counts1 - counts2                 # subtract (keeps positive only)
counts1 & counts2                 # intersection (min of each)
counts1 | counts2                 # union (max of each)
list(counts.elements())           # repeat elements: ['a','a','a','a','a','b','b',...]
+counts                           # remove zero and negative counts

# ===== deque =====
dq = deque()
dq = deque([1, 2, 3])
dq.append(x)          # right — O(1)
dq.appendleft(x)      # left — O(1)
dq.pop()               # right — O(1)
dq.popleft()           # left — O(1) — THIS is why you use deque, not list
dq[0]; dq[-1]          # peek — O(1)
dq[i]                  # random access — O(n)!
dq.rotate(k)           # rotate right by k (negative = left)
dq.extend(iterable)    # extend right
dq.extendleft(iterable) # extend left (reverses order)
len(dq)
dq.count(x)
dq.remove(x)           # O(n)
dq.reverse()
dq.clear()
deque(iterable, maxlen=k)  # fixed-size: auto-discards from opposite end

# ===== OrderedDict =====
# dict preserves insertion order since Python 3.7, but OrderedDict has:
od = OrderedDict()
od.move_to_end(key)          # move to end
od.move_to_end(key, last=False) # move to beginning
od.popitem(last=True)        # pop last (LIFO) — useful for LRU/MRU

# ===== ChainMap =====
from collections import ChainMap
defaults = {'color': 'red', 'size': 10}
overrides = {'color': 'blue'}
merged = ChainMap(overrides, defaults)
merged['color']  # 'blue' — first found wins
merged['size']   # 10 — falls through to defaults
```

### `heapq` — Min-Heap

```python
import heapq

heap = []
heapq.heappush(heap, x)     # O(log n)
heapq.heappop(heap)          # O(log n) — returns and removes smallest
heap[0]                      # O(1) — peek smallest (don't modify!)
heapq.heapify(lst)           # O(n) — convert list to heap in-place
heapq.heapreplace(heap, x)  # pop then push — more efficient than separate calls
heapq.heappushpop(heap, x)  # push then pop — more efficient

heapq.nlargest(k, iterable)        # top-k largest — O(n log k)
heapq.nsmallest(k, iterable)       # top-k smallest
heapq.nlargest(k, iterable, key=fn) # with key function

# Max-heap trick: negate values
heapq.heappush(heap, -x)
max_val = -heapq.heappop(heap)

# With tuples (compared lexicographically — handy for priority)
heapq.heappush(heap, (priority, index, data))  # use index as tiebreaker

# Merge k sorted iterables — O(n log k)
merged = list(heapq.merge(sorted1, sorted2, sorted3))
merged = list(heapq.merge(*lists, key=lambda x: x[0]))  # with key
```

### `itertools`

```python
from itertools import (
    permutations, combinations, combinations_with_replacement,
    product, accumulate, chain, groupby,
    zip_longest, islice, count, cycle, repeat,
    starmap, tee, pairwise
)

# ===== Combinatorics =====
list(permutations([1,2,3]))         # all orderings: 6 items
list(permutations([1,2,3], 2))      # length-2 orderings: 6 items
list(combinations([1,2,3], 2))       # choose 2: [(1,2),(1,3),(2,3)]
list(combinations_with_replacement([1,2,3], 2)) # with repetition
list(product([0,1], repeat=3))       # cartesian: 000,001,...,111
list(product([1,2], [3,4]))          # all pairs: (1,3),(1,4),(2,3),(2,4)

# ===== Accumulation =====
list(accumulate([1,2,3,4]))          # prefix sums: [1,3,6,10]
list(accumulate([1,2,3,4], operator.mul)) # prefix products: [1,2,6,24]
list(accumulate([3,1,4,1], max))     # running max: [3,3,4,4]
list(accumulate([3,1,4,1], min))     # running min: [3,1,1,1]

# ===== Chaining & Flattening =====
list(chain([1,2], [3,4], [5]))       # flatten: [1,2,3,4,5]
list(chain.from_iterable([[1,2],[3,4]])) # flatten iterable of iterables

# ===== Grouping =====
# groupby requires SORTED input for meaningful groups!
data = sorted(data, key=key_func)
for key, group in groupby(data, key=key_func):
    items = list(group)

# ===== Zipping =====
list(zip_longest([1,2], [3,4,5], fillvalue=0))  # [(1,3),(2,4),(0,5)]

# ===== Slicing Iterators =====
list(islice(range(100), 5))          # first 5: [0,1,2,3,4]
list(islice(range(100), 2, 8, 2))    # [2,4,6]

# ===== Infinite Iterators =====
count(10)         # 10, 11, 12, 13, ...
count(0, 0.5)     # 0, 0.5, 1.0, ...
cycle([1,2,3])    # 1, 2, 3, 1, 2, 3, ...
repeat(5, 3)      # 5, 5, 5

# ===== Other =====
list(starmap(pow, [(2,3),(3,2)]))    # [8, 9] — unpacks args
a, b = tee(iterator)                 # duplicate iterator
list(pairwise([1,2,3,4]))           # [(1,2),(2,3),(3,4)] — Python 3.10+
```

### `functools`

```python
from functools import lru_cache, cache, reduce, partial, cmp_to_key

# ===== Memoization for DP =====
@lru_cache(maxsize=None)   # unbounded cache
def dp(i, j):
    ...

@cache                      # Python 3.9+ shorthand for lru_cache(maxsize=None)
def fib(n):
    if n <= 1: return n
    return fib(n-1) + fib(n-2)

# IMPORTANT: arguments must be hashable (no lists — use tuples!)
# Clear cache: fib.cache_clear()
# Cache info: fib.cache_info()

# ===== Reduce =====
from operator import xor, add, mul
reduce(xor, nums)           # XOR all elements
reduce(add, nums)            # sum all
reduce(mul, nums)            # product all
reduce(lambda a, b: a if a > b else b, nums)  # max

# With initial value
reduce(add, nums, 0)         # safe even if empty

# ===== Partial =====
from functools import partial
int_from_binary = partial(int, base=2)
int_from_binary('1010')      # 10

# ===== cmp_to_key =====
# Convert old-style cmp function to key function (see Comparators section)
```

### `math`

```python
import math

math.inf, -math.inf
math.ceil(3.2)             # 4
math.floor(3.8)            # 3
math.trunc(3.8)            # 3 (toward zero, like int())
math.sqrt(16)              # 4.0
math.isqrt(16)             # 4 (integer, Python 3.8+)
math.gcd(12, 8)            # 4
math.lcm(12, 8)            # 24 (Python 3.9+)
math.gcd(12, 8, 6)         # 2 (multiple args, Python 3.9+)
math.log(x)                # natural log
math.log2(x)               # log base 2
math.log10(x)              # log base 10
math.comb(n, k)            # n choose k (Python 3.8+)
math.perm(n, k)            # n permute k (Python 3.8+)
math.factorial(n)
math.pow(x, y)             # float result
math.copysign(x, y)        # |x| with sign of y
math.fabs(x)               # float absolute value
math.dist([x1,y1], [x2,y2]) # Euclidean distance (Python 3.8+)
```

### `sortedcontainers` (Third-Party, Common in Competitive Programming)

```python
# pip install sortedcontainers
from sortedcontainers import SortedList, SortedDict, SortedSet

# ===== SortedList =====
sl = SortedList()
sl = SortedList([5, 1, 3])        # [1, 3, 5]

sl.add(x)               # O(log n) — inserts in sorted position
sl.discard(x)            # O(log n) — remove, no error if absent
sl.remove(x)             # O(log n) — remove, ValueError if absent
sl.pop(i)                # O(log n) — remove by index
sl[i]                    # O(log n) — index access
sl[-1]                   # largest element
sl[0]                    # smallest element
sl.bisect_left(x)        # index of leftmost position for x
sl.bisect_right(x)       # index of rightmost position for x
sl.count(x)              # count of x
sl.index(x)              # first index of x
len(sl)

# SortedList with custom key
sl = SortedList(key=lambda x: -x)  # descending order

# ===== SortedDict =====
sd = SortedDict()
sd[key] = val
sd.peekitem(0)            # (smallest_key, val)
sd.peekitem(-1)           # (largest_key, val)
sd.irange(lo, hi)         # iterate keys in range [lo, hi]
sd.bisect_left(key)       # index of key position

# ===== SortedSet =====
ss = SortedSet()
ss.add(x)
ss[0]; ss[-1]             # min, max
ss.irange(lo, hi)         # iterate in range

# Use case: sliding window min/max, order statistics, median finding
```

---

## Built-in Functions

```python
# ===== Iteration =====
for i, val in enumerate(lst):       # index + value
for i, val in enumerate(lst, 1):    # start from 1
for a, b in zip(lst1, lst2):        # parallel (stops at shorter)
for a, b, c in zip(l1, l2, l3):     # three lists
reversed(lst)                        # reverse iterator (no copy!)
iter(lst)                            # explicit iterator

# ===== Aggregation =====
sum(lst)                             # sum
sum(lst, start=0)                    # with initial value
min(lst); max(lst)                   # minimum/maximum
min(lst, key=len)                    # min by custom key
min(lst, default=float('inf'))       # default if empty
min(a, b, c)                        # min of multiple values
any(lst)                             # True if any truthy
all(lst)                             # True if all truthy
any(x > 5 for x in lst)             # with generator expression
all(x > 0 for x in lst)

# ===== Transformation =====
list(map(func, iterable))            # apply func to each
list(filter(pred, iterable))         # keep where pred is True
list(zip(keys, values))              # pair up
dict(zip(keys, values))              # zip to dict

# ===== Numeric =====
abs(x)                               # absolute value
pow(base, exp)                       # base ** exp
pow(base, exp, mod)                  # modular exponentiation — O(log exp)
divmod(a, b)                         # (a // b, a % b)
round(3.14159, 2)                    # 3.14

# ===== Type Conversion =====
int(x); float(x); str(x); bool(x)
list(iterable); tuple(iterable); set(iterable)
dict(pairs)
sorted(iterable)                     # returns new sorted list

# ===== Other =====
len(x)                               # length
hash(x)                              # hash value (hashable types only)
id(x)                                # object identity
type(x)                              # type of object
isinstance(x, Type)                  # type check
range(n)                             # 0 to n-1
range(start, stop, step)             # custom range
```

---

## Common Interview Idioms

### List Comprehensions & Generator Expressions

```python
squares = [x**2 for x in range(10)]
evens = [x for x in lst if x % 2 == 0]
flat = [x for row in grid for x in row]   # flatten 2D
pairs = [(i, j) for i in range(n) for j in range(m)]

# Generator expression (lazy — no list created)
total = sum(x**2 for x in range(1000))   # no brackets!
exists = any(x > 10 for x in lst)

# Set / dict comprehensions
unique = {x for x in lst}
freq = {x: lst.count(x) for x in set(lst)}
```

### Dictionary Patterns

```python
# Frequency counter
from collections import Counter
freq = Counter(nums)                  # best way
freq = {}; [freq.__setitem__(x, freq.get(x, 0) + 1) for x in nums]  # manual

# Invert a dict
inv = {v: k for k, v in d.items()}

# Group by
from collections import defaultdict
groups = defaultdict(list)
for item in items:
    groups[key_func(item)].append(item)

# Merge dicts
merged = {**d1, **d2}          # d2 overrides d1
merged = d1 | d2               # Python 3.9+
```

### Infinity & Sentinels

```python
float('inf')           # positive infinity
float('-inf')          # negative infinity
math.inf               # alternative
# Python ints have no overflow, so sys.maxsize is rarely needed
```

### Swap & Multiple Assignment

```python
a, b = b, a           # no temp variable needed
a, b, c = 1, 2, 3     # multiple assignment
```

### Direction Arrays

```python
# 4-directional
dirs = [(0,1), (0,-1), (1,0), (-1,0)]
for dr, dc in dirs:
    nr, nc = r + dr, c + dc
    if 0 <= nr < rows and 0 <= nc < cols:
        ...

# 8-directional
dirs8 = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
```

### Modular Arithmetic

```python
MOD = 10**9 + 7
result = (a * b) % MOD
# Python handles negative mod correctly: (-7) % 3 == 2

# Modular exponentiation
pow(base, exp, MOD)    # O(log exp), built-in
```

---

## Python-Specific Gotchas

| Gotcha | Solution |
|--------|----------|
| `list.pop(0)` is O(n) | Use `collections.deque.popleft()` for O(1) |
| Mutable default arguments | `def f(lst=None): lst = lst or []` |
| `[[0]*n]*m` shares rows | Use `[[0]*n for _ in range(m)]` |
| `is` vs `==` | `is` checks identity; `==` checks equality. Use `is` only for `None` |
| Integer division `//` floors toward -∞ | `-7 // 2 == -4`. Use `int(-7/2)` for truncation toward 0 |
| Slicing creates copies | `lst[:]` is O(n), not O(1) |
| `set` and `dict` keys must be hashable | Lists are unhashable — convert to `tuple` |
| Recursion limit (default 1000) | `sys.setrecursionlimit(10000)` — or use iterative approach |
| `lru_cache` requires hashable args | Can't cache functions with list args — use tuples |
| Global variables in nested functions | Use `nonlocal` keyword to modify enclosing scope variable |
| `sort()` is in-place (returns `None`) | `sorted()` returns a new list |
| `dict.keys()` is a view, not a list | Wrap in `list()` if you need indexing |
| `heapq` is min-heap only | Negate values for max-heap |
| Comparing objects without `__lt__` | `sort`/`heapq` will raise `TypeError` — define `__lt__` |
| `Counter` subtraction | `Counter(a) - Counter(b)` drops zero/negative counts |

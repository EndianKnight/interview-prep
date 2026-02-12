# DSA in C++ — Complete Interview Refresher

A comprehensive C++ refresher for coding interviews, covering STL containers, algorithms, idioms, and common patterns.

---

## Table of Contents
- [Data Types & Sizes](#data-types--sizes)
- [STL Containers Deep Dive](#stl-containers-deep-dive)
- [STL Algorithms](#stl-algorithms)
- [Strings](#strings)
- [Iterators & Ranges](#iterators--ranges)
- [Common Interview Idioms](#common-interview-idioms)
- [Memory & Gotchas](#memory--gotchas)

---

## Data Types & Sizes

```cpp
int         // 32-bit, [-2^31, 2^31-1]
long long   // 64-bit, [-2^63, 2^63-1]
double      // 64-bit floating point
char        // 8-bit
bool        // true/false
size_t      // unsigned, platform-dependent (for sizes)

// Limits
#include <climits>
INT_MIN, INT_MAX           // -2147483648, 2147483647
LONG_LONG_MIN, LONG_LONG_MAX
#include <cfloat>
DBL_MAX, DBL_MIN
```

---

## STL Containers Deep Dive

### `vector<T>` — Dynamic Array

```cpp
#include <vector>
vector<int> v;                    // empty
vector<int> v(n, val);            // n copies of val
vector<int> v = {1, 2, 3};       // initializer list
vector<vector<int>> grid(m, vector<int>(n, 0)); // 2D grid

// Operations
v.push_back(x);          // O(1) amortized
v.pop_back();             // O(1)
v[i];                     // O(1), no bounds check
v.at(i);                  // O(1), throws if out of range
v.size();                 // O(1)
v.empty();                // O(1)
v.front(); v.back();      // first/last element
v.begin(); v.end();       // iterators
v.resize(n);              // resize
v.reserve(n);             // pre-allocate (avoids reallocation)
v.clear();                // remove all elements
v.erase(v.begin() + i);  // O(n) — remove at index
v.insert(v.begin() + i, x); // O(n) — insert at index

// Sorting
sort(v.begin(), v.end());                   // ascending
sort(v.begin(), v.end(), greater<int>());   // descending
sort(v.begin(), v.end(), [](int a, int b) { return a > b; }); // custom
```

### `string` — Mutable String

```cpp
#include <string>
string s = "hello";
s += " world";          // O(m) append
s.size(); s.length();   // same
s[i]; s.at(i);          // character access
s.substr(pos, len);     // O(len) substring
s.find("lo");           // O(n*m), returns index or string::npos
s.rfind("l");           // reverse find
s.compare(other);       // lexicographic compare
to_string(42);          // int → string
stoi("42");             // string → int
stol("123456789");      // string → long
s.push_back('!');       // append char
s.pop_back();           // remove last char
reverse(s.begin(), s.end()); // in-place reverse
```

### `unordered_map<K,V>` — Hash Map

```cpp
#include <unordered_map>
unordered_map<string, int> mp;

mp["key"] = 10;              // insert/update — creates entry if missing!
mp.at("key");                // access — throws if missing
mp.count("key");             // 0 or 1 — check existence
mp.find("key");              // returns iterator, mp.end() if missing
mp.erase("key");             // remove
mp.size();                   // count of entries

// Safe access pattern
if (mp.count("key")) { ... }
// or
auto it = mp.find("key");
if (it != mp.end()) { ... it->second ... }

// Iteration
for (auto& [key, val] : mp) { ... }  // C++17 structured bindings
```

### `unordered_set<T>` — Hash Set

```cpp
#include <unordered_set>
unordered_set<int> st;
st.insert(x);        // O(1) avg
st.count(x);         // 0 or 1
st.erase(x);         // O(1) avg
st.find(x);          // iterator
```

### `map<K,V>` — Ordered Map (Red-Black Tree)

```cpp
#include <map>
map<int, string> mp;          // sorted by key
mp.lower_bound(key);          // first >= key
mp.upper_bound(key);          // first > key
// All operations O(log n)
```

### `set<T>` — Ordered Set

```cpp
#include <set>
set<int> st;
st.insert(x);             // O(log n)
st.lower_bound(x);        // first >= x
st.upper_bound(x);        // first > x
st.erase(x);              // O(log n)
auto it = st.find(x);     // O(log n)
*st.begin();               // min element
*st.rbegin();              // max element
```

### `priority_queue<T>` — Heap

```cpp
#include <queue>
// Max-heap (default)
priority_queue<int> maxPQ;
maxPQ.push(x);     // O(log n)
maxPQ.top();        // O(1) — largest element
maxPQ.pop();        // O(log n) — remove largest

// Min-heap
priority_queue<int, vector<int>, greater<int>> minPQ;

// Custom comparator (e.g., by second element of pair)
auto cmp = [](pair<int,int>& a, pair<int,int>& b) {
    return a.second > b.second; // min-heap by second
};
priority_queue<pair<int,int>, vector<pair<int,int>>, decltype(cmp)> pq(cmp);
```

### `stack<T>` and `queue<T>`

```cpp
#include <stack>
stack<int> st;
st.push(x); st.pop(); st.top(); st.empty(); st.size();

#include <queue>
queue<int> q;
q.push(x); q.pop(); q.front(); q.back(); q.empty(); q.size();
```

### `deque<T>` — Double-Ended Queue

```cpp
#include <deque>
deque<int> dq;
dq.push_back(x); dq.push_front(x);
dq.pop_back();   dq.pop_front();
dq.front();      dq.back();
dq[i];           // O(1) random access
```

---

## STL Algorithms

```cpp
#include <algorithm>

// Sorting
sort(v.begin(), v.end());
stable_sort(v.begin(), v.end());
partial_sort(v.begin(), v.begin() + k, v.end()); // top-k

// Binary search (requires sorted range)
binary_search(v.begin(), v.end(), target);     // bool
lower_bound(v.begin(), v.end(), target);       // first >= target
upper_bound(v.begin(), v.end(), target);       // first > target

// Min / Max
*min_element(v.begin(), v.end());
*max_element(v.begin(), v.end());
auto [lo, hi] = minmax_element(v.begin(), v.end());

// Accumulate
#include <numeric>
int sum = accumulate(v.begin(), v.end(), 0);
long long sum = accumulate(v.begin(), v.end(), 0LL);

// Fill / Iota
fill(v.begin(), v.end(), 0);
iota(v.begin(), v.end(), 0);  // fills with 0, 1, 2, ...

// Reverse / Rotate
reverse(v.begin(), v.end());
rotate(v.begin(), v.begin() + k, v.end()); // rotate left by k

// Remove (erase-remove idiom)
v.erase(remove(v.begin(), v.end(), val), v.end());

// Unique (after sorting)
v.erase(unique(v.begin(), v.end()), v.end());

// Next permutation
next_permutation(v.begin(), v.end()); // modifies in-place, returns false if last
```

---

## Common Interview Idioms

### Reading All Input

```cpp
// Parse int from string
int x = stoi(s);
long long x = stoll(s);

// Split string by delimiter
#include <sstream>
string s = "hello world foo";
istringstream iss(s);
string word;
while (iss >> word) { ... }
```

### Infinity Values

```cpp
int INF = INT_MAX;
long long INF = LONG_LONG_MAX;
// Be careful with INT_MAX + 1 overflow!
// Safer: use INT_MAX / 2 or 1e9
```

### Lambda Comparators

```cpp
// Sort by second element descending
sort(v.begin(), v.end(), [](auto& a, auto& b) {
    return a.second > b.second;
});

// Priority queue with lambda
auto cmp = [](auto& a, auto& b) { return a.first > b.first; };
priority_queue<pair<int,int>, vector<pair<int,int>>, decltype(cmp)> pq(cmp);
```

### Structured Bindings (C++17)

```cpp
auto [x, y] = make_pair(1, 2);
for (auto& [key, val] : map) { ... }
auto [it, inserted] = map.insert({key, val});
```

### Bit Operations

```cpp
__builtin_popcount(x);       // count set bits (int)
__builtin_popcountll(x);     // count set bits (long long)
__builtin_clz(x);            // count leading zeros
__builtin_ctz(x);            // count trailing zeros
```

---

## Memory & Gotchas

| Gotcha | Solution |
|--------|----------|
| `unordered_map[key]` inserts 0 if key missing | Use `count()` or `find()` first |
| `vector<bool>` is special (bit-packed) | Use `vector<char>` if you need normal behavior |
| Integer overflow in `(a + b) / 2` | Use `a + (b - a) / 2` |
| Passing large containers by value | Always pass by reference (`const vector<int>&`) |
| `size()` returns `size_t` (unsigned) | `v.size() - 1` wraps to huge number if empty! Cast or check first |
| Stack overflow with deep recursion | Increase stack size or switch to iterative |
| `endl` vs `'\n'` | `'\n'` is faster (no flush) |

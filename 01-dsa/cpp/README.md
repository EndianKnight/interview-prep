# DSA in C++ — Complete Interview Refresher

A comprehensive C++ refresher for coding interviews, covering STL containers, algorithms, lambdas, comparators, classes, and common patterns.

---

## Table of Contents
- [Data Types & Sizes](#data-types--sizes)
- [STL Containers Deep Dive](#stl-containers-deep-dive)
- [Pair, Tuple & Structured Bindings](#pair-tuple--structured-bindings)
- [Lambda Functions](#lambda-functions)
- [Custom Comparators](#custom-comparators)
- [Classes & Structs](#classes--structs)
- [STL Algorithms](#stl-algorithms)
- [Numeric Algorithms](#numeric-algorithms)
- [Strings](#strings)
- [Bit Operations](#bit-operations)
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

// Type conversions
static_cast<int>(3.14);           // 3
static_cast<double>(5) / 2;      // 2.5
(int)('a');                       // 97
(char)(65);                       // 'A'
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
v.emplace_back(x);       // O(1) — constructs in-place (avoids copy)
v.pop_back();             // O(1)
v[i];                     // O(1), no bounds check
v.at(i);                  // O(1), throws if out of range
v.size();                 // O(1)
v.empty();                // O(1)
v.front(); v.back();      // first/last element
v.begin(); v.end();       // iterators
v.rbegin(); v.rend();     // reverse iterators
v.resize(n);              // resize (default-initializes new elements)
v.resize(n, val);         // resize with fill value
v.reserve(n);             // pre-allocate (avoids reallocation)
v.clear();                // remove all elements
v.erase(v.begin() + i);  // O(n) — remove at index
v.erase(v.begin() + i, v.begin() + j); // remove range [i, j)
v.insert(v.begin() + i, x); // O(n) — insert at index
v.assign(n, val);         // replace contents with n copies of val

// Sorting
sort(v.begin(), v.end());                   // ascending
sort(v.begin(), v.end(), greater<int>());   // descending
sort(v.begin(), v.end(), [](int a, int b) { return a > b; }); // custom

// Conversion
vector<int> v2(v.begin() + 1, v.begin() + 4); // subvector [1, 4)
```

### `array<T, N>` — Fixed-Size Array

```cpp
#include <array>
array<int, 5> arr = {1, 2, 3, 4, 5};
arr.size();                // compile-time constant
arr[i];                    // O(1)
arr.fill(0);               // fill with value
sort(arr.begin(), arr.end());

// Advantages over C arrays:
// - knows its size, supports iterators, can be passed by value
// - works with STL algorithms
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
s.find_first_of("aeiou");    // first vowel position
s.find_last_of("aeiou");     // last vowel position
s.find_first_not_of(" ");    // first non-space
s.compare(other);       // lexicographic compare
to_string(42);          // int → string
stoi("42");             // string → int
stol("123456789");      // string → long
stoll("1234567890123"); // string → long long
stod("3.14");           // string → double
s.push_back('!');       // append char
s.pop_back();           // remove last char
s.insert(pos, "text");  // insert substring
s.erase(pos, len);      // erase len chars from pos
s.replace(pos, len, "new"); // replace substring
reverse(s.begin(), s.end()); // in-place reverse

// Character checks
#include <cctype>
isalpha(c); isdigit(c); isalnum(c); isspace(c);
tolower(c); toupper(c);
```

### `unordered_map<K,V>` — Hash Map

```cpp
#include <unordered_map>
unordered_map<string, int> mp;

mp["key"] = 10;              // insert/update — creates entry if missing!
mp.at("key");                // access — throws if missing
mp.count("key");             // 0 or 1 — check existence
mp.find("key");              // returns iterator, mp.end() if missing
mp.erase("key");             // remove by key
mp.erase(it);                // remove by iterator
mp.size();                   // count of entries
mp.empty();
mp.clear();

// Safe access pattern
if (mp.count("key")) { ... }
// or
auto it = mp.find("key");
if (it != mp.end()) { ... it->second ... }

// Insert if not present
mp.insert({"key", 10});          // only inserts if key doesn't exist
mp.emplace("key", 10);          // same, constructs in-place
auto [it, ok] = mp.insert({"key", 10}); // ok = true if inserted

// Iteration
for (auto& [key, val] : mp) { ... }  // C++17 structured bindings
```

### `unordered_set<T>` — Hash Set

```cpp
#include <unordered_set>
unordered_set<int> st;
unordered_set<int> st = {1, 2, 3};
unordered_set<int> st(v.begin(), v.end()); // from vector

st.insert(x);        // O(1) avg
st.emplace(x);       // O(1) avg, constructs in-place
st.count(x);         // 0 or 1
st.erase(x);         // O(1) avg
st.find(x);          // iterator
st.size();
st.empty();
st.clear();
```

### `map<K,V>` — Ordered Map (Red-Black Tree)

```cpp
#include <map>
map<int, string> mp;          // sorted by key ascending
map<int, string, greater<int>> mp; // sorted descending

mp[key] = val;                // insert/update
mp.lower_bound(key);          // first >= key
mp.upper_bound(key);          // first > key
mp.begin()->first;            // smallest key
mp.rbegin()->first;           // largest key
mp.erase(mp.begin());         // erase smallest
mp.count(key);

// Range iteration
for (auto it = mp.lower_bound(lo); it != mp.upper_bound(hi); it++) {
    // iterate keys in [lo, hi]
}
// All operations O(log n)
```

### `set<T>` — Ordered Set

```cpp
#include <set>
set<int> st;
st.insert(x);             // O(log n) — duplicates ignored
st.lower_bound(x);        // first >= x
st.upper_bound(x);        // first > x
st.erase(x);              // O(log n) — removes all equal
st.erase(st.find(x));     // removes exactly one (useful for multiset)
auto it = st.find(x);     // O(log n)
*st.begin();               // min element
*st.rbegin();              // max element
st.count(x);               // 0 or 1

// Range [lo, hi]
for (auto it = st.lower_bound(lo); it != st.end() && *it <= hi; it++) { ... }
```

### `multiset<T>` — Ordered Set with Duplicates

```cpp
#include <set>
multiset<int> ms;
ms.insert(5);   ms.insert(5);   // {5, 5}
ms.count(5);                     // 2
ms.erase(5);                     // removes ALL 5s
ms.erase(ms.find(5));           // removes exactly ONE 5
*ms.begin();                     // smallest
*ms.rbegin();                    // largest

// Useful for: sliding window median, maintaining sorted order with dups
```

### `multimap<K,V>` — Ordered Map with Duplicate Keys

```cpp
#include <map>
multimap<int, string> mm;
mm.insert({1, "a"});
mm.insert({1, "b"});        // both stored
mm.count(1);                 // 2
auto [lo, hi] = mm.equal_range(1); // range of entries with key 1
for (auto it = lo; it != hi; it++) { ... }
```

### `priority_queue<T>` — Heap

```cpp
#include <queue>
// Max-heap (default)
priority_queue<int> maxPQ;
maxPQ.push(x);     // O(log n)
maxPQ.top();        // O(1) — largest element
maxPQ.pop();        // O(log n) — remove largest
maxPQ.size();
maxPQ.empty();

// Min-heap
priority_queue<int, vector<int>, greater<int>> minPQ;

// Custom comparator (e.g., by second element of pair)
auto cmp = [](pair<int,int>& a, pair<int,int>& b) {
    return a.second > b.second; // min-heap by second
};
priority_queue<pair<int,int>, vector<pair<int,int>>, decltype(cmp)> pq(cmp);

// From vector
vector<int> v = {3, 1, 4, 1, 5};
priority_queue<int> pq(v.begin(), v.end()); // O(n) heapify
```

### `stack<T>` and `queue<T>`

```cpp
#include <stack>
stack<int> st;
st.push(x); st.emplace(x); // add to top
st.pop();                   // remove top (returns void!)
st.top();                   // peek top
st.empty(); st.size();

#include <queue>
queue<int> q;
q.push(x); q.emplace(x);   // add to back
q.pop();                    // remove front (returns void!)
q.front(); q.back();        // peek front/back
q.empty(); q.size();
```

### `deque<T>` — Double-Ended Queue

```cpp
#include <deque>
deque<int> dq;
dq.push_back(x);  dq.push_front(x);  dq.emplace_back(x);
dq.pop_back();     dq.pop_front();
dq.front();        dq.back();
dq[i];             // O(1) random access
dq.size();         dq.empty();

// Supports iterators — works with sort, etc.
sort(dq.begin(), dq.end());
```

### `bitset<N>` — Fixed-Size Bit Array

```cpp
#include <bitset>
bitset<32> bs;                // all zeros
bitset<8> bs(42);             // from integer: 00101010
bitset<8> bs("10110");        // from string: 00010110

bs.set(i);        // set bit i to 1
bs.reset(i);      // set bit i to 0
bs.flip(i);       // toggle bit i
bs.test(i);       // check bit i (returns bool)
bs[i];            // access (no bounds check)
bs.count();       // number of 1-bits
bs.size();        // total bits (N)
bs.any();         // true if any bit is 1
bs.none();        // true if all bits are 0
bs.all();         // true if all bits are 1

// Bitwise operations
bs1 & bs2;  bs1 | bs2;  bs1 ^ bs2;  ~bs;
bs <<= 2;  bs >>= 1;

// Conversions
bs.to_ulong();    // to unsigned long
bs.to_string();   // to "0010..." string

// Use case: DP bitmask, sieve of Eratosthenes, subset tracking
```

---

## Pair, Tuple & Structured Bindings

### `pair<T1, T2>`

```cpp
#include <utility>
pair<int, int> p = {1, 2};
pair<int, int> p = make_pair(1, 2);
p.first;  p.second;

// Pairs compare lexicographically — great for sorting
vector<pair<int, int>> v = {{3,1}, {1,2}, {1,1}};
sort(v.begin(), v.end()); // → {1,1}, {1,2}, {3,1}

// As map key
map<pair<int,int>, int> mp;
mp[{1, 2}] = 3;

// In priority_queue
priority_queue<pair<int,int>, vector<pair<int,int>>, greater<pair<int,int>>> pq;
// min-heap: compares by first, then second
```

### `tuple<T...>`

```cpp
#include <tuple>
tuple<int, string, double> t = {1, "hello", 3.14};
auto t = make_tuple(1, "hello", 3.14);

// Access
get<0>(t);         // 1
get<1>(t);         // "hello"

// Structured bindings (C++17)
auto [x, y, z] = t;

// Comparison — lexicographic
tuple<int, int> a = {1, 2}, b = {1, 3};
a < b;  // true

// tie for assignment
int a, b;
tie(a, b) = make_pair(3, 4);   // a=3, b=4
tie(a, ignore) = make_pair(5, 6);  // a=5, ignore second
```

### Structured Bindings (C++17)

```cpp
// With pairs
auto [x, y] = make_pair(1, 2);

// With maps
for (auto& [key, val] : map) { ... }

// With insert result
auto [it, inserted] = map.insert({key, val});

// With arrays/structs
int arr[] = {1, 2, 3};
auto [a, b, c] = arr;
```

---

## Lambda Functions

```cpp
// Basic syntax
auto square = [](int x) { return x * x; };
square(5);  // 25

// With explicit return type
auto divide = [](int a, int b) -> double { return (double)a / b; };

// Captures
int x = 10;
auto by_val  = [x]()  { return x; };       // capture x by value (read-only)
auto by_ref  = [&x]() { x++; };            // capture x by reference (modifiable)
auto all_val = [=]()  { return x; };       // capture ALL by value
auto all_ref = [&]()  { x++; };            // capture ALL by reference
auto mixed   = [=, &x]() { x++; };        // all by value, x by reference

// Mutable — allows modifying captured-by-value variables
int count = 0;
auto counter = [count]() mutable { return ++count; };
counter();  // 1
counter();  // 2
// original `count` is still 0

// Generic lambda (C++14) — auto parameters
auto add = [](auto a, auto b) { return a + b; };
add(1, 2);         // 3
add(1.5, 2.5);     // 4.0
add(string("a"), string("b")); // "ab"

// Immediately invoked
int result = [](int a, int b) { return a + b; }(3, 4); // 7

// Storing in std::function
#include <functional>
function<int(int, int)> fn = [](int a, int b) { return a + b; };
```

### Lambda with STL

```cpp
// sort
sort(v.begin(), v.end(), [](int a, int b) { return a > b; });

// find_if
auto it = find_if(v.begin(), v.end(), [](int x) { return x > 10; });

// count_if
int cnt = count_if(v.begin(), v.end(), [](int x) { return x % 2 == 0; });

// transform
vector<int> out(v.size());
transform(v.begin(), v.end(), out.begin(), [](int x) { return x * 2; });

// for_each
for_each(v.begin(), v.end(), [](int& x) { x *= 2; }); // modifies in-place

// accumulate with lambda
int product = accumulate(v.begin(), v.end(), 1, [](int a, int b) { return a * b; });

// remove_if
v.erase(remove_if(v.begin(), v.end(), [](int x) { return x < 0; }), v.end());

// any_of / all_of / none_of
bool has_neg = any_of(v.begin(), v.end(), [](int x) { return x < 0; });
bool all_pos = all_of(v.begin(), v.end(), [](int x) { return x > 0; });
```

---

## Custom Comparators

### For `sort`

```cpp
// Ascending (default)
sort(v.begin(), v.end());

// Descending — built-in functor
sort(v.begin(), v.end(), greater<int>());

// Lambda
sort(v.begin(), v.end(), [](int a, int b) { return a > b; });

// Sort by absolute value
sort(v.begin(), v.end(), [](int a, int b) {
    return abs(a) < abs(b);
});

// Sort vector of vectors by first element, then second descending
sort(intervals.begin(), intervals.end(), [](auto& a, auto& b) {
    return a[0] == b[0] ? a[1] > b[1] : a[0] < b[0];
});

// Sort vector of pairs — by second desc, then first asc
sort(v.begin(), v.end(), [](auto& a, auto& b) {
    if (a.second != b.second) return a.second > b.second;
    return a.first < b.first;
});
```

### For `priority_queue`

```cpp
// Min-heap — greater means "a has lower priority if a > b"
priority_queue<int, vector<int>, greater<int>> minPQ;

// Custom: min-heap by second element of pair
auto cmp = [](pair<int,int>& a, pair<int,int>& b) {
    return a.second > b.second;  // ">" for min-heap!
};
priority_queue<pair<int,int>, vector<pair<int,int>>, decltype(cmp)> pq(cmp);

// Custom struct comparator
struct Compare {
    bool operator()(const pair<int,int>& a, const pair<int,int>& b) {
        return a.second > b.second;
    }
};
priority_queue<pair<int,int>, vector<pair<int,int>>, Compare> pq;
```

> ⚠️ **Priority queue comparator is inverted!** `>` gives min-heap, `<` gives max-heap. This is the opposite of `sort`.

### For `set` / `map`

```cpp
// Default: ascending
set<int> st;  // {1, 2, 3}

// Descending
set<int, greater<int>> st;  // {3, 2, 1}

// Custom: set of pairs sorted by second element
struct CmpBySecond {
    bool operator()(const pair<int,int>& a, const pair<int,int>& b) const {
        return a.second < b.second;  // ascending by second
    }
};
set<pair<int,int>, CmpBySecond> st;

// Lambda comparator for set (C++20 or with decltype trick)
auto cmp = [](int a, int b) { return abs(a) < abs(b); };
set<int, decltype(cmp)> st(cmp);
```

### Function Object (Functor) Pattern

```cpp
struct AbsCompare {
    bool operator()(int a, int b) const {
        return abs(a) < abs(b);
    }
};

// Use with any container or algorithm
sort(v.begin(), v.end(), AbsCompare());
set<int, AbsCompare> st;
priority_queue<int, vector<int>, AbsCompare> pq;
```

---

## Classes & Structs

### Basic Struct (Most Common in Interviews)

```cpp
struct ListNode {
    int val;
    ListNode* next;
    ListNode(int x) : val(x), next(nullptr) {}
    ListNode(int x, ListNode* next) : val(x), next(next) {}
};

struct TreeNode {
    int val;
    TreeNode* left;
    TreeNode* right;
    TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
};

// Usage
ListNode* head = new ListNode(1);
head->next = new ListNode(2);
TreeNode* root = new TreeNode(1);
```

### Operator Overloading for Containers

```cpp
// Define < for use in set/map/sort/priority_queue
struct Point {
    int x, y;

    // Less-than for ordered containers (set, map, sort)
    bool operator<(const Point& other) const {
        if (x != other.x) return x < other.x;
        return y < other.y;
    }

    // Equality
    bool operator==(const Point& other) const {
        return x == other.x && y == other.y;
    }
};

// Now works with:
set<Point> points;
points.insert({1, 2});
vector<Point> v = {{3,1}, {1,2}};
sort(v.begin(), v.end());
```

### Custom Hash for `unordered_set` / `unordered_map`

```cpp
struct Point { int x, y; };

// Custom hash
struct PointHash {
    size_t operator()(const Point& p) const {
        return hash<int>()(p.x) ^ (hash<int>()(p.y) << 16);
    }
};

// Custom equality (needed for unordered containers)
struct PointEqual {
    bool operator()(const Point& a, const Point& b) const {
        return a.x == b.x && a.y == b.y;
    }
};

unordered_set<Point, PointHash, PointEqual> visited;
unordered_map<Point, int, PointHash, PointEqual> dist;

// Shortcut: use pair<int,int> instead (but no default hash!)
// Or encode as single int: key = x * N + y
```

### Class with Member Functions

```cpp
class UnionFind {
    vector<int> parent, rank_;
public:
    UnionFind(int n) : parent(n), rank_(n, 0) {
        iota(parent.begin(), parent.end(), 0); // parent[i] = i
    }

    int find(int x) {
        if (parent[x] != x) parent[x] = find(parent[x]); // path compression
        return parent[x];
    }

    void unite(int x, int y) {
        int px = find(x), py = find(y);
        if (px == py) return;
        if (rank_[px] < rank_[py]) swap(px, py);
        parent[py] = px;
        if (rank_[px] == rank_[py]) rank_[px]++;
    }

    bool connected(int x, int y) { return find(x) == find(y); }
};

// Usage
UnionFind uf(100);
uf.unite(0, 1);
uf.connected(0, 1);  // true
```

---

## STL Algorithms

```cpp
#include <algorithm>

// ===== Sorting =====
sort(v.begin(), v.end());                        // O(n log n), unstable
stable_sort(v.begin(), v.end());                 // O(n log n), stable
partial_sort(v.begin(), v.begin() + k, v.end()); // top-k in O(n log k)
nth_element(v.begin(), v.begin() + k, v.end());  // partition around k-th element O(n)
// After nth_element: v[k] is correct, elements before are ≤, after are ≥
is_sorted(v.begin(), v.end());                   // check if sorted

// ===== Binary Search (requires sorted range) =====
binary_search(v.begin(), v.end(), target);     // bool — is target present?
lower_bound(v.begin(), v.end(), target);       // iterator to first >= target
upper_bound(v.begin(), v.end(), target);       // iterator to first > target
equal_range(v.begin(), v.end(), target);       // pair<it,it> — range of target

// ===== Searching =====
find(v.begin(), v.end(), val);                 // first occurrence (iterator)
find_if(v.begin(), v.end(), pred);             // first matching predicate
find_if_not(v.begin(), v.end(), pred);         // first NOT matching
adjacent_find(v.begin(), v.end());             // first pair of equal adjacent
search(v.begin(), v.end(), sub.begin(), sub.end()); // subsequence search

// ===== Counting =====
count(v.begin(), v.end(), val);                // count occurrences
count_if(v.begin(), v.end(), pred);            // count matching predicate

// ===== Predicates =====
all_of(v.begin(), v.end(), pred);              // true if ALL match
any_of(v.begin(), v.end(), pred);              // true if ANY match
none_of(v.begin(), v.end(), pred);             // true if NONE match

// ===== Min / Max =====
*min_element(v.begin(), v.end());
*max_element(v.begin(), v.end());
auto [lo, hi] = minmax_element(v.begin(), v.end());
min(a, b); max(a, b); min({a, b, c}); max({a, b, c}); // direct values
clamp(val, lo, hi);                            // C++17: clamp val to [lo, hi]

// ===== Modifying =====
fill(v.begin(), v.end(), 0);                   // fill with value
generate(v.begin(), v.end(), []{ return rand(); }); // fill with generator
transform(v.begin(), v.end(), out.begin(), func); // apply function
replace(v.begin(), v.end(), old_val, new_val);
replace_if(v.begin(), v.end(), pred, new_val);
swap(a, b);                                    // swap two variables
swap_ranges(v1.begin(), v1.end(), v2.begin()); // swap two ranges
copy(v.begin(), v.end(), out.begin());         // copy range
copy_if(v.begin(), v.end(), back_inserter(result), pred); // filtered copy
move(v.begin(), v.end(), out.begin());         // move range

// ===== Removing =====
// remove doesn't resize — returns new "end" iterator
auto new_end = remove(v.begin(), v.end(), val);
v.erase(new_end, v.end());                    // actual removal
// Combined: erase-remove idiom
v.erase(remove(v.begin(), v.end(), val), v.end());
v.erase(remove_if(v.begin(), v.end(), pred), v.end());

// ===== Unique =====
sort(v.begin(), v.end());
v.erase(unique(v.begin(), v.end()), v.end()); // remove consecutive duplicates

// ===== Reversing / Rotating =====
reverse(v.begin(), v.end());
rotate(v.begin(), v.begin() + k, v.end()); // rotate left by k

// ===== Permutations =====
next_permutation(v.begin(), v.end()); // modifies in-place, false if last
prev_permutation(v.begin(), v.end());
// To generate ALL permutations: sort first, then loop with next_permutation

// ===== Partitioning =====
partition(v.begin(), v.end(), pred);         // elements matching pred come first
stable_partition(v.begin(), v.end(), pred);  // preserves relative order
auto it = partition_point(v.begin(), v.end(), pred); // first not-matching

// ===== Set Operations (on sorted ranges) =====
set_union(a.begin(), a.end(), b.begin(), b.end(), back_inserter(result));
set_intersection(a.begin(), a.end(), b.begin(), b.end(), back_inserter(result));
set_difference(a.begin(), a.end(), b.begin(), b.end(), back_inserter(result));
includes(a.begin(), a.end(), b.begin(), b.end()); // a contains b?
merge(a.begin(), a.end(), b.begin(), b.end(), back_inserter(result));

// ===== Heap Operations =====
make_heap(v.begin(), v.end());             // convert to max-heap
push_heap(v.begin(), v.end());             // after push_back, restore heap
pop_heap(v.begin(), v.end());              // move max to end, then pop_back
sort_heap(v.begin(), v.end());             // heap → sorted array
is_heap(v.begin(), v.end());               // check if valid heap

// ===== For Each =====
for_each(v.begin(), v.end(), [](int& x) { x *= 2; });
```

---

## Numeric Algorithms

```cpp
#include <numeric>

// Sum
int sum = accumulate(v.begin(), v.end(), 0);
long long sum = accumulate(v.begin(), v.end(), 0LL); // avoid overflow!

// Product
int prod = accumulate(v.begin(), v.end(), 1, multiplies<int>());

// Custom operation
int xor_all = accumulate(v.begin(), v.end(), 0, [](int a, int b) { return a ^ b; });

// Prefix sums
vector<int> prefix(v.size());
partial_sum(v.begin(), v.end(), prefix.begin());
// prefix = {v[0], v[0]+v[1], v[0]+v[1]+v[2], ...}

// Adjacent differences
vector<int> diff(v.size());
adjacent_difference(v.begin(), v.end(), diff.begin());
// diff = {v[0], v[1]-v[0], v[2]-v[1], ...}

// Inner product (dot product)
int dot = inner_product(a.begin(), a.end(), b.begin(), 0);

// Iota — fill with incrementing values
iota(v.begin(), v.end(), 0);  // v = {0, 1, 2, 3, ...}
iota(v.begin(), v.end(), 1);  // v = {1, 2, 3, 4, ...}

// GCD / LCM (C++17)
int g = gcd(12, 8);  // 4
int l = lcm(12, 8);  // 24

// Reduce (C++17, parallel-friendly accumulate)
int sum = reduce(v.begin(), v.end());
int sum = reduce(v.begin(), v.end(), 0, plus<>());

// Inclusive / exclusive scan (C++17)
inclusive_scan(v.begin(), v.end(), out.begin()); // like partial_sum
exclusive_scan(v.begin(), v.end(), out.begin(), 0); // excludes current
```

---

## Bit Operations

```cpp
// Built-in GCC intrinsics
__builtin_popcount(x);       // count set bits (int)
__builtin_popcountll(x);     // count set bits (long long)
__builtin_clz(x);            // count leading zeros (int)
__builtin_ctz(x);            // count trailing zeros (int)
__builtin_clzll(x);          // leading zeros (long long)
__builtin_ctzll(x);          // trailing zeros (long long)

// Common bit operations
x & (x - 1);      // clear lowest set bit
x & (-x);         // isolate lowest set bit
x | (x + 1);      // set lowest unset bit
(1 << k);         // k-th power of 2
(x >> k) & 1;     // check k-th bit
x | (1 << k);     // set k-th bit
x & ~(1 << k);    // clear k-th bit
x ^ (1 << k);     // toggle k-th bit

// For long long, use (1LL << k) to avoid overflow
```

---

## Common Interview Idioms

### Reading / Parsing Input

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

// Split by custom delimiter
string s = "a,b,c";
istringstream iss(s);
string token;
while (getline(iss, token, ',')) { ... }
```

### Infinity Values

```cpp
int INF = INT_MAX;
long long INF = LONG_LONG_MAX;
// Be careful with INT_MAX + 1 overflow!
// Safer: use INT_MAX / 2 or 1e9 as sentinel
int INF = 1e9;
long long INF = 1e18;
```

### Useful Patterns

```cpp
// Swap without temp
swap(a, b);

// Integer ceiling division (positive numbers)
int ceil_div = (a + b - 1) / b;

// Modular arithmetic
(a % MOD + MOD) % MOD;  // handle negative remainders

// 2D direction arrays
int dx[] = {0, 0, 1, -1};
int dy[] = {1, -1, 0, 0};
// Or combined:
int dirs[] = {0, 1, 0, -1, 0}; // 4 directions from consecutive pairs

// 8 directions
int dx8[] = {-1,-1,-1,0,0,1,1,1};
int dy8[] = {-1,0,1,-1,1,-1,0,1};
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
| `priority_queue` comparator is **inverted** | `>` gives min-heap, `<` gives max-heap |
| `erase` on `unordered_map` during iteration | Undefined behavior — use `it = mp.erase(it)` |
| `struct` members uninitialized by default | Always use initializer list or default values |
| No default hash for `pair`/`tuple` in `unordered_*` | Use custom hash or `map`/`set` instead |

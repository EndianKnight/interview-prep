# DSA in Java — Complete Interview Refresher

A comprehensive Java refresher for coding interviews, covering Collections framework, algorithms, lambdas, comparators, classes, and common patterns.

---

## Table of Contents
- [Primitive Types & Wrappers](#primitive-types--wrappers)
- [Collections Deep Dive](#collections-deep-dive)
- [Strings](#strings)
- [Arrays Utility](#arrays-utility)
- [Lambda Functions & Functional Interfaces](#lambda-functions--functional-interfaces)
- [Comparable vs Comparator](#comparable-vs-comparator)
- [Classes, Records & Objects](#classes-records--objects)
- [Collections & Arrays Algorithms](#collections--arrays-algorithms)
- [Streams](#streams)
- [Common Interview Idioms](#common-interview-idioms)
- [Java-Specific Gotchas](#java-specific-gotchas)

---

## Primitive Types & Wrappers

```java
int       // 32-bit, [-2^31, 2^31-1]     → Integer
long      // 64-bit                       → Long
double    // 64-bit floating point        → Double
char      // 16-bit Unicode               → Character
boolean   // true/false                   → Boolean
byte      // 8-bit                        → Byte
short     // 16-bit                       → Short
float     // 32-bit floating point        → Float

// Limits
Integer.MIN_VALUE  // -2147483648
Integer.MAX_VALUE  //  2147483647
Long.MIN_VALUE, Long.MAX_VALUE
Double.MAX_VALUE, Double.MIN_VALUE

// Conversions
Integer.parseInt("42")         // String → int
Long.parseLong("123456789")    // String → long
Double.parseDouble("3.14")     // String → double
String.valueOf(42)             // int → String
Integer.toString(42)           // int → String
(char)('a' + 3)                // → 'd'
(int)'a'                       // → 97

// Character utilities
Character.isLetter(c);    Character.isDigit(c);
Character.isLetterOrDigit(c);  Character.isWhitespace(c);
Character.toLowerCase(c);  Character.toUpperCase(c);
```

---

## Collections Deep Dive

### `ArrayList<T>` — Dynamic Array

```java
import java.util.*;

List<Integer> list = new ArrayList<>();
List<Integer> list = new ArrayList<>(Arrays.asList(1, 2, 3));
List<Integer> list = new ArrayList<>(List.of(1, 2, 3)); // Java 9+
List<Integer> list = new ArrayList<>(Collections.nCopies(n, 0)); // n zeros

// Operations
list.add(x);                   // O(1) amortized — add to end
list.add(index, x);            // O(n) insert at index
list.get(i);                   // O(1)
list.set(i, x);                // O(1)
list.remove(index);            // O(n), removes by index (returns removed)
list.remove(Integer.valueOf(x)); // O(n), removes first occurrence of value
list.size();
list.isEmpty();
list.contains(x);              // O(n)
list.indexOf(x);               // O(n), -1 if absent
list.lastIndexOf(x);           // O(n)
list.subList(from, to);        // view (not copy), [from, to)
list.toArray(new Integer[0]);  // convert to array
list.clear();
list.addAll(otherList);        // append all

// Sorting
Collections.sort(list);
list.sort(Comparator.naturalOrder());
list.sort(Comparator.reverseOrder());
list.sort((a, b) -> Integer.compare(a, b));  // custom
list.sort(Comparator.comparingInt(a -> a[0])); // safer — no overflow
```

### `HashMap<K,V>` — Hash Map

```java
Map<String, Integer> map = new HashMap<>();

map.put("key", 10);                    // insert/update
map.get("key");                        // null if missing
map.getOrDefault("key", 0);           // default if missing
map.containsKey("key");               // boolean
map.containsValue(10);                // O(n)
map.remove("key");                     // remove
map.size();
map.isEmpty();
map.clear();
map.keySet();                          // Set<K> view
map.values();                          // Collection<V> view
map.entrySet();                        // Set<Entry<K,V>> view

// Frequency counting patterns
map.merge(key, 1, Integer::sum);                    // increment counter
map.put(key, map.getOrDefault(key, 0) + 1);       // alternative
map.compute(key, (k, v) -> v == null ? 1 : v + 1); // compute

// putIfAbsent / computeIfAbsent
map.putIfAbsent(key, 0);                            // set only if absent
map.computeIfAbsent(key, k -> new ArrayList<>());   // lazy init list

// Iteration
for (Map.Entry<String, Integer> e : map.entrySet()) {
    String key = e.getKey();
    int val = e.getValue();
}
for (var e : map.entrySet()) { ... }  // Java 10+ var
map.forEach((k, v) -> { ... });

// Replace
map.replace(key, newVal);              // only if key exists
map.replaceAll((k, v) -> v * 2);       // transform all values
```

### `HashSet<T>` — Hash Set

```java
Set<Integer> set = new HashSet<>();
Set<Integer> set = new HashSet<>(Arrays.asList(1, 2, 3));
Set<Integer> set = new HashSet<>(list); // from list

set.add(x);             // O(1) avg — returns false if already present
set.contains(x);        // O(1) avg
set.remove(x);          // O(1) avg — returns false if absent
set.size();
set.isEmpty();
set.clear();
set.toArray(new Integer[0]);

// Set operations
set1.retainAll(set2);    // intersection (modifies set1!)
set1.addAll(set2);       // union (modifies set1!)
set1.removeAll(set2);    // difference (modifies set1!)
set1.containsAll(set2);  // subset check

// Immutable
Set<Integer> immutable = Set.of(1, 2, 3); // Java 9+
```

### `LinkedHashMap<K,V>` — Insertion-Ordered Map

```java
// Preserves insertion order during iteration
Map<String, Integer> map = new LinkedHashMap<>();
map.put("c", 3); map.put("a", 1); map.put("b", 2);
// Iteration order: c, a, b (insertion order)

// Access-order mode (for LRU cache!)
Map<String, Integer> lru = new LinkedHashMap<>(16, 0.75f, true);
// Last accessed key moves to end

// LRU Cache implementation
class LRUCache<K, V> extends LinkedHashMap<K, V> {
    private final int capacity;

    LRUCache(int capacity) {
        super(capacity, 0.75f, true); // access-order = true
        this.capacity = capacity;
    }

    @Override
    protected boolean removeEldestEntry(Map.Entry<K, V> eldest) {
        return size() > capacity;
    }
}
```

### `LinkedHashSet<T>` — Insertion-Ordered Set

```java
Set<Integer> set = new LinkedHashSet<>();
// Maintains insertion order
// Otherwise same API as HashSet
```

### `TreeMap<K,V>` — Sorted Map (Red-Black Tree)

```java
TreeMap<Integer, String> tm = new TreeMap<>();
TreeMap<Integer, String> tm = new TreeMap<>(Comparator.reverseOrder()); // descending

tm.put(key, val);
tm.floorKey(key);        // greatest key ≤ key (or null)
tm.ceilingKey(key);      // smallest key ≥ key (or null)
tm.lowerKey(key);        // greatest key < key
tm.higherKey(key);       // smallest key > key
tm.firstKey();           // minimum key
tm.lastKey();            // maximum key
tm.firstEntry();         // min entry
tm.lastEntry();          // max entry
tm.pollFirstEntry();     // remove and return min entry
tm.pollLastEntry();      // remove and return max entry
tm.subMap(from, to);     // range view [from, to)
tm.subMap(from, true, to, true); // inclusive range [from, to]
tm.headMap(to);          // keys < to
tm.tailMap(from);        // keys >= from
tm.descendingMap();      // reversed view
// All operations O(log n)
```

### `TreeSet<T>` — Sorted Set

```java
TreeSet<Integer> ts = new TreeSet<>();
TreeSet<Integer> ts = new TreeSet<>(Comparator.reverseOrder()); // descending

ts.add(x);
ts.floor(x);      // greatest ≤ x (or null)
ts.ceiling(x);    // smallest ≥ x (or null)
ts.lower(x);      // greatest < x
ts.higher(x);     // smallest > x
ts.first();        // min
ts.last();         // max
ts.pollFirst();    // remove and return min
ts.pollLast();     // remove and return max
ts.subSet(from, to);       // range [from, to)
ts.subSet(from, true, to, true); // inclusive [from, to]
ts.headSet(to);    // elements < to
ts.tailSet(from);  // elements >= from
ts.descendingSet(); // reversed view
// All operations O(log n)
```

### `PriorityQueue<T>` — Heap

```java
// Min-heap (default)
PriorityQueue<Integer> minPQ = new PriorityQueue<>();
minPQ.offer(x);      // O(log n) — add
minPQ.peek();         // O(1) — view smallest
minPQ.poll();         // O(log n) — remove and return smallest
minPQ.size();
minPQ.isEmpty();
minPQ.contains(x);   // O(n) — linear search
minPQ.remove(x);     // O(n) — remove specific element
minPQ.toArray();

// Max-heap
PriorityQueue<Integer> maxPQ = new PriorityQueue<>(Collections.reverseOrder());

// Custom comparator
PriorityQueue<int[]> pq = new PriorityQueue<>(
    Comparator.comparingInt(a -> a[1])  // min-heap by second element
);

// Multi-criteria
PriorityQueue<int[]> pq = new PriorityQueue<>(
    Comparator.comparingInt((int[] a) -> a[0])
              .thenComparingInt(a -> a[1])
);

// From collection
PriorityQueue<Integer> pq = new PriorityQueue<>(existingList);
```

### `ArrayDeque<T>` — Stack & Queue (Preferred)

```java
// As Stack (LIFO)
Deque<Integer> stack = new ArrayDeque<>();
stack.push(x);        // push to front
stack.pop();           // pop from front — throws if empty
stack.peek();          // view front — null if empty
stack.isEmpty();
stack.size();

// As Queue (FIFO)
Deque<Integer> queue = new ArrayDeque<>();
queue.offer(x);        // add to back
queue.poll();           // remove from front — null if empty
queue.peek();           // view front — null if empty

// As Deque (double-ended)
Deque<Integer> dq = new ArrayDeque<>();
dq.offerFirst(x); dq.offerLast(x);
dq.pollFirst();    dq.pollLast();     // null if empty
dq.peekFirst();    dq.peekLast();     // null if empty
dq.removeFirst();  dq.removeLast();   // throws if empty
dq.getFirst();     dq.getLast();      // throws if empty
```

> **Note:** Prefer `ArrayDeque` over `Stack` (legacy, synchronized) and `LinkedList` (more GC overhead).

### `LinkedList<T>` — Doubly-Linked List

```java
LinkedList<Integer> ll = new LinkedList<>();
ll.addFirst(x); ll.addLast(x);
ll.removeFirst(); ll.removeLast();
ll.getFirst(); ll.getLast();
ll.get(i);                 // O(n) — random access is slow!
ll.add(index, x);          // O(n)
// Also implements Deque and List interfaces
// Use only when you need O(1) insertion/deletion at known positions (rare)
```

### `BitSet` — Dynamic Bit Array

```java
BitSet bs = new BitSet();        // grows automatically
BitSet bs = new BitSet(100);     // initial capacity hint

bs.set(i);             // set bit i to 1
bs.set(i, false);      // set bit i to 0
bs.clear(i);           // set bit i to 0
bs.flip(i);            // toggle bit i
bs.get(i);             // check bit i (returns boolean)
bs.cardinality();      // count of set bits
bs.isEmpty();          // true if no bits set
bs.length();           // index of highest set bit + 1
bs.size();             // internal storage size in bits

// Range operations
bs.set(from, to);      // set bits [from, to)
bs.clear(from, to);    // clear bits [from, to)
bs.flip(from, to);     // flip bits [from, to)

// Bitwise operations (modify in-place)
bs1.and(bs2);          // intersection
bs1.or(bs2);           // union
bs1.xor(bs2);          // symmetric difference
bs1.andNot(bs2);       // bs1 & ~bs2 (difference)

// Iteration over set bits
for (int i = bs.nextSetBit(0); i >= 0; i = bs.nextSetBit(i + 1)) {
    // process bit i
}

// Use case: sieve of Eratosthenes, DP bitmask, visited tracking
```

---

## Strings

```java
String s = "hello";
s.length();
s.charAt(i);
s.substring(start, end);     // [start, end) — O(n)
s.substring(start);           // from start to end
s.indexOf("ll");             // first occurrence, -1 if not found
s.indexOf('l', fromIndex);   // search from position
s.lastIndexOf("l");
s.contains("ell");
s.equals(other);             // always use .equals(), never ==
s.equalsIgnoreCase(other);
s.compareTo(other);          // lexicographic (-/0/+)
s.toCharArray();             // String → char[]
new String(charArray);       // char[] → String
new String(charArray, offset, count); // partial array → String
s.split(",");                // split by regex delimiter
s.split(",", limit);         // limit number of splits
String.join(",", list);      // join list with delimiter
String.join(",", "a", "b");  // join varargs
s.trim();                    // remove leading/trailing whitespace
s.strip();                   // Unicode-aware trim (Java 11+)
s.toLowerCase(); s.toUpperCase();
s.replace('l', 'r');         // replace all char occurrences
s.replace("ll", "rr");       // replace all string occurrences
s.replaceAll("[aeiou]", ""); // regex replace
s.startsWith("he"); s.endsWith("lo");
s.isEmpty();                 // true if length == 0
s.isBlank();                 // true if only whitespace (Java 11+)
s.repeat(3);                 // "hellohellohello" (Java 11+)
s.chars();                   // IntStream of char values
s.codePoints();              // IntStream of Unicode code points

// StringBuilder (mutable — use in loops!)
StringBuilder sb = new StringBuilder();
sb.append("hello");
sb.append(' ');
sb.append(42);
sb.insert(0, "prefix");
sb.delete(start, end);
sb.deleteCharAt(i);
sb.setCharAt(i, c);
sb.reverse();
sb.toString();               // convert back to String
sb.length();
sb.charAt(i);
```

---

## Arrays Utility

```java
import java.util.Arrays;

int[] arr = new int[n];           // default 0
int[] arr = {1, 2, 3};
int[][] grid = new int[m][n];
boolean[] visited = new boolean[n]; // default false
char[] chars = new char[n];        // default '\0'

// Sorting
Arrays.sort(arr);                 // O(n log n) — primitive uses dual-pivot quicksort
Arrays.sort(arr, from, to);      // sort subarray [from, to)
Arrays.sort(objArr, comparator); // objects only — uses TimSort (stable)
Arrays.parallelSort(arr);        // parallel sort for large arrays

// Fill
Arrays.fill(arr, val);           // fill entire array
Arrays.fill(arr, from, to, val); // fill range [from, to)

// Copy
Arrays.copyOf(arr, newLen);      // copy with new length (truncate or pad)
Arrays.copyOfRange(arr, from, to); // copy range [from, to)
arr.clone();                      // shallow copy

// Comparison
Arrays.equals(a, b);             // content equality (1D)
Arrays.deepEquals(a2d, b2d);     // deep equality (2D+)

// Search
Arrays.binarySearch(arr, target); // sorted array — returns index or -(insertion)-1

// Conversion
Arrays.toString(arr);            // "[1, 2, 3]" for debugging
Arrays.deepToString(grid);      // for 2D arrays
Arrays.asList(objArr);           // array → fixed-size List (primitives won't work!)
Arrays.stream(arr);              // int[] → IntStream

// Streams for computation
Arrays.stream(arr).sum();
Arrays.stream(arr).min().getAsInt();
Arrays.stream(arr).max().getAsInt();
Arrays.stream(arr).average().getAsDouble();

// 2D array sort
Arrays.sort(intervals, Comparator.comparingInt(a -> a[0]));
Arrays.sort(intervals, (a, b) -> a[0] != b[0] ? a[0] - b[0] : a[1] - b[1]);
```

---

## Lambda Functions & Functional Interfaces

### Lambda Syntax

```java
// No parameters
Runnable r = () -> System.out.println("hello");

// Single parameter (parentheses optional)
Function<Integer, Integer> square = x -> x * x;

// Multiple parameters
BiFunction<Integer, Integer, Integer> add = (a, b) -> a + b;

// Block body
Comparator<String> cmp = (a, b) -> {
    if (a.length() != b.length()) return a.length() - b.length();
    return a.compareTo(b);
};
```

### Core Functional Interfaces (`java.util.function`)

```java
// Predicate<T> — takes T, returns boolean
Predicate<Integer> isEven = n -> n % 2 == 0;
isEven.test(4);           // true
isEven.negate();           // opposite
isEven.and(isPositive);    // combine with AND
isEven.or(isOdd);          // combine with OR

// Function<T, R> — takes T, returns R
Function<String, Integer> len = String::length;
len.apply("hello");       // 5
len.andThen(x -> x * 2);  // compose: length * 2
len.compose(otherFn);     // compose: apply other first

// Consumer<T> — takes T, returns void
Consumer<String> printer = System.out::println;
printer.accept("hello");

// Supplier<T> — takes nothing, returns T
Supplier<List<Integer>> listFactory = ArrayList::new;
listFactory.get();        // returns new ArrayList

// BiFunction<T, U, R>, BiPredicate<T, U>, BiConsumer<T, U>
BiFunction<Integer, Integer, Integer> add = Integer::sum;
BiPredicate<String, String> eq = String::equals;

// UnaryOperator<T> — Function<T, T>
UnaryOperator<String> upper = String::toUpperCase;

// BinaryOperator<T> — BiFunction<T, T, T>
BinaryOperator<Integer> max = Integer::max;
```

### Method References

```java
// Static method
Function<String, Integer> parse = Integer::parseInt;

// Instance method on parameter
Function<String, Integer> len = String::length;  // s -> s.length()

// Instance method on specific object
Consumer<String> printer = System.out::println;

// Constructor
Supplier<ArrayList<Integer>> factory = ArrayList::new;
Function<Integer, int[]> arrFactory = int[]::new;

// Common uses
list.forEach(System.out::println);
list.stream().map(String::toLowerCase).toList();
list.stream().filter(Objects::nonNull).toList();
```

---

## Comparable vs Comparator

### `Comparable<T>` — Natural Ordering (on the class itself)

```java
class Student implements Comparable<Student> {
    String name;
    int grade;

    Student(String name, int grade) {
        this.name = name;
        this.grade = grade;
    }

    @Override
    public int compareTo(Student other) {
        return Integer.compare(this.grade, other.grade); // ascending by grade
    }
}

// Now works directly with:
Collections.sort(students);
TreeSet<Student> ts = new TreeSet<>();
PriorityQueue<Student> pq = new PriorityQueue<>();
```

### `Comparator<T>` — External, Reusable Comparator

```java
// Lambda
Comparator<Student> byName = (a, b) -> a.name.compareTo(b.name);

// Comparator.comparing (preferred — safer, cleaner)
Comparator<Student> byGrade = Comparator.comparingInt(s -> s.grade);
Comparator<Student> byName = Comparator.comparing(s -> s.name);

// Multi-criteria chaining
Comparator<Student> cmp = Comparator
    .comparingInt((Student s) -> s.grade)
    .thenComparing(s -> s.name);                 // ascending grade, then name

// Reversed
Comparator<Student> descGrade = Comparator
    .comparingInt((Student s) -> s.grade)
    .reversed();

// Null-safe
Comparator<Student> nullSafe = Comparator.nullsFirst(
    Comparator.comparing(s -> s.name)
);

// Using with collections
students.sort(byGrade);
students.sort(byGrade.reversed());
TreeSet<Student> ts = new TreeSet<>(byGrade);
PriorityQueue<Student> pq = new PriorityQueue<>(byGrade);

// For arrays
Arrays.sort(intervals, Comparator.comparingInt(a -> a[0]));
Arrays.sort(points, Comparator.<int[]>comparingInt(a -> a[0])
                               .thenComparingInt(a -> a[1]));
```

> ⚠️ **Never use `(a, b) -> a - b`** for comparators — it overflows! Use `Integer.compare(a, b)` or `Comparator.comparingInt()`.

---

## Classes, Records & Objects

### Custom Class with `equals` + `hashCode`

```java
class Point {
    int x, y;

    Point(int x, int y) { this.x = x; this.y = y; }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof Point p)) return false;
        return x == p.x && y == p.y;
    }

    @Override
    public int hashCode() {
        return Objects.hash(x, y);  // or: 31 * x + y
    }

    @Override
    public String toString() {
        return "(" + x + ", " + y + ")";
    }
}

// Now works in HashMap/HashSet:
Set<Point> visited = new HashSet<>();
visited.add(new Point(1, 2));
visited.contains(new Point(1, 2));  // true (because equals/hashCode)
```

> ⚠️ **Contract:** If `a.equals(b)` then `a.hashCode() == b.hashCode()`. Always override both together.

### `record` (Java 16+)

```java
// Records auto-generate constructor, equals, hashCode, toString, getters
record Point(int x, int y) {}

Point p = new Point(1, 2);
p.x();        // 1 (getter, not field access)
p.y();        // 2

// Works directly in HashMap, HashSet, etc.
Set<Point> visited = new HashSet<>();
visited.add(new Point(1, 2));

// Records can implement interfaces
record Edge(int u, int v, int weight) implements Comparable<Edge> {
    @Override
    public int compareTo(Edge other) {
        return Integer.compare(this.weight, other.weight);
    }
}

// Custom record with validation
record Interval(int start, int end) {
    Interval {  // compact constructor
        if (start > end) throw new IllegalArgumentException();
    }
}
```

### Common Node Classes

```java
// Linked list node
class ListNode {
    int val;
    ListNode next;
    ListNode(int val) { this.val = val; }
    ListNode(int val, ListNode next) { this.val = val; this.next = next; }
}

// Tree node
class TreeNode {
    int val;
    TreeNode left, right;
    TreeNode(int val) { this.val = val; }
}

// Graph node (adjacency list is more common)
Map<Integer, List<Integer>> graph = new HashMap<>();
graph.computeIfAbsent(node, k -> new ArrayList<>()).add(neighbor);
```

### Enum

```java
enum Direction {
    UP(0, -1), DOWN(0, 1), LEFT(-1, 0), RIGHT(1, 0);

    final int dx, dy;
    Direction(int dx, int dy) { this.dx = dx; this.dy = dy; }
}

for (Direction d : Direction.values()) {
    int nr = r + d.dx, nc = c + d.dy;
}
```

---

## Collections & Arrays Algorithms

```java
import java.util.Collections;

// ===== Collections utility =====
Collections.sort(list);                        // sort
Collections.sort(list, comparator);            // sort with comparator
Collections.reverse(list);                     // reverse in-place
Collections.swap(list, i, j);                 // swap elements
Collections.rotate(list, distance);            // rotate right by distance
Collections.shuffle(list);                     // random shuffle
Collections.frequency(list, obj);              // count of obj
Collections.min(list);                         // minimum element
Collections.max(list);                         // maximum element
Collections.min(list, comparator);             // min with comparator
Collections.binarySearch(list, key);           // sorted list — returns index
Collections.nCopies(n, obj);                   // immutable list of n copies
Collections.disjoint(c1, c2);                  // true if no common elements
Collections.unmodifiableList(list);            // read-only wrapper
Collections.synchronizedList(list);            // thread-safe wrapper
Collections.singletonList(obj);                // immutable single-element list
Collections.emptyList();                       // immutable empty list

// ===== Immutable factories (Java 9+) =====
List.of(1, 2, 3);
Set.of("a", "b", "c");
Map.of("key1", 1, "key2", 2);
Map.ofEntries(Map.entry("k1", 1), Map.entry("k2", 2));

// ===== Arrays utility =====
Arrays.sort(arr);
Arrays.sort(arr, from, to);
Arrays.fill(arr, val);
Arrays.copyOf(arr, newLen);
Arrays.binarySearch(arr, target);
Arrays.equals(a, b);
Arrays.toString(arr);

// ===== Integer / Math =====
Math.max(a, b); Math.min(a, b);
Math.abs(x);
Math.pow(base, exp);          // returns double
Math.sqrt(x);
Math.cbrt(x);                 // cube root
Math.ceil(x); Math.floor(x);
Math.round(x);                // rounds to long
Math.log(x); Math.log10(x);
Math.floorDiv(a, b);          // floor division (handles negatives)
Math.floorMod(a, b);          // floor modulus (always non-negative)
(int) Math.ceil((double) a / b);  // ceiling division
(a + b - 1) / b;                  // integer ceiling division (positive only)

// ===== Bit operations =====
Integer.bitCount(n);           // popcount
Integer.highestOneBit(n);      // highest power of 2 ≤ n
Integer.lowestOneBit(n);       // lowest power of 2 dividing n
Integer.numberOfLeadingZeros(n);
Integer.numberOfTrailingZeros(n);
Integer.reverse(n);            // reverse all bits
Integer.rotateLeft(n, dist);
Integer.rotateRight(n, dist);
Long.bitCount(n);              // for long
```

---

## Streams

```java
import java.util.stream.*;

// ===== Creating streams =====
list.stream();                           // from collection
Arrays.stream(arr);                      // from array
Stream.of(1, 2, 3);                     // from values
IntStream.range(0, n);                   // [0, n)
IntStream.rangeClosed(1, n);             // [1, n]
Stream.empty();                          // empty stream

// ===== Intermediate operations (lazy) =====
.filter(x -> x > 0)                     // keep matching
.map(x -> x * 2)                        // transform
.flatMap(list -> list.stream())          // flatten nested streams
.mapToInt(Integer::intValue)             // → IntStream
.distinct()                              // remove duplicates
.sorted()                               // natural order
.sorted(Comparator.reverseOrder())       // custom order
.limit(n)                                // take first n
.skip(n)                                 // skip first n
.peek(System.out::println)              // debug — don't use for side effects

// ===== Terminal operations =====
.collect(Collectors.toList())            // → List
.toList()                                // Java 16+ shorthand
.toArray()                               // → Object[]
.toArray(Integer[]::new)                 // → Integer[]
.forEach(System.out::println)            // consume
.count()                                 // count elements
.sum()                                   // IntStream only
.min()                                   // OptionalInt
.max()                                   // OptionalInt
.average()                               // OptionalDouble
.findFirst()                             // Optional<T>
.findAny()                               // Optional<T>
.anyMatch(pred)                          // boolean
.allMatch(pred)                          // boolean
.noneMatch(pred)                         // boolean

// ===== Reduce =====
int sum = stream.reduce(0, Integer::sum);
Optional<Integer> max = stream.reduce(Integer::max);

// ===== Collectors =====
.collect(Collectors.toList())
.collect(Collectors.toSet())
.collect(Collectors.toMap(keyFn, valFn))
.collect(Collectors.toMap(keyFn, valFn, (v1, v2) -> v1)) // merge on conflict
.collect(Collectors.joining(", "))       // String joining
.collect(Collectors.groupingBy(keyFn))   // Map<K, List<V>>
.collect(Collectors.groupingBy(keyFn, Collectors.counting())) // Map<K, Long>
.collect(Collectors.partitioningBy(pred)) // Map<Boolean, List<V>>
.collect(Collectors.toUnmodifiableList()) // immutable result

// ===== Examples =====
// Frequency map
Map<Integer, Long> freq = list.stream()
    .collect(Collectors.groupingBy(x -> x, Collectors.counting()));

// Group anagrams
Map<String, List<String>> groups = words.stream()
    .collect(Collectors.groupingBy(w -> {
        char[] c = w.toCharArray();
        Arrays.sort(c);
        return new String(c);
    }));

// Flatten 2D list
List<Integer> flat = grid.stream()
    .flatMap(List::stream)
    .collect(Collectors.toList());

// IntStream to int[]
int[] arr = IntStream.range(0, n).toArray();

// int[] to List<Integer>
List<Integer> list = Arrays.stream(arr).boxed().collect(Collectors.toList());
```

---

## Common Interview Idioms

### Infinity

```java
int INF = Integer.MAX_VALUE;
long INF = Long.MAX_VALUE;
// Safer sentinel (avoids overflow):
int INF = (int) 1e9;
long INF = (long) 1e18;
```

### Direction Arrays

```java
int[][] dirs = {{0,1}, {0,-1}, {1,0}, {-1,0}};         // 4-directional
int[][] dirs8 = {{-1,-1},{-1,0},{-1,1},{0,-1},{0,1},{1,-1},{1,0},{1,1}}; // 8-dir

for (int[] d : dirs) {
    int nr = r + d[0], nc = c + d[1];
    if (nr >= 0 && nr < rows && nc >= 0 && nc < cols) { ... }
}
```

### Swap

```java
// Arrays
int temp = arr[i]; arr[i] = arr[j]; arr[j] = temp;

// Collections
Collections.swap(list, i, j);

// No built-in variable swap — use temp
```

### Modular Arithmetic

```java
int MOD = 1_000_000_007;
long result = ((long) a * b) % MOD;
long result = ((a % MOD) + MOD) % MOD; // handle negative
```

---

## Java-Specific Gotchas

| Gotcha | Solution |
|--------|----------|
| `Integer` comparison with `==` | Use `.equals()` — `==` only works for -128 to 127 due to caching |
| String comparison with `==` | Always use `.equals()` |
| `(a - b)` overflow in comparators | Use `Integer.compare(a, b)` or `Comparator.comparingInt()` |
| `PriorityQueue` is min-heap | Use `Collections.reverseOrder()` for max-heap |
| `Stack` class is legacy | Use `ArrayDeque` instead |
| `List.remove(int)` vs `List.remove(Object)` | `list.remove(0)` by index; `list.remove(Integer.valueOf(0))` by value |
| `Arrays.asList()` returns fixed-size list | Wrap in `new ArrayList<>(Arrays.asList(...))` to make mutable |
| Generic arrays not allowed | Use `List<List<Integer>>` or `@SuppressWarnings` with `new ArrayList[n]` |
| `HashMap` not ordered | Use `LinkedHashMap` for insertion order, `TreeMap` for sorted |
| `HashSet`/`HashMap` requires `equals`+`hashCode` | Override both for custom objects |
| `Deque.push`/`pop` work on **front** | `push` = `addFirst`, `pop` = `removeFirst` |
| Autoboxing overhead | Use primitive arrays (`int[]`) over `ArrayList<Integer>` for performance |
| `toArray()` returns `Object[]` | Use `toArray(new Integer[0])` or `.stream().mapToInt().toArray()` |

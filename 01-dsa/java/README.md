# DSA in Java — Complete Interview Refresher

A comprehensive Java refresher for coding interviews, covering Collections framework, utilities, idioms, and common patterns.

---

## Table of Contents
- [Primitive Types & Wrappers](#primitive-types--wrappers)
- [Collections Deep Dive](#collections-deep-dive)
- [Strings](#strings)
- [Arrays Utility](#arrays-utility)
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

// Limits
Integer.MIN_VALUE  // -2147483648
Integer.MAX_VALUE  //  2147483647
Long.MIN_VALUE, Long.MAX_VALUE

// Conversions
Integer.parseInt("42")         // String → int
String.valueOf(42)             // int → String
Integer.toString(42)           // int → String
(char)('a' + 3)                // → 'd'
Character.isLetterOrDigit(c)   // char classification
```

---

## Collections Deep Dive

### `ArrayList<T>` — Dynamic Array

```java
import java.util.*;

List<Integer> list = new ArrayList<>();
list.add(x);                   // O(1) amortized
list.add(index, x);            // O(n) insert at index
list.get(i);                   // O(1)
list.set(i, x);                // O(1)
list.remove(index);            // O(n), removes by index
list.remove(Integer.valueOf(x)); // O(n), removes first occurrence of value
list.size();
list.isEmpty();
list.contains(x);              // O(n)
list.indexOf(x);               // O(n)
list.subList(from, to);        // view (not copy)
list.toArray(new int[0]);      // convert to array

// Sorting
Collections.sort(list);
list.sort(Comparator.naturalOrder());
list.sort(Comparator.reverseOrder());
list.sort((a, b) -> a[0] - b[0]);  // custom (careful with overflow!)
list.sort(Comparator.comparingInt(a -> a[0])); // safer custom sort
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

// Frequency counting patterns
map.merge(key, 1, Integer::sum);                    // increment
map.put(key, map.getOrDefault(key, 0) + 1);       // alternative
map.compute(key, (k, v) -> v == null ? 1 : v + 1); // compute

// Iteration
for (Map.Entry<String, Integer> e : map.entrySet()) {
    String key = e.getKey();
    int val = e.getValue();
}
for (var e : map.entrySet()) { ... }  // Java 10+ var
map.forEach((k, v) -> { ... });
```

### `HashSet<T>` — Hash Set

```java
Set<Integer> set = new HashSet<>();
set.add(x);             // O(1) avg
set.contains(x);        // O(1) avg
set.remove(x);          // O(1) avg
set.size();
set.isEmpty();

// Set operations
set1.retainAll(set2);    // intersection (modifies set1)
set1.addAll(set2);       // union
set1.removeAll(set2);    // difference
```

### `TreeMap<K,V>` — Sorted Map (Red-Black Tree)

```java
TreeMap<Integer, String> tm = new TreeMap<>();
tm.floorKey(key);        // greatest key <= key (or null)
tm.ceilingKey(key);      // smallest key >= key (or null)
tm.lowerKey(key);        // greatest key < key
tm.higherKey(key);       // smallest key > key
tm.firstKey();           // minimum key
tm.lastKey();            // maximum key
tm.subMap(from, to);     // range view
// All operations O(log n)
```

### `TreeSet<T>` — Sorted Set

```java
TreeSet<Integer> ts = new TreeSet<>();
ts.floor(x);      // greatest <= x
ts.ceiling(x);    // smallest >= x
ts.lower(x);      // greatest < x
ts.higher(x);     // smallest > x
ts.first();        // min
ts.last();         // max
```

### `PriorityQueue<T>` — Heap

```java
// Min-heap (default)
PriorityQueue<Integer> minPQ = new PriorityQueue<>();
minPQ.offer(x);      // O(log n)
minPQ.peek();         // O(1) — smallest
minPQ.poll();         // O(log n) — remove smallest
minPQ.size();

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
```

### `ArrayDeque<T>` — Stack & Queue (Preferred)

```java
// As Stack (LIFO)
Deque<Integer> stack = new ArrayDeque<>();
stack.push(x);        // push to front
stack.pop();           // pop from front
stack.peek();          // view front
stack.isEmpty();

// As Queue (FIFO)
Deque<Integer> queue = new ArrayDeque<>();
queue.offer(x);        // add to back
queue.poll();           // remove from front
queue.peek();           // view front

// As Deque
Deque<Integer> dq = new ArrayDeque<>();
dq.offerFirst(x); dq.offerLast(x);
dq.pollFirst();    dq.pollLast();
dq.peekFirst();    dq.peekLast();
```

> **Note:** Prefer `ArrayDeque` over `Stack` (legacy, synchronized) and `LinkedList` (more GC overhead).

### `LinkedList<T>` — Doubly-Linked List

```java
LinkedList<Integer> ll = new LinkedList<>();
ll.addFirst(x); ll.addLast(x);
ll.removeFirst(); ll.removeLast();
ll.getFirst(); ll.getLast();
// Also implements Deque and List interfaces
```

---

## Strings

```java
String s = "hello";
s.length();
s.charAt(i);
s.substring(start, end);     // [start, end) — O(n)
s.indexOf("ll");             // first occurrence, -1 if not found
s.lastIndexOf("l");
s.contains("ell");
s.equals(other);             // always use .equals(), never ==
s.compareTo(other);          // lexicographic
s.toCharArray();             // String → char[]
new String(charArray);       // char[] → String
s.split(",");                // split by delimiter
String.join(",", list);      // join list with delimiter
s.trim();                    // remove leading/trailing whitespace
s.toLowerCase(); s.toUpperCase();
s.replace('l', 'r');         // replace all occurrences
s.startsWith("he"); s.endsWith("lo");

// StringBuilder (mutable — use in loops!)
StringBuilder sb = new StringBuilder();
sb.append("hello");
sb.append(' ');
sb.append(42);
sb.insert(0, "prefix");
sb.reverse();
sb.toString();               // convert back to String
sb.deleteCharAt(i);
sb.setCharAt(i, c);
```

---

## Arrays Utility

```java
import java.util.Arrays;

int[] arr = new int[n];           // default 0
int[] arr = {1, 2, 3};
int[][] grid = new int[m][n];

Arrays.sort(arr);                 // O(n log n)
Arrays.sort(arr, from, to);      // sort subarray
Arrays.fill(arr, val);           // fill with value
Arrays.copyOf(arr, newLen);      // copy with new length
Arrays.equals(a, b);             // content equality
Arrays.binarySearch(arr, target); // sorted array, returns index
Arrays.toString(arr);            // "[1, 2, 3]" for debugging
Arrays.stream(arr).sum();        // sum via stream
Arrays.stream(arr).max().getAsInt(); // max via stream

// 2D array sort
Arrays.sort(intervals, Comparator.comparingInt(a -> a[0]));
```

---

## Common Interview Idioms

### Integer / Math

```java
Math.max(a, b); Math.min(a, b);
Math.abs(x);
Math.pow(base, exp);          // returns double
Math.sqrt(x);
Math.ceil(x); Math.floor(x);
(int) Math.ceil((double) a / b);  // ceiling division
// Or: (a + b - 1) / b             // integer ceiling division

Integer.bitCount(n);          // popcount
Integer.highestOneBit(n);
Integer.numberOfLeadingZeros(n);
```

### Collections Utilities

```java
Collections.sort(list);
Collections.reverse(list);
Collections.swap(list, i, j);
Collections.frequency(list, obj);
Collections.unmodifiableList(list);  // read-only view
List.of(1, 2, 3);                   // immutable list (Java 9+)
Map.of("a", 1, "b", 2);             // immutable map (Java 9+)
```

### Stream Basics

```java
List<Integer> result = list.stream()
    .filter(x -> x > 0)
    .map(x -> x * 2)
    .sorted()
    .collect(Collectors.toList());

int sum = list.stream().mapToInt(Integer::intValue).sum();
Map<String, List<String>> groups = list.stream()
    .collect(Collectors.groupingBy(s -> sortString(s)));
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
| `List.remove(int)` vs `List.remove(Object)` | `list.remove(0)` removes by index; `list.remove(Integer.valueOf(0))` removes by value |
| `Arrays.asList()` returns fixed-size list | Wrap in `new ArrayList<>(Arrays.asList(...))` to make mutable |
| Generic arrays not allowed | Use `List<List<Integer>>` or `@SuppressWarnings` with `new ArrayList[n]` |
| `HashMap` not ordered | Use `LinkedHashMap` for insertion order, `TreeMap` for sorted |

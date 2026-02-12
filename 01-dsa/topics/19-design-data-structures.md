# Design Data Structures

Implement custom data structures from scratch — a favorite FAANG interview category that tests OOP, hash maps, linked lists, and algorithmic thinking all at once.

---

## Pattern 1: LRU Cache

**Least Recently Used** — evicts the least recently accessed item when capacity is reached.

**Data structure:** HashMap + Doubly Linked List → O(1) get and put.

**C++**
```cpp
class LRUCache {
    int capacity;
    list<pair<int,int>> cache;  // doubly linked list: (key, value)
    unordered_map<int, list<pair<int,int>>::iterator> map;

public:
    LRUCache(int capacity) : capacity(capacity) {}

    int get(int key) {
        auto it = map.find(key);
        if (it == map.end()) return -1;
        cache.splice(cache.begin(), cache, it->second);  // move to front
        return it->second->second;
    }

    void put(int key, int value) {
        auto it = map.find(key);
        if (it != map.end()) {
            it->second->second = value;
            cache.splice(cache.begin(), cache, it->second);
            return;
        }
        if ((int)cache.size() == capacity) {
            map.erase(cache.back().first);
            cache.pop_back();
        }
        cache.emplace_front(key, value);
        map[key] = cache.begin();
    }
};
```

**Java**
```java
class LRUCache extends LinkedHashMap<Integer, Integer> {
    private final int capacity;

    public LRUCache(int capacity) {
        super(capacity, 0.75f, true);  // accessOrder = true
        this.capacity = capacity;
    }

    public int get(int key) {
        return super.getOrDefault(key, -1);
    }

    public void put(int key, int value) {
        super.put(key, value);
    }

    @Override
    protected boolean removeEldestEntry(Map.Entry<Integer, Integer> eldest) {
        return size() > capacity;
    }
}

// Manual implementation with HashMap + Doubly Linked List
class LRUCacheManual {
    class Node {
        int key, val;
        Node prev, next;
        Node(int k, int v) { key = k; val = v; }
    }

    int capacity;
    Map<Integer, Node> map = new HashMap<>();
    Node head = new Node(0, 0), tail = new Node(0, 0);

    LRUCacheManual(int capacity) {
        this.capacity = capacity;
        head.next = tail;
        tail.prev = head;
    }

    int get(int key) {
        if (!map.containsKey(key)) return -1;
        Node node = map.get(key);
        remove(node);
        insertFront(node);
        return node.val;
    }

    void put(int key, int val) {
        if (map.containsKey(key)) remove(map.get(key));
        if (map.size() == capacity) {
            remove(tail.prev);
        }
        Node node = new Node(key, val);
        insertFront(node);
    }

    private void remove(Node node) {
        map.remove(node.key);
        node.prev.next = node.next;
        node.next.prev = node.prev;
    }

    private void insertFront(Node node) {
        map.put(node.key, node);
        node.next = head.next;
        node.prev = head;
        head.next.prev = node;
        head.next = node;
    }
}
```

**Python**
```python
from collections import OrderedDict

class LRUCache:
    def __init__(self, capacity: int):
        self.cache = OrderedDict()
        self.capacity = capacity

    def get(self, key: int) -> int:
        if key not in self.cache:
            return -1
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)  # remove oldest
```

---

## Pattern 2: LFU Cache

**Least Frequently Used** — evicts the least frequently accessed item (ties broken by LRU).

**Python**
```python
from collections import defaultdict, OrderedDict

class LFUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.key_to_val = {}
        self.key_to_freq = {}
        self.freq_to_keys = defaultdict(OrderedDict)
        self.min_freq = 0

    def _update(self, key):
        freq = self.key_to_freq[key]
        self.freq_to_keys[freq].pop(key)
        if not self.freq_to_keys[freq]:
            del self.freq_to_keys[freq]
            if self.min_freq == freq:
                self.min_freq += 1
        self.key_to_freq[key] = freq + 1
        self.freq_to_keys[freq + 1][key] = None

    def get(self, key: int) -> int:
        if key not in self.key_to_val:
            return -1
        self._update(key)
        return self.key_to_val[key]

    def put(self, key: int, value: int) -> None:
        if self.capacity <= 0:
            return
        if key in self.key_to_val:
            self.key_to_val[key] = value
            self._update(key)
            return
        if len(self.key_to_val) >= self.capacity:
            # Evict LFU (LRU among ties)
            evict_key, _ = self.freq_to_keys[self.min_freq].popitem(last=False)
            if not self.freq_to_keys[self.min_freq]:
                del self.freq_to_keys[self.min_freq]
            del self.key_to_val[evict_key]
            del self.key_to_freq[evict_key]
        self.key_to_val[key] = value
        self.key_to_freq[key] = 1
        self.freq_to_keys[1][key] = None
        self.min_freq = 1
```

---

## Pattern 3: Min Stack

**Push, pop, top, and getMin all in O(1).**

**C++**
```cpp
class MinStack {
    stack<pair<int,int>> stk;  // (value, current_min)
public:
    void push(int val) {
        int m = stk.empty() ? val : min(val, stk.top().second);
        stk.push({val, m});
    }
    void pop() { stk.pop(); }
    int top() { return stk.top().first; }
    int getMin() { return stk.top().second; }
};
```

**Java**
```java
class MinStack {
    Deque<int[]> stack = new ArrayDeque<>();

    public void push(int val) {
        int min = stack.isEmpty() ? val : Math.min(val, stack.peek()[1]);
        stack.push(new int[]{val, min});
    }
    public void pop() { stack.pop(); }
    public int top() { return stack.peek()[0]; }
    public int getMin() { return stack.peek()[1]; }
}
```

**Python**
```python
class MinStack:
    def __init__(self):
        self.stack = []   # [(value, current_min)]

    def push(self, val: int) -> None:
        m = min(val, self.stack[-1][1]) if self.stack else val
        self.stack.append((val, m))

    def pop(self) -> None:
        self.stack.pop()

    def top(self) -> int:
        return self.stack[-1][0]

    def getMin(self) -> int:
        return self.stack[-1][1]
```

---

## Pattern 4: Insert Delete GetRandom O(1)

**Array + HashMap** — array for random access, hashmap for O(1) lookup.

**C++**
```cpp
class RandomizedSet {
    vector<int> nums;
    unordered_map<int, int> idx;  // value -> index in nums

public:
    bool insert(int val) {
        if (idx.count(val)) return false;
        idx[val] = nums.size();
        nums.push_back(val);
        return true;
    }

    bool remove(int val) {
        if (!idx.count(val)) return false;
        int last = nums.back();
        int i = idx[val];
        nums[i] = last;
        idx[last] = i;
        nums.pop_back();
        idx.erase(val);
        return true;
    }

    int getRandom() {
        return nums[rand() % nums.size()];
    }
};
```

**Java**
```java
class RandomizedSet {
    List<Integer> nums = new ArrayList<>();
    Map<Integer, Integer> idx = new HashMap<>();
    Random rand = new Random();

    public boolean insert(int val) {
        if (idx.containsKey(val)) return false;
        idx.put(val, nums.size());
        nums.add(val);
        return true;
    }

    public boolean remove(int val) {
        if (!idx.containsKey(val)) return false;
        int i = idx.get(val);
        int last = nums.get(nums.size() - 1);
        nums.set(i, last);
        idx.put(last, i);
        nums.remove(nums.size() - 1);
        idx.remove(val);
        return true;
    }

    public int getRandom() {
        return nums.get(rand.nextInt(nums.size()));
    }
}
```

**Python**
```python
import random

class RandomizedSet:
    def __init__(self):
        self.nums = []
        self.idx = {}

    def insert(self, val: int) -> bool:
        if val in self.idx:
            return False
        self.idx[val] = len(self.nums)
        self.nums.append(val)
        return True

    def remove(self, val: int) -> bool:
        if val not in self.idx:
            return False
        i = self.idx[val]
        last = self.nums[-1]
        self.nums[i] = last
        self.idx[last] = i
        self.nums.pop()
        del self.idx[val]
        return True

    def getRandom(self) -> int:
        return random.choice(self.nums)
```

---

## Pattern 5: Design HashMap / HashSet

### HashMap from Scratch

**C++**
```cpp
class MyHashMap {
    static const int SIZE = 1000;
    vector<list<pair<int,int>>> buckets;
public:
    MyHashMap() : buckets(SIZE) {}
    void put(int key, int value) {
        auto& bucket = buckets[key % SIZE];
        for (auto& [k, v] : bucket) {
            if (k == key) { v = value; return; }
        }
        bucket.emplace_back(key, value);
    }
    int get(int key) {
        auto& bucket = buckets[key % SIZE];
        for (auto& [k, v] : bucket)
            if (k == key) return v;
        return -1;
    }
    void remove(int key) {
        auto& bucket = buckets[key % SIZE];
        bucket.remove_if([key](auto& p) { return p.first == key; });
    }
};
```

**Java**
```java
class MyHashMap {
    private static final int SIZE = 1000;
    private List<int[]>[] buckets;

    public MyHashMap() {
        buckets = new LinkedList[SIZE];
        for (int i = 0; i < SIZE; i++) buckets[i] = new LinkedList<>();
    }
    public void put(int key, int value) {
        var bucket = buckets[key % SIZE];
        for (int[] pair : bucket) {
            if (pair[0] == key) { pair[1] = value; return; }
        }
        bucket.add(new int[]{key, value});
    }
    public int get(int key) {
        for (int[] pair : buckets[key % SIZE])
            if (pair[0] == key) return pair[1];
        return -1;
    }
    public void remove(int key) {
        buckets[key % SIZE].removeIf(p -> p[0] == key);
    }
}
```

**Python**
```python
class MyHashMap:
    def __init__(self):
        self.size = 1000
        self.buckets = [[] for _ in range(self.size)]

    def _hash(self, key):
        return key % self.size

    def put(self, key: int, value: int) -> None:
        bucket = self.buckets[self._hash(key)]
        for i, (k, v) in enumerate(bucket):
            if k == key:
                bucket[i] = (key, value)
                return
        bucket.append((key, value))

    def get(self, key: int) -> int:
        bucket = self.buckets[self._hash(key)]
        for k, v in bucket:
            if k == key:
                return v
        return -1

    def remove(self, key: int) -> None:
        bucket = self.buckets[self._hash(key)]
        for i, (k, v) in enumerate(bucket):
            if k == key:
                bucket.pop(i)
                return
```

---

## Pattern 6: Iterator Design

### Flatten Nested List Iterator

**C++**
```cpp
class NestedIterator {
    stack<NestedInteger> stk;
public:
    NestedIterator(vector<NestedInteger>& nestedList) {
        for (int i = nestedList.size() - 1; i >= 0; i--)
            stk.push(nestedList[i]);
    }
    int next() {
        int val = stk.top().getInteger(); stk.pop();
        return val;
    }
    bool hasNext() {
        while (!stk.empty() && !stk.top().isInteger()) {
            auto list = stk.top().getList(); stk.pop();
            for (int i = list.size() - 1; i >= 0; i--)
                stk.push(list[i]);
        }
        return !stk.empty();
    }
};
```

**Java**
```java
public class NestedIterator implements Iterator<Integer> {
    Deque<NestedInteger> stack = new ArrayDeque<>();
    public NestedIterator(List<NestedInteger> nestedList) {
        for (int i = nestedList.size() - 1; i >= 0; i--)
            stack.push(nestedList.get(i));
    }
    public Integer next() { return stack.pop().getInteger(); }
    public boolean hasNext() {
        while (!stack.isEmpty() && !stack.peek().isInteger()) {
            List<NestedInteger> list = stack.pop().getList();
            for (int i = list.size() - 1; i >= 0; i--)
                stack.push(list.get(i));
        }
        return !stack.isEmpty();
    }
}
```

**Python**
```python
class NestedIterator:
    def __init__(self, nestedList):
        self.stack = list(reversed(nestedList))

    def next(self) -> int:
        return self.stack.pop().getInteger()

    def hasNext(self) -> bool:
        while self.stack:
            top = self.stack[-1]
            if top.isInteger():
                return True
            self.stack.pop()
            self.stack.extend(reversed(top.getList()))
        return False
```

### Peeking Iterator

**C++**
```cpp
class PeekingIterator : public Iterator {
    int peekedVal;
    bool hasPeeked = false;
public:
    PeekingIterator(const vector<int>& nums) : Iterator(nums) {}
    int peek() {
        if (!hasPeeked) { peekedVal = Iterator::next(); hasPeeked = true; }
        return peekedVal;
    }
    int next() {
        if (hasPeeked) { hasPeeked = false; return peekedVal; }
        return Iterator::next();
    }
    bool hasNext() const { return hasPeeked || Iterator::hasNext(); }
};
```

**Java**
```java
class PeekingIterator implements Iterator<Integer> {
    Iterator<Integer> iter;
    Integer peeked = null;
    public PeekingIterator(Iterator<Integer> iter) { this.iter = iter; }
    public Integer peek() {
        if (peeked == null) peeked = iter.next();
        return peeked;
    }
    public Integer next() {
        if (peeked != null) { Integer val = peeked; peeked = null; return val; }
        return iter.next();
    }
    public boolean hasNext() { return peeked != null || iter.hasNext(); }
}
```

**Python**
```python
class PeekingIterator:
    def __init__(self, iterator):
        self.iterator = iterator
        self.peeked = None
        self.has_peeked = False

    def peek(self):
        if not self.has_peeked:
            self.peeked = self.iterator.next()
            self.has_peeked = True
        return self.peeked

    def next(self):
        if self.has_peeked:
            self.has_peeked = False
            return self.peeked
        return self.iterator.next()

    def hasNext(self):
        return self.has_peeked or self.iterator.hasNext()
```

### BSTIterator (Inorder)

**C++**
```cpp
class BSTIterator {
    stack<TreeNode*> stk;
    void pushLeft(TreeNode* node) {
        while (node) { stk.push(node); node = node->left; }
    }
public:
    BSTIterator(TreeNode* root) { pushLeft(root); }
    int next() {
        auto node = stk.top(); stk.pop();
        pushLeft(node->right);
        return node->val;
    }
    bool hasNext() { return !stk.empty(); }
};
```

**Java**
```java
class BSTIterator {
    Deque<TreeNode> stack = new ArrayDeque<>();
    public BSTIterator(TreeNode root) { pushLeft(root); }
    private void pushLeft(TreeNode node) {
        while (node != null) { stack.push(node); node = node.left; }
    }
    public int next() {
        TreeNode node = stack.pop();
        pushLeft(node.right);
        return node.val;
    }
    public boolean hasNext() { return !stack.isEmpty(); }
}
```

**Python**
```python
class BSTIterator:
    def __init__(self, root):
        self.stack = []
        self._push_left(root)

    def _push_left(self, node):
        while node:
            self.stack.append(node)
            node = node.left

    def next(self) -> int:
        node = self.stack.pop()
        self._push_left(node.right)
        return node.val

    def hasNext(self) -> bool:
        return len(self.stack) > 0
```

---

## Pattern 7: Median Finder (Two Heaps)

**C++**
```cpp
class MedianFinder {
    priority_queue<int> lo;                              // max-heap
    priority_queue<int, vector<int>, greater<>> hi;      // min-heap
public:
    void addNum(int num) {
        lo.push(num);
        hi.push(lo.top()); lo.pop();
        if (hi.size() > lo.size()) { lo.push(hi.top()); hi.pop(); }
    }
    double findMedian() {
        return lo.size() > hi.size() ? lo.top() : (lo.top() + hi.top()) / 2.0;
    }
};
```

**Java**
```java
class MedianFinder {
    PriorityQueue<Integer> lo = new PriorityQueue<>(Collections.reverseOrder()); // max-heap
    PriorityQueue<Integer> hi = new PriorityQueue<>(); // min-heap

    public void addNum(int num) {
        lo.offer(num);
        hi.offer(lo.poll());
        if (hi.size() > lo.size()) lo.offer(hi.poll());
    }

    public double findMedian() {
        return lo.size() > hi.size() ? lo.peek() : (lo.peek() + hi.peek()) / 2.0;
    }
}
```

**Python**
```python
import heapq

class MedianFinder:
    def __init__(self):
        self.lo = []  # max-heap (negated) — stores smaller half
        self.hi = []  # min-heap — stores larger half

    def addNum(self, num: int) -> None:
        heapq.heappush(self.lo, -num)
        heapq.heappush(self.hi, -heapq.heappop(self.lo))
        if len(self.hi) > len(self.lo):
            heapq.heappush(self.lo, -heapq.heappop(self.hi))

    def findMedian(self) -> float:
        if len(self.lo) > len(self.hi):
            return -self.lo[0]
        return (-self.lo[0] + self.hi[0]) / 2
```
```

---

## Pattern 8: Trie (Prefix Tree)

**C++**
```cpp
class Trie {
    struct Node {
        Node* children[26] = {};
        bool isEnd = false;
    };
    Node* root = new Node();
public:
    void insert(string word) {
        auto node = root;
        for (char c : word) {
            if (!node->children[c - 'a']) node->children[c - 'a'] = new Node();
            node = node->children[c - 'a'];
        }
        node->isEnd = true;
    }
    bool search(string word) {
        auto node = find(word);
        return node && node->isEnd;
    }
    bool startsWith(string prefix) { return find(prefix) != nullptr; }
private:
    Node* find(string& s) {
        auto node = root;
        for (char c : s) {
            if (!node->children[c - 'a']) return nullptr;
            node = node->children[c - 'a'];
        }
        return node;
    }
};
```

**Java**
```java
class Trie {
    private Trie[] children = new Trie[26];
    private boolean isEnd = false;

    public void insert(String word) {
        Trie node = this;
        for (char c : word.toCharArray()) {
            if (node.children[c - 'a'] == null) node.children[c - 'a'] = new Trie();
            node = node.children[c - 'a'];
        }
        node.isEnd = true;
    }
    public boolean search(String word) {
        Trie node = find(word);
        return node != null && node.isEnd;
    }
    public boolean startsWith(String prefix) { return find(prefix) != null; }
    private Trie find(String s) {
        Trie node = this;
        for (char c : s.toCharArray()) {
            if (node.children[c - 'a'] == null) return null;
            node = node.children[c - 'a'];
        }
        return node;
    }
}
```

**Python**
```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word: str) -> None:
        node = self.root
        for c in word:
            if c not in node.children:
                node.children[c] = TrieNode()
            node = node.children[c]
        node.is_end = True

    def search(self, word: str) -> bool:
        node = self._find(word)
        return node is not None and node.is_end

    def startsWith(self, prefix: str) -> bool:
        return self._find(prefix) is not None

    def _find(self, prefix):
        node = self.root
        for c in prefix:
            if c not in node.children:
                return None
            node = node.children[c]
        return node
```

---

## Pitfalls & Edge Cases

| Pitfall | How to Handle |
|---------|---------------|
| LRU eviction order | Most recently used at front, least at back |
| LFU tie-breaking | Break ties by LRU order within same frequency |
| RandomizedSet remove | Swap with last element to maintain O(1) |
| HashMap collisions | Use chaining (linked list per bucket) |
| Iterator invalidation | Don't modify collection during iteration |
| Empty cache operations | Handle `get` on empty cache gracefully |
| Capacity 0 | Edge case: some problems allow capacity = 0 |

---

## Practice Problems

| # | Problem | Difficulty | Pattern | Link |
|---|---------|-----------|---------|------|
| 1 | LRU Cache | Medium | HashMap + DLL | [LC 146](https://leetcode.com/problems/lru-cache/) |
| 2 | LFU Cache | Hard | Multi-Map | [LC 460](https://leetcode.com/problems/lfu-cache/) |
| 3 | Min Stack | Medium | Aux Stack | [LC 155](https://leetcode.com/problems/min-stack/) |
| 4 | Insert Delete GetRandom | Medium | Array + HashMap | [LC 380](https://leetcode.com/problems/insert-delete-getrandom-o1/) |
| 5 | Design HashMap | Easy | Hashing | [LC 706](https://leetcode.com/problems/design-hashmap/) |
| 6 | Design HashSet | Easy | Hashing | [LC 705](https://leetcode.com/problems/design-hashset/) |
| 7 | Implement Trie | Medium | Trie | [LC 208](https://leetcode.com/problems/implement-trie-prefix-tree/) |
| 8 | Flatten Nested List Iterator | Medium | Stack | [LC 341](https://leetcode.com/problems/flatten-nested-list-iterator/) |
| 9 | Peeking Iterator | Medium | Wrapper | [LC 284](https://leetcode.com/problems/peeking-iterator/) |
| 10 | BST Iterator | Medium | Stack | [LC 173](https://leetcode.com/problems/binary-search-tree-iterator/) |
| 11 | Find Median from Data Stream | Hard | Two Heaps | [LC 295](https://leetcode.com/problems/find-median-from-data-stream/) |
| 12 | Design Twitter | Medium | HashMap + Heap | [LC 355](https://leetcode.com/problems/design-twitter/) |
| 13 | Time Based Key-Value Store | Medium | HashMap + Binary Search | [LC 981](https://leetcode.com/problems/time-based-key-value-store/) |

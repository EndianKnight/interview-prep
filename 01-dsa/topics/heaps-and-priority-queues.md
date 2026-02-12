# Heaps & Priority Queues

Essential for Top-K problems, merge-K, stream processing, and scheduling.

---

## Core Concepts

- **Min-Heap:** smallest element at root; parent ≤ children
- **Max-Heap:** largest element at root; parent ≥ children
- Implemented as a complete binary tree stored in an array
- `parent(i) = (i-1)/2`, `left(i) = 2i+1`, `right(i) = 2i+2`

| Operation | Time |
|-----------|------|
| Insert | O(log n) |
| Extract min/max | O(log n) |
| Peek | O(1) |
| Build heap | O(n) |
| Heapify (sift down) | O(log n) |

### Language Implementations

| | C++ | Java | Python |
|-|-----|------|--------|
| **Default** | `priority_queue<int>` (max-heap) | `PriorityQueue<Integer>` (min-heap) | `heapq` (min-heap) |
| **Min-heap** | `priority_queue<int, vector<int>, greater<>>` | default | default |
| **Max-heap** | default | `new PriorityQueue<>(Collections.reverseOrder())` | negate values |
| **Custom** | lambda comparator | `Comparator` | tuple ordering or `__lt__` |

---

## Pattern 1: Top-K Elements

**When to use:** K largest, K smallest, K most frequent, K closest.

### Technique
Use a heap of size K: min-heap for "K largest" (pop smallest when size > K), max-heap for "K smallest."

### Example: Kth Largest Element

**C++**
```cpp
int findKthLargest(vector<int>& nums, int k) {
    // Min-heap of size k
    priority_queue<int, vector<int>, greater<int>> pq;
    for (int num : nums) {
        pq.push(num);
        if (pq.size() > k) pq.pop();
    }
    return pq.top();
}
```

**Java**
```java
public int findKthLargest(int[] nums, int k) {
    PriorityQueue<Integer> pq = new PriorityQueue<>(); // min-heap
    for (int num : nums) {
        pq.offer(num);
        if (pq.size() > k) pq.poll();
    }
    return pq.peek();
}
```

**Python**
```python
import heapq

def find_kth_largest(nums: list[int], k: int) -> int:
    # nlargest internally uses a min-heap of size k
    return heapq.nlargest(k, nums)[-1]

# Manual approach
def find_kth_largest_manual(nums: list[int], k: int) -> int:
    min_heap = []
    for num in nums:
        heapq.heappush(min_heap, num)
        if len(min_heap) > k:
            heapq.heappop(min_heap)
    return min_heap[0]
```

---

## Pattern 2: Merge K Sorted

**When to use:** Merge K sorted lists/arrays, smallest range covering elements.

### Example: Merge K Sorted Lists

**Python**
```python
import heapq

def merge_k_lists(lists: list[ListNode | None]) -> ListNode | None:
    heap = []
    for i, node in enumerate(lists):
        if node:
            heapq.heappush(heap, (node.val, i, node))

    dummy = tail = ListNode(0)
    while heap:
        val, i, node = heapq.heappop(heap)
        tail.next = node
        tail = tail.next
        if node.next:
            heapq.heappush(heap, (node.next.val, i, node.next))

    return dummy.next
```

**C++**
```cpp
ListNode* mergeKLists(vector<ListNode*>& lists) {
    auto cmp = [](ListNode* a, ListNode* b) { return a->val > b->val; };
    priority_queue<ListNode*, vector<ListNode*>, decltype(cmp)> pq(cmp);

    for (auto* node : lists)
        if (node) pq.push(node);

    ListNode dummy(0);
    ListNode* tail = &dummy;
    while (!pq.empty()) {
        auto* node = pq.top(); pq.pop();
        tail->next = node;
        tail = tail->next;
        if (node->next) pq.push(node->next);
    }
    return dummy.next;
}
```

**Complexity:** Time O(N log K) where N = total elements, K = number of lists

---

## Pattern 3: Two Heaps (Median Finding)

**When to use:** Running median, or splitting data into two halves by value.

### Example: Find Median from Data Stream

**Python**
```python
import heapq

class MedianFinder:
    def __init__(self):
        self.lo = []  # max-heap (negated) — smaller half
        self.hi = []  # min-heap — larger half

    def add_num(self, num: int) -> None:
        heapq.heappush(self.lo, -num)
        heapq.heappush(self.hi, -heapq.heappop(self.lo))
        if len(self.hi) > len(self.lo):
            heapq.heappush(self.lo, -heapq.heappop(self.hi))

    def find_median(self) -> float:
        if len(self.lo) > len(self.hi):
            return -self.lo[0]
        return (-self.lo[0] + self.hi[0]) / 2
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

---

## Pattern 4: Scheduling / Greedy with Heap

**When to use:** Task scheduling, meeting rooms, CPU intervals.

### Example: Meeting Rooms II (Minimum Rooms)

**Python**
```python
import heapq

def min_meeting_rooms(intervals: list[list[int]]) -> int:
    intervals.sort(key=lambda x: x[0])
    heap = []  # end times of active meetings

    for start, end in intervals:
        if heap and heap[0] <= start:
            heapq.heappop(heap)  # room freed up
        heapq.heappush(heap, end)
    return len(heap)
```

---

## Pitfalls & Edge Cases

| Pitfall | How to Handle |
|---------|--------------|
| C++ `priority_queue` is max-heap | Use `greater<>` for min-heap |
| Java `PriorityQueue` is min-heap | Use `Collections.reverseOrder()` for max-heap |
| Python `heapq` is min-heap only | Negate values for max-heap behavior |
| Custom objects in heap | C++: custom comparator. Java: `Comparator`. Python: tuples or `__lt__` |
| K = 0 or K > n | Validate inputs |
| Equal elements | Define tiebreaker in comparator |

---

## Practice Problems

| # | Problem | Difficulty | Pattern | Link |
|---|---------|-----------|---------|------|
| 1 | Kth Largest Element | Medium | Top-K | [LeetCode 215](https://leetcode.com/problems/kth-largest-element-in-an-array/) |
| 2 | Top K Frequent Elements | Medium | Top-K | [LeetCode 347](https://leetcode.com/problems/top-k-frequent-elements/) |
| 3 | K Closest Points to Origin | Medium | Top-K | [LeetCode 973](https://leetcode.com/problems/k-closest-points-to-origin/) |
| 4 | Merge K Sorted Lists | Hard | Merge-K | [LeetCode 23](https://leetcode.com/problems/merge-k-sorted-lists/) |
| 5 | Find Median from Data Stream | Hard | Two Heaps | [LeetCode 295](https://leetcode.com/problems/find-median-from-data-stream/) |
| 6 | Meeting Rooms II | Medium | Scheduling | [LeetCode 253](https://leetcode.com/problems/meeting-rooms-ii/) |
| 7 | Task Scheduler | Medium | Greedy + Heap | [LeetCode 621](https://leetcode.com/problems/task-scheduler/) |

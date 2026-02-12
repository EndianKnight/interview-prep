# Queues & Deques

FIFO (First In, First Out) data structures — essential for BFS, task scheduling, sliding window maximums, and stream processing.

---

## Core Concepts

### Queue (FIFO)
- **Enqueue** — add to back, O(1)
- **Dequeue** — remove from front, O(1)
- **Peek/Front** — view front element, O(1)
- Use cases: BFS, task scheduling, rate limiting, buffering, producer-consumer

### Deque (Double-Ended Queue)
- Add/remove from **both** ends in O(1)
- Superset of stack AND queue
- Powers the sliding window maximum pattern

### Language Implementations

| | C++ | Java | Python |
|-|-----|------|--------|
| **Queue** | `queue<T>` | `ArrayDeque<T>` | `collections.deque` |
| **Deque** | `deque<T>` | `ArrayDeque<T>` | `collections.deque` |
| Enqueue | `q.push(x)` | `q.offer(x)` | `q.append(x)` |
| Dequeue | `q.pop()` (returns void) | `q.poll()` | `q.popleft()` |
| Front | `q.front()` | `q.peek()` | `q[0]` |
| Back | `q.back()` | N/A | `q[-1]` |

> ⚠️ **Python:** Never use `list.pop(0)` for a queue — it's O(n). Always use `collections.deque.popleft()` which is O(1).

---

## Pattern 1: BFS (Breadth-First Search)

The most common use of queues. Process nodes level by level — guarantees shortest path in unweighted graphs.

### Example 1a: Binary Tree Level-Order Traversal

**C++**
```cpp
#include <queue>
#include <vector>
using namespace std;

vector<vector<int>> levelOrder(TreeNode* root) {
    vector<vector<int>> result;
    if (!root) return result;

    queue<TreeNode*> q;
    q.push(root);

    while (!q.empty()) {
        int size = q.size();
        vector<int> level;
        for (int i = 0; i < size; i++) {
            TreeNode* node = q.front(); q.pop();
            level.push_back(node->val);
            if (node->left) q.push(node->left);
            if (node->right) q.push(node->right);
        }
        result.push_back(level);
    }
    return result;
}
```

**Java**
```java
public List<List<Integer>> levelOrder(TreeNode root) {
    List<List<Integer>> result = new ArrayList<>();
    if (root == null) return result;

    Queue<TreeNode> queue = new ArrayDeque<>();
    queue.offer(root);

    while (!queue.isEmpty()) {
        int size = queue.size();
        List<Integer> level = new ArrayList<>();
        for (int i = 0; i < size; i++) {
            TreeNode node = queue.poll();
            level.add(node.val);
            if (node.left != null) queue.offer(node.left);
            if (node.right != null) queue.offer(node.right);
        }
        result.add(level);
    }
    return result;
}
```

**Python**
```python
from collections import deque

def level_order(root) -> list[list[int]]:
    if not root:
        return []
    result, queue = [], deque([root])
    while queue:
        level = []
        for _ in range(len(queue)):
            node = queue.popleft()
            level.append(node.val)
            if node.left:  queue.append(node.left)
            if node.right: queue.append(node.right)
        result.append(level)
    return result
```

**Complexity:** Time O(n), Space O(n)

### Example 1b: Number of Islands (BFS variant)

Count the number of islands in a binary grid.

**Python**
```python
from collections import deque

def num_islands(grid: list[list[str]]) -> int:
    if not grid:
        return 0
    rows, cols = len(grid), len(grid[0])
    count = 0

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == '1':
                count += 1
                # BFS to mark entire island as visited
                queue = deque([(r, c)])
                grid[r][c] = '0'
                while queue:
                    row, col = queue.popleft()
                    for dr, dc in [(0,1),(0,-1),(1,0),(-1,0)]:
                        nr, nc = row + dr, col + dc
                        if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == '1':
                            grid[nr][nc] = '0'
                            queue.append((nr, nc))
    return count
```

**C++**
```cpp
int numIslands(vector<vector<char>>& grid) {
    int rows = grid.size(), cols = grid[0].size(), count = 0;
    int dirs[] = {0,1,0,-1,0};

    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            if (grid[r][c] == '1') {
                count++;
                queue<pair<int,int>> q;
                q.push({r, c});
                grid[r][c] = '0';
                while (!q.empty()) {
                    auto [row, col] = q.front(); q.pop();
                    for (int d = 0; d < 4; d++) {
                        int nr = row + dirs[d], nc = col + dirs[d+1];
                        if (nr >= 0 && nr < rows && nc >= 0 && nc < cols
                            && grid[nr][nc] == '1') {
                            grid[nr][nc] = '0';
                            q.push({nr, nc});
                        }
                    }
                }
            }
        }
    }
    return count;
}
```

**Java**
```java
public int numIslands(char[][] grid) {
    int rows = grid.length, cols = grid[0].length, count = 0;
    int[][] dirs = {{0,1},{0,-1},{1,0},{-1,0}};

    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            if (grid[r][c] == '1') {
                count++;
                Queue<int[]> q = new ArrayDeque<>();
                q.offer(new int[]{r, c});
                grid[r][c] = '0';
                while (!q.isEmpty()) {
                    int[] pos = q.poll();
                    for (int[] d : dirs) {
                        int nr = pos[0] + d[0], nc = pos[1] + d[1];
                        if (nr >= 0 && nr < rows && nc >= 0 && nc < cols
                            && grid[nr][nc] == '1') {
                            grid[nr][nc] = '0';
                            q.offer(new int[]{nr, nc});
                        }
                    }
                }
            }
        }
    }
    return count;
}
```

### Example 1c: Shortest Path in Binary Matrix

Find the shortest clear path from top-left to bottom-right in a binary grid (8-directional).

**Python**
```python
from collections import deque

def shortest_path_binary_matrix(grid: list[list[int]]) -> int:
    n = len(grid)
    if grid[0][0] == 1 or grid[n-1][n-1] == 1:
        return -1

    queue = deque([(0, 0, 1)])  # (row, col, distance)
    grid[0][0] = 1  # mark visited

    directions = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]

    while queue:
        r, c, dist = queue.popleft()
        if r == n - 1 and c == n - 1:
            return dist
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < n and 0 <= nc < n and grid[nr][nc] == 0:
                grid[nr][nc] = 1
                queue.append((nr, nc, dist + 1))
    return -1
```

### Example 1d: Rotting Oranges (Multi-Source BFS)

All rotten oranges spread simultaneously. How many minutes until all oranges are rotten?

**Python**
```python
from collections import deque

def oranges_rotting(grid: list[list[int]]) -> int:
    rows, cols = len(grid), len(grid[0])
    queue = deque()
    fresh = 0

    # Find all initial rotten oranges and count fresh
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 2:
                queue.append((r, c))
            elif grid[r][c] == 1:
                fresh += 1

    if fresh == 0:
        return 0

    minutes = 0
    while queue and fresh > 0:
        minutes += 1
        for _ in range(len(queue)):
            r, c = queue.popleft()
            for dr, dc in [(0,1),(0,-1),(1,0),(-1,0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == 1:
                    grid[nr][nc] = 2
                    fresh -= 1
                    queue.append((nr, nc))

    return minutes if fresh == 0 else -1
```

**Complexity:** Time O(m×n), Space O(m×n)

---

## Pattern 2: Sliding Window Maximum (Monotonic Deque)

**When to use:** Maximum/minimum in a sliding window of fixed size.

### Core Idea
Maintain a deque of **indices** where corresponding values are in **decreasing order**. The front of the deque is always the maximum of the current window.

### Example 2a: Sliding Window Maximum

Given `nums = [1,3,-1,-3,5,3,6,7]` and `k = 3`, return `[3,3,5,5,6,7]`.

**C++**
```cpp
#include <deque>
#include <vector>
using namespace std;

vector<int> maxSlidingWindow(vector<int>& nums, int k) {
    deque<int> dq; // indices with decreasing values
    vector<int> result;

    for (int i = 0; i < nums.size(); i++) {
        // Remove indices outside window
        while (!dq.empty() && dq.front() <= i - k) {
            dq.pop_front();
        }
        // Remove smaller elements from back
        while (!dq.empty() && nums[dq.back()] <= nums[i]) {
            dq.pop_back();
        }
        dq.push_back(i);

        // Window is full — record max
        if (i >= k - 1) {
            result.push_back(nums[dq.front()]);
        }
    }
    return result;
}
```

**Java**
```java
public int[] maxSlidingWindow(int[] nums, int k) {
    Deque<Integer> dq = new ArrayDeque<>();
    int[] result = new int[nums.length - k + 1];
    int ri = 0;

    for (int i = 0; i < nums.length; i++) {
        while (!dq.isEmpty() && dq.peekFirst() <= i - k)
            dq.pollFirst();
        while (!dq.isEmpty() && nums[dq.peekLast()] <= nums[i])
            dq.pollLast();
        dq.offerLast(i);

        if (i >= k - 1)
            result[ri++] = nums[dq.peekFirst()];
    }
    return result;
}
```

**Python**
```python
from collections import deque

def max_sliding_window(nums: list[int], k: int) -> list[int]:
    dq = deque()  # indices, values decreasing
    result = []

    for i, num in enumerate(nums):
        # Remove indices outside window
        while dq and dq[0] <= i - k:
            dq.popleft()
        # Remove smaller elements from back
        while dq and nums[dq[-1]] <= num:
            dq.pop()
        dq.append(i)

        if i >= k - 1:
            result.append(nums[dq[0]])
    return result
```

**Complexity:** Time O(n), Space O(k)

---

## Pattern 3: Implement One with the Other

Classic interview question — implement a queue using two stacks, or a stack using two queues.

### Example 3a: Queue using Two Stacks

**Python**
```python
class MyQueue:
    def __init__(self):
        self.in_stack = []   # push here
        self.out_stack = []  # pop from here

    def push(self, x: int) -> None:
        self.in_stack.append(x)

    def pop(self) -> int:
        self._transfer()
        return self.out_stack.pop()

    def peek(self) -> int:
        self._transfer()
        return self.out_stack[-1]

    def empty(self) -> bool:
        return not self.in_stack and not self.out_stack

    def _transfer(self):
        """Move in_stack → out_stack only when out_stack is empty"""
        if not self.out_stack:
            while self.in_stack:
                self.out_stack.append(self.in_stack.pop())
```

**C++**
```cpp
class MyQueue {
    stack<int> inStack, outStack;

    void transfer() {
        if (outStack.empty()) {
            while (!inStack.empty()) {
                outStack.push(inStack.top());
                inStack.pop();
            }
        }
    }
public:
    void push(int x) { inStack.push(x); }
    int pop() { transfer(); int val = outStack.top(); outStack.pop(); return val; }
    int peek() { transfer(); return outStack.top(); }
    bool empty() { return inStack.empty() && outStack.empty(); }
};
```

**Java**
```java
class MyQueue {
    private Deque<Integer> inStack = new ArrayDeque<>();
    private Deque<Integer> outStack = new ArrayDeque<>();

    private void transfer() {
        if (outStack.isEmpty())
            while (!inStack.isEmpty())
                outStack.push(inStack.pop());
    }

    public void push(int x) { inStack.push(x); }
    public int pop() { transfer(); return outStack.pop(); }
    public int peek() { transfer(); return outStack.peek(); }
    public boolean empty() { return inStack.isEmpty() && outStack.isEmpty(); }
}
```

**Key insight:** Each element moves between stacks at most once, so amortized O(1) per operation.

### Example 3b: Circular Queue (Ring Buffer)

**Python**
```python
class MyCircularQueue:
    def __init__(self, k: int):
        self.data = [0] * k
        self.head = 0
        self.count = 0
        self.capacity = k

    def enqueue(self, value: int) -> bool:
        if self.is_full():
            return False
        tail = (self.head + self.count) % self.capacity
        self.data[tail] = value
        self.count += 1
        return True

    def dequeue(self) -> bool:
        if self.is_empty():
            return False
        self.head = (self.head + 1) % self.capacity
        self.count -= 1
        return True

    def front(self) -> int:
        return -1 if self.is_empty() else self.data[self.head]

    def rear(self) -> int:
        if self.is_empty():
            return -1
        tail = (self.head + self.count - 1) % self.capacity
        return self.data[tail]

    def is_empty(self) -> bool:
        return self.count == 0

    def is_full(self) -> bool:
        return self.count == self.capacity
```

---

## Pattern 4: Task Scheduling / Rate Limiting

**When to use:** Processing tasks with constraints, cooldowns, or ordering requirements.

### Example 4a: Task Scheduler

Given tasks with a cooldown period, find minimum intervals to finish all tasks.

**Python**
```python
from collections import Counter

def least_interval(tasks: list[str], n: int) -> int:
    freq = Counter(tasks)
    max_count = max(freq.values())
    max_count_tasks = sum(1 for v in freq.values() if v == max_count)
    return max(len(tasks), (max_count - 1) * (n + 1) + max_count_tasks)
```

**Java**
```java
public int leastInterval(char[] tasks, int n) {
    int[] freq = new int[26];
    for (char t : tasks) freq[t - 'A']++;
    int maxCount = 0, maxCountTasks = 0;
    for (int f : freq) {
        if (f > maxCount) { maxCount = f; maxCountTasks = 1; }
        else if (f == maxCount) maxCountTasks++;
    }
    return Math.max(tasks.length, (maxCount - 1) * (n + 1) + maxCountTasks);
}
```

### Example 4b: Design Hit Counter (Sliding Window Queue)

Count hits in the last 300 seconds.

**Python**
```python
from collections import deque

class HitCounter:
    def __init__(self):
        self.queue = deque()

    def hit(self, timestamp: int) -> None:
        self.queue.append(timestamp)

    def get_hits(self, timestamp: int) -> int:
        while self.queue and self.queue[0] <= timestamp - 300:
            self.queue.popleft()
        return len(self.queue)
```

**C++**
```cpp
class HitCounter {
    queue<int> q;
public:
    void hit(int timestamp) { q.push(timestamp); }

    int getHits(int timestamp) {
        while (!q.empty() && q.front() <= timestamp - 300)
            q.pop();
        return q.size();
    }
};
```

---

## Pattern 5: Fixed-Size Queue / Bounded Buffer

**When to use:** Moving average, recent items, bounded caches.

### Example 5a: Moving Average from Data Stream

**Python**
```python
from collections import deque

class MovingAverage:
    def __init__(self, size: int):
        self.queue = deque(maxlen=size)  # auto-evicts oldest
        self.total = 0
        self.size = size

    def next(self, val: int) -> float:
        if len(self.queue) == self.size:
            self.total -= self.queue[0]  # will be evicted
        self.queue.append(val)
        self.total += val
        return self.total / len(self.queue)
```

**Java**
```java
class MovingAverage {
    private Deque<Integer> queue = new ArrayDeque<>();
    private int size;
    private double total = 0;

    public MovingAverage(int size) { this.size = size; }

    public double next(int val) {
        if (queue.size() == size) total -= queue.poll();
        queue.offer(val);
        total += val;
        return total / queue.size();
    }
}
```

---

## Pitfalls & Edge Cases

| Pitfall | How to Handle |
|---------|--------------|
| Python `list.pop(0)` is O(n) | Use `collections.deque.popleft()` — O(1) |
| Java `LinkedList` as queue | Prefer `ArrayDeque` — less GC overhead |
| BFS without visited check | Will infinite-loop on graphs with cycles |
| Multi-source BFS — forgetting to enqueue all sources | Add ALL starting positions before the BFS loop |
| Deque sliding window — wrong eviction | Compare with `i - k`, not `i - k + 1` |
| Circular queue off-by-one | Use `(head + count) % capacity` for tail |

---

## Practice Problems

| # | Problem | Difficulty | Pattern | Link |
|---|---------|-----------|---------|------|
| 1 | Implement Queue using Stacks | Easy | Implementation | [LeetCode 232](https://leetcode.com/problems/implement-queue-using-stacks/) |
| 2 | Number of Islands | Medium | BFS | [LeetCode 200](https://leetcode.com/problems/number-of-islands/) |
| 3 | Binary Tree Level Order | Medium | BFS | [LeetCode 102](https://leetcode.com/problems/binary-tree-level-order-traversal/) |
| 4 | Rotting Oranges | Medium | Multi-Source BFS | [LeetCode 994](https://leetcode.com/problems/rotting-oranges/) |
| 5 | Design Hit Counter | Medium | Sliding Queue | [LeetCode 362](https://leetcode.com/problems/design-hit-counter/) |
| 6 | Task Scheduler | Medium | Scheduling | [LeetCode 621](https://leetcode.com/problems/task-scheduler/) |
| 7 | Design Circular Queue | Medium | Ring Buffer | [LeetCode 622](https://leetcode.com/problems/design-circular-queue/) |
| 8 | Sliding Window Maximum | Hard | Monotonic Deque | [LeetCode 239](https://leetcode.com/problems/sliding-window-maximum/) |
| 9 | Shortest Path in Binary Matrix | Medium | BFS | [LeetCode 1091](https://leetcode.com/problems/shortest-path-in-binary-matrix/) |
| 10 | Walls and Gates | Medium | Multi-Source BFS | [LeetCode 286](https://leetcode.com/problems/walls-and-gates/) |

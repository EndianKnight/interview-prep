    # Stacks & Queues

Fundamental LIFO and FIFO data structures — critical for parsing, BFS, monotonic patterns, and state management.

---

## Core Concepts

### Stack (LIFO — Last In, First Out)
- **Push** — add to top, O(1)
- **Pop** — remove from top, O(1)
- **Peek/Top** — view top element, O(1)
- Use cases: undo, DFS, expression evaluation, matching brackets

### Queue (FIFO — First In, First Out)
- **Enqueue** — add to back, O(1)
- **Dequeue** — remove from front, O(1)
- **Peek/Front** — view front element, O(1)
- Use cases: BFS, task scheduling, rate limiting, buffering

### Deque (Double-Ended Queue)
- Add/remove from **both** ends in O(1)
- Powers the sliding window maximum pattern

### Language Implementations

| Structure | C++ | Java | Python |
|-----------|-----|------|--------|
| Stack | `stack<T>` | `ArrayDeque<T>` (preferred over `Stack`) | `list` (use `append`/`pop`) |
| Queue | `queue<T>` | `ArrayDeque<T>` or `LinkedList<T>` | `collections.deque` |
| Deque | `deque<T>` | `ArrayDeque<T>` | `collections.deque` |

---

## Pattern 1: Matching / Validation

**When to use:** Balanced brackets, HTML tag matching, expression validation.

### Example: Valid Parentheses

**C++**
```cpp
#include <stack>
#include <string>
#include <unordered_map>
using namespace std;

bool isValid(string s) {
    stack<char> st;
    unordered_map<char, char> pairs = {{')', '('}, {']', '['}, {'}', '{'}};

    for (char c : s) {
        if (pairs.count(c)) {
            if (st.empty() || st.top() != pairs[c]) return false;
            st.pop();
        } else {
            st.push(c);
        }
    }
    return st.empty();
}
```

**Java**
```java
import java.util.*;

public boolean isValid(String s) {
    Deque<Character> stack = new ArrayDeque<>();
    Map<Character, Character> pairs = Map.of(')', '(', ']', '[', '}', '{');

    for (char c : s.toCharArray()) {
        if (pairs.containsKey(c)) {
            if (stack.isEmpty() || stack.peek() != pairs.get(c)) return false;
            stack.pop();
        } else {
            stack.push(c);
        }
    }
    return stack.isEmpty();
}
```

**Python**
```python
def is_valid(s: str) -> bool:
    stack = []
    pairs = {')': '(', ']': '[', '}': '{'}

    for c in s:
        if c in pairs:
            if not stack or stack[-1] != pairs[c]:
                return False
            stack.pop()
        else:
            stack.append(c)
    return not stack
```

---

## Pattern 2: Monotonic Stack

**When to use:** Next greater/smaller element, daily temperatures, stock span, histogram problems.

### Technique
Maintain a stack where elements are in monotonic (increasing or decreasing) order. When a new element violates the order, pop and process.

### Example: Next Greater Element

For each element, find the next element to the right that is greater.

**C++**
```cpp
#include <vector>
#include <stack>
using namespace std;

vector<int> nextGreaterElement(vector<int>& nums) {
    int n = nums.size();
    vector<int> result(n, -1);
    stack<int> st; // stores indices

    for (int i = 0; i < n; i++) {
        while (!st.empty() && nums[st.top()] < nums[i]) {
            result[st.top()] = nums[i];
            st.pop();
        }
        st.push(i);
    }
    return result;
}
```

**Java**
```java
public int[] nextGreaterElement(int[] nums) {
    int n = nums.length;
    int[] result = new int[n];
    Arrays.fill(result, -1);
    Deque<Integer> stack = new ArrayDeque<>(); // stores indices

    for (int i = 0; i < n; i++) {
        while (!stack.isEmpty() && nums[stack.peek()] < nums[i]) {
            result[stack.pop()] = nums[i];
        }
        stack.push(i);
    }
    return result;
}
```

**Python**
```python
def next_greater_element(nums: list[int]) -> list[int]:
    n = len(nums)
    result = [-1] * n
    stack = []  # stores indices

    for i in range(n):
        while stack and nums[stack[-1]] < nums[i]:
            result[stack.pop()] = nums[i]
        stack.append(i)
    return result
```

**Complexity:** Time O(n) — each element pushed and popped at most once. Space O(n).

### Example: Daily Temperatures

**Python**
```python
def daily_temperatures(temperatures: list[int]) -> list[int]:
    n = len(temperatures)
    result = [0] * n
    stack = []

    for i in range(n):
        while stack and temperatures[stack[-1]] < temperatures[i]:
            prev = stack.pop()
            result[prev] = i - prev
        stack.append(i)
    return result
```

---

## Pattern 3: Stack-Based Evaluation

**When to use:** Calculator, RPN (reverse Polish notation), infix to postfix conversion.

### Example: Evaluate Reverse Polish Notation

**C++**
```cpp
#include <vector>
#include <stack>
#include <string>
using namespace std;

int evalRPN(vector<string>& tokens) {
    stack<int> st;

    for (const string& t : tokens) {
        if (t == "+" || t == "-" || t == "*" || t == "/") {
            int b = st.top(); st.pop();
            int a = st.top(); st.pop();
            if (t == "+") st.push(a + b);
            else if (t == "-") st.push(a - b);
            else if (t == "*") st.push(a * b);
            else st.push(a / b);
        } else {
            st.push(stoi(t));
        }
    }
    return st.top();
}
```

**Java**
```java
public int evalRPN(String[] tokens) {
    Deque<Integer> stack = new ArrayDeque<>();

    for (String t : tokens) {
        switch (t) {
            case "+", "-", "*", "/" -> {
                int b = stack.pop(), a = stack.pop();
                stack.push(switch (t) {
                    case "+" -> a + b;
                    case "-" -> a - b;
                    case "*" -> a * b;
                    case "/" -> a / b;
                    default -> 0;
                });
            }
            default -> stack.push(Integer.parseInt(t));
        }
    }
    return stack.pop();
}
```

**Python**
```python
def eval_rpn(tokens: list[str]) -> int:
    stack = []
    ops = {
        '+': lambda a, b: a + b,
        '-': lambda a, b: a - b,
        '*': lambda a, b: a * b,
        '/': lambda a, b: int(a / b),  # truncate towards zero
    }

    for t in tokens:
        if t in ops:
            b, a = stack.pop(), stack.pop()
            stack.append(ops[t](a, b))
        else:
            stack.append(int(t))
    return stack[0]
```

---

## Pattern 4: BFS with Queue

**When to use:** Level-order traversal, shortest path in unweighted graph, rotting oranges.

### Example: Binary Tree Level-Order Traversal

**Python**
```python
from collections import deque

def level_order(root) -> list[list[int]]:
    if not root:
        return []
    result = []
    queue = deque([root])

    while queue:
        level = []
        for _ in range(len(queue)):
            node = queue.popleft()
            level.append(node.val)
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        result.append(level)
    return result
```

---

## Pattern 5: Min/Max Stack

**When to use:** Track minimum/maximum in O(1) alongside standard stack operations.

### Example: Min Stack

**C++**
```cpp
class MinStack {
    stack<int> data;
    stack<int> mins;
public:
    void push(int val) {
        data.push(val);
        mins.push(mins.empty() ? val : min(val, mins.top()));
    }
    void pop() { data.pop(); mins.pop(); }
    int top() { return data.top(); }
    int getMin() { return mins.top(); }
};
```

**Java**
```java
class MinStack {
    private Deque<Integer> data = new ArrayDeque<>();
    private Deque<Integer> mins = new ArrayDeque<>();

    public void push(int val) {
        data.push(val);
        mins.push(mins.isEmpty() ? val : Math.min(val, mins.peek()));
    }
    public void pop() { data.pop(); mins.pop(); }
    public int top() { return data.peek(); }
    public int getMin() { return mins.peek(); }
}
```

**Python**
```python
class MinStack:
    def __init__(self):
        self.data = []
        self.mins = []

    def push(self, val: int) -> None:
        self.data.append(val)
        self.mins.append(min(val, self.mins[-1]) if self.mins else val)

    def pop(self) -> None:
        self.data.pop()
        self.mins.pop()

    def top(self) -> int:
        return self.data[-1]

    def get_min(self) -> int:
        return self.mins[-1]
```

---

## Pitfalls & Edge Cases

| Pitfall | How to Handle |
|---------|--------------|
| Popping from empty stack/queue | Always check `empty()` first |
| Java `Stack` class | Use `ArrayDeque` instead — `Stack` is legacy/synchronized |
| Python `list` as queue | Don't use `list.pop(0)` — it's O(n). Use `collections.deque` |
| Monotonic stack direction | Clarify: increasing from bottom→top for "next greater" |
| Integer division in RPN | Python `//` floors; use `int(a/b)` for truncation toward zero |

---

## Practice Problems

| # | Problem | Difficulty | Pattern | Link |
|---|---------|-----------|---------|------|
| 1 | Valid Parentheses | Easy | Matching | [LeetCode 20](https://leetcode.com/problems/valid-parentheses/) |
| 2 | Min Stack | Medium | Min Stack | [LeetCode 155](https://leetcode.com/problems/min-stack/) |
| 3 | Daily Temperatures | Medium | Monotonic Stack | [LeetCode 739](https://leetcode.com/problems/daily-temperatures/) |
| 4 | Evaluate RPN | Medium | Evaluation | [LeetCode 150](https://leetcode.com/problems/evaluate-reverse-polish-notation/) |
| 5 | Implement Queue using Stacks | Easy | Two Stacks | [LeetCode 232](https://leetcode.com/problems/implement-queue-using-stacks/) |
| 6 | Next Greater Element | Easy | Monotonic Stack | [LeetCode 496](https://leetcode.com/problems/next-greater-element-i/) |
| 7 | Sliding Window Maximum | Hard | Monotonic Deque | [LeetCode 239](https://leetcode.com/problems/sliding-window-maximum/) |
| 8 | Largest Rectangle in Histogram | Hard | Monotonic Stack | [LeetCode 84](https://leetcode.com/problems/largest-rectangle-in-histogram/) |
| 9 | Basic Calculator | Hard | Stack Evaluation | [LeetCode 224](https://leetcode.com/problems/basic-calculator/) |

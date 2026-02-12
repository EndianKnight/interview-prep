# Stacks

A LIFO (Last In, First Out) data structure — critical for parsing, expression evaluation, state management, and monotonic patterns.

---

## Core Concepts

- **Push** — add to top, O(1)
- **Pop** — remove from top, O(1)
- **Peek/Top** — view top element without removing, O(1)
- **Empty?** — check if stack has elements, O(1)

### When to Reach for a Stack
- Matching/ balancing (brackets, tags, parentheses)
- Tracking "most recent" state (undo, DFS, function call stack)
- Monotonic problems (next greater/smaller element)
- Expression evaluation (infix, postfix, calculator)
- Reversing sequences

### Language Implementations

| | C++ | Java | Python |
|-|-----|------|--------|
| Recommended | `stack<T>` | `ArrayDeque<T>` | `list` |
| Push | `st.push(x)` | `stack.push(x)` | `stack.append(x)` |
| Pop | `st.pop()` | `stack.pop()` | `stack.pop()` |
| Top/Peek | `st.top()` | `stack.peek()` | `stack[-1]` |
| Empty | `st.empty()` | `stack.isEmpty()` | `not stack` or `len(stack) == 0` |

> ⚠️ **Java:** Don't use the `Stack` class — it's legacy and synchronized. Always prefer `ArrayDeque`.

---

## Pattern 1: Matching / Validation

**When to use:** Balanced brackets, HTML tag matching, expression validation, nested structure verification.

### Example 1a: Valid Parentheses

Given a string containing `()[]{}`, determine if the input is valid.

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

**Complexity:** Time O(n), Space O(n)

### Example 1b: Minimum Remove to Make Valid Parentheses

Remove the minimum number of parentheses to make the string valid.

**Python**
```python
def min_remove_to_make_valid(s: str) -> str:
    stack = []       # indices of unmatched '('
    to_remove = set()

    for i, c in enumerate(s):
        if c == '(':
            stack.append(i)
        elif c == ')':
            if stack:
                stack.pop()      # matched
            else:
                to_remove.add(i) # unmatched ')'

    to_remove.update(stack)      # remaining unmatched '('
    return ''.join(c for i, c in enumerate(s) if i not in to_remove)
```

**C++**
```cpp
string minRemoveToMakeValid(string s) {
    stack<int> st; // indices of unmatched '('
    unordered_set<int> toRemove;

    for (int i = 0; i < s.size(); i++) {
        if (s[i] == '(') {
            st.push(i);
        } else if (s[i] == ')') {
            if (!st.empty()) st.pop();
            else toRemove.insert(i);
        }
    }
    while (!st.empty()) { toRemove.insert(st.top()); st.pop(); }

    string result;
    for (int i = 0; i < s.size(); i++) {
        if (!toRemove.count(i)) result += s[i];
    }
    return result;
}
```

**Complexity:** Time O(n), Space O(n)

### Example 1c: Decode String

Given `s = "3[a2[c]]"`, return `"accaccacc"`.

**Python**
```python
def decode_string(s: str) -> str:
    stack = []
    current_str = ""
    current_num = 0

    for c in s:
        if c.isdigit():
            current_num = current_num * 10 + int(c)
        elif c == '[':
            stack.append((current_str, current_num))
            current_str = ""
            current_num = 0
        elif c == ']':
            prev_str, num = stack.pop()
            current_str = prev_str + current_str * num
        else:
            current_str += c
    return current_str
```

**C++**
```cpp
string decodeString(string s) {
    stack<pair<string, int>> st;
    string curr;
    int num = 0;

    for (char c : s) {
        if (isdigit(c)) {
            num = num * 10 + (c - '0');
        } else if (c == '[') {
            st.push({curr, num});
            curr = "";
            num = 0;
        } else if (c == ']') {
            auto [prev, repeat] = st.top(); st.pop();
            string expanded;
            for (int i = 0; i < repeat; i++) expanded += curr;
            curr = prev + expanded;
        } else {
            curr += c;
        }
    }
    return curr;
}
```

**Java**
```java
public String decodeString(String s) {
    Deque<Object[]> stack = new ArrayDeque<>();
    StringBuilder curr = new StringBuilder();
    int num = 0;

    for (char c : s.toCharArray()) {
        if (Character.isDigit(c)) {
            num = num * 10 + (c - '0');
        } else if (c == '[') {
            stack.push(new Object[]{curr.toString(), num});
            curr = new StringBuilder();
            num = 0;
        } else if (c == ']') {
            Object[] top = stack.pop();
            String prev = (String) top[0];
            int repeat = (int) top[1];
            curr = new StringBuilder(prev + curr.toString().repeat(repeat));
        } else {
            curr.append(c);
        }
    }
    return curr.toString();
}
```

**Complexity:** Time O(output length), Space O(depth of nesting)

---

## Pattern 2: Monotonic Stack

**When to use:** Next greater/smaller element, daily temperatures, stock span, histogram problems. Maintains elements in monotonic (increasing or decreasing) order.

### Core Idea
When a new element violates the monotonic order, pop and process all violated elements. Each element is pushed and popped **at most once**, giving O(n) total time.

### Example 2a: Next Greater Element

For each element, find the next element to the right that is greater.

**C++**
```cpp
#include <vector>
#include <stack>
using namespace std;

vector<int> nextGreaterElement(vector<int>& nums) {
    int n = nums.size();
    vector<int> result(n, -1);
    stack<int> st; // stores indices, monotonically decreasing values

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
    Deque<Integer> stack = new ArrayDeque<>();

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
    stack = []  # stores indices, monotonically decreasing values

    for i in range(n):
        while stack and nums[stack[-1]] < nums[i]:
            result[stack.pop()] = nums[i]
        stack.append(i)
    return result
```

**Complexity:** Time O(n) — each element pushed/popped at most once. Space O(n).

### Example 2b: Daily Temperatures

How many days until a warmer temperature? Returns `[1, 1, 4, 2, 1, 1, 0, 0]` for `[73, 74, 75, 71, 69, 72, 76, 73]`.

**Python**
```python
def daily_temperatures(temperatures: list[int]) -> list[int]:
    n = len(temperatures)
    result = [0] * n
    stack = []  # indices of temperatures waiting for a warmer day

    for i in range(n):
        while stack and temperatures[stack[-1]] < temperatures[i]:
            prev = stack.pop()
            result[prev] = i - prev  # days between
        stack.append(i)
    return result
```

**Java**
```java
public int[] dailyTemperatures(int[] temperatures) {
    int n = temperatures.length;
    int[] result = new int[n];
    Deque<Integer> stack = new ArrayDeque<>();

    for (int i = 0; i < n; i++) {
        while (!stack.isEmpty() && temperatures[stack.peek()] < temperatures[i]) {
            int prev = stack.pop();
            result[prev] = i - prev;
        }
        stack.push(i);
    }
    return result;
}
```

**C++**
```cpp
vector<int> dailyTemperatures(vector<int>& temps) {
    int n = temps.size();
    vector<int> result(n, 0);
    stack<int> st;

    for (int i = 0; i < n; i++) {
        while (!st.empty() && temps[st.top()] < temps[i]) {
            int prev = st.top(); st.pop();
            result[prev] = i - prev;
        }
        st.push(i);
    }
    return result;
}
```

**Complexity:** Time O(n), Space O(n)

### Example 2c: Largest Rectangle in Histogram

Find the area of the largest rectangle that can be formed in a histogram.

**Python**
```python
def largest_rectangle_area(heights: list[int]) -> int:
    stack = []  # (index, height)
    max_area = 0

    for i, h in enumerate(heights):
        start = i
        while stack and stack[-1][1] > h:
            idx, height = stack.pop()
            max_area = max(max_area, height * (i - idx))
            start = idx  # extend new bar back
        stack.append((start, h))

    # Handle remaining bars that extend to the end
    for idx, height in stack:
        max_area = max(max_area, height * (len(heights) - idx))
    return max_area
```

**C++**
```cpp
int largestRectangleArea(vector<int>& heights) {
    stack<pair<int, int>> st; // (index, height)
    int maxArea = 0;

    for (int i = 0; i < heights.size(); i++) {
        int start = i;
        while (!st.empty() && st.top().second > heights[i]) {
            auto [idx, h] = st.top(); st.pop();
            maxArea = max(maxArea, h * (i - idx));
            start = idx;
        }
        st.push({start, heights[i]});
    }
    for (auto& [idx, h] : st) {
        maxArea = max(maxArea, h * (int)(heights.size() - idx));
    }
    return maxArea;
}
```

**Java**
```java
public int largestRectangleArea(int[] heights) {
    Deque<int[]> stack = new ArrayDeque<>(); // [index, height]
    int maxArea = 0;

    for (int i = 0; i < heights.length; i++) {
        int start = i;
        while (!stack.isEmpty() && stack.peek()[1] > heights[i]) {
            int[] top = stack.pop();
            maxArea = Math.max(maxArea, top[1] * (i - top[0]));
            start = top[0];
        }
        stack.push(new int[]{start, heights[i]});
    }
    while (!stack.isEmpty()) {
        int[] top = stack.pop();
        maxArea = Math.max(maxArea, top[1] * (heights.length - top[0]));
    }
    return maxArea;
}
```

**Complexity:** Time O(n), Space O(n)

### Example 2d: Stock Span Problem

For each day, how many consecutive previous days had price ≤ today's price?

**Python**
```python
class StockSpanner:
    def __init__(self):
        self.stack = []  # (price, span)

    def next(self, price: int) -> int:
        span = 1
        while self.stack and self.stack[-1][0] <= price:
            span += self.stack.pop()[1]
        self.stack.append((price, span))
        return span
```

---

## Pattern 3: Stack-Based Expression Evaluation

**When to use:** Calculator problems, RPN (postfix), infix to postfix conversion.

### Example 3a: Evaluate Reverse Polish Notation

Evaluate `["2","1","+","3","*"]` → `(2+1)*3 = 9`

**C++**
```cpp
int evalRPN(vector<string>& tokens) {
    stack<int> st;

    for (const string& t : tokens) {
        if (t == "+" || t == "-" || t == "*" || t == "/") {
            int b = st.top(); st.pop();
            int a = st.top(); st.pop();
            if (t == "+") st.push(a + b);
            else if (t == "-") st.push(a - b);
            else if (t == "*") st.push(a * b);
            else st.push(a / b); // truncates toward zero in C++
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
        '/': lambda a, b: int(a / b),  # truncate toward zero
    }

    for t in tokens:
        if t in ops:
            b, a = stack.pop(), stack.pop()
            stack.append(ops[t](a, b))
        else:
            stack.append(int(t))
    return stack[0]
```

### Example 3b: Basic Calculator (with +, -, parentheses)

Evaluate `"(1+(4+5+2)-3)+(6+8)"` → `23`

**Python**
```python
def calculate(s: str) -> int:
    stack = []
    num = 0
    sign = 1
    result = 0

    for c in s:
        if c.isdigit():
            num = num * 10 + int(c)
        elif c in '+-':
            result += sign * num
            num = 0
            sign = 1 if c == '+' else -1
        elif c == '(':
            stack.append(result)
            stack.append(sign)
            result = 0
            sign = 1
        elif c == ')':
            result += sign * num
            num = 0
            result *= stack.pop()  # sign before paren
            result += stack.pop()  # result before paren

    return result + sign * num
```

**C++**
```cpp
int calculate(string s) {
    stack<int> st;
    int num = 0, sign = 1, result = 0;

    for (char c : s) {
        if (isdigit(c)) {
            num = num * 10 + (c - '0');
        } else if (c == '+' || c == '-') {
            result += sign * num;
            num = 0;
            sign = (c == '+') ? 1 : -1;
        } else if (c == '(') {
            st.push(result);
            st.push(sign);
            result = 0;
            sign = 1;
        } else if (c == ')') {
            result += sign * num;
            num = 0;
            result *= st.top(); st.pop(); // sign
            result += st.top(); st.pop(); // prev result
        }
    }
    return result + sign * num;
}
```

---

## Pattern 4: Min/Max Stack

**When to use:** Track the minimum or maximum in O(1) alongside standard stack operations.

### Example 4a: Min Stack

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

### Example 4b: Max Stack (with pop max)

A more advanced variant that also supports `popMax()`:

**Python**
```python
from sortedcontainers import SortedList

class MaxStack:
    def __init__(self):
        self.stack = []         # (value, id)
        self.sorted = SortedList()  # (value, id)
        self.removed = set()
        self.id = 0

    def push(self, x: int) -> None:
        self.stack.append((x, self.id))
        self.sorted.add((x, self.id))
        self.id += 1

    def pop(self) -> int:
        while self.stack[-1][1] in self.removed:
            self.stack.pop()
        val, uid = self.stack.pop()
        self.sorted.remove((val, uid))
        return val

    def top(self) -> int:
        while self.stack[-1][1] in self.removed:
            self.stack.pop()
        return self.stack[-1][0]

    def peek_max(self) -> int:
        while self.sorted[-1][1] in self.removed:
            self.sorted.pop()
        return self.sorted[-1][0]

    def pop_max(self) -> int:
        while self.sorted[-1][1] in self.removed:
            self.sorted.pop()
        val, uid = self.sorted.pop()
        self.removed.add(uid)
        return val
```

---

## Pattern 5: Stack for DFS / Iterative Traversal

**When to use:** Converting recursive DFS to iterative, or when recursion depth is a concern.

### Example 5a: Iterative Preorder Traversal

**C++**
```cpp
vector<int> preorderTraversal(TreeNode* root) {
    if (!root) return {};
    vector<int> result;
    stack<TreeNode*> st;
    st.push(root);
    while (!st.empty()) {
        auto* node = st.top(); st.pop();
        result.push_back(node->val);
        if (node->right) st.push(node->right);
        if (node->left) st.push(node->left);
    }
    return result;
}
```

**Java**
```java
public List<Integer> preorderTraversal(TreeNode root) {
    List<Integer> result = new ArrayList<>();
    if (root == null) return result;
    Deque<TreeNode> stack = new ArrayDeque<>();
    stack.push(root);
    while (!stack.isEmpty()) {
        TreeNode node = stack.pop();
        result.add(node.val);
        if (node.right != null) stack.push(node.right);
        if (node.left != null) stack.push(node.left);
    }
    return result;
}
```

**Python**
```python
def preorder_traversal(root) -> list[int]:
    if not root:
        return []
    result = []
    stack = [root]
    while stack:
        node = stack.pop()
        result.append(node.val)
        if node.right: stack.append(node.right)  # right first
        if node.left:  stack.append(node.left)    # so left is popped first
    return result
```

### Example 5b: Flatten Nested List Iterator

Given a nested list like `[[1,1],2,[1,1]]`, iterate through values `[1,1,2,1,1]`.

**Python**
```python
class NestedIterator:
    def __init__(self, nested_list):
        self.stack = list(reversed(nested_list))

    def next(self) -> int:
        return self.stack.pop().getInteger()

    def has_next(self) -> bool:
        while self.stack and not self.stack[-1].isInteger():
            top = self.stack.pop()
            self.stack.extend(reversed(top.getList()))
        return bool(self.stack)
```

---

## Pitfalls & Edge Cases

| Pitfall | How to Handle |
|---------|--------------|
| Popping from empty stack | Always check `empty()` / `not stack` first |
| Java `Stack` class | Use `ArrayDeque` — `Stack` is legacy/synchronized |
| Integer division in RPN | Python `//` floors toward −∞; use `int(a/b)` for truncation toward 0 |
| Operator precedence in calculators | Handle `*` `/` before `+` `-`; use two stacks or Shunting-yard |
| Forgetting to process remaining stack | After loop, check if stack still has unprocessed items |
| Monotonic stack direction | **Decreasing** from bottom→top for "next greater" |

---

## Practice Problems

| # | Problem | Difficulty | Pattern | Link |
|---|---------|-----------|---------|------|
| 1 | Valid Parentheses | Easy | Matching | [LeetCode 20](https://leetcode.com/problems/valid-parentheses/) |
| 2 | Min Stack | Medium | Min/Max Stack | [LeetCode 155](https://leetcode.com/problems/min-stack/) |
| 3 | Daily Temperatures | Medium | Monotonic Stack | [LeetCode 739](https://leetcode.com/problems/daily-temperatures/) |
| 4 | Evaluate RPN | Medium | Evaluation | [LeetCode 150](https://leetcode.com/problems/evaluate-reverse-polish-notation/) |
| 5 | Decode String | Medium | Matching | [LeetCode 394](https://leetcode.com/problems/decode-string/) |
| 6 | Remove Invalid Parentheses | Medium | Matching | [LeetCode 1249](https://leetcode.com/problems/minimum-remove-to-make-valid-parentheses/) |
| 7 | Next Greater Element | Easy | Monotonic Stack | [LeetCode 496](https://leetcode.com/problems/next-greater-element-i/) |
| 8 | Online Stock Span | Medium | Monotonic Stack | [LeetCode 901](https://leetcode.com/problems/online-stock-span/) |
| 9 | Largest Rectangle in Histogram | Hard | Monotonic Stack | [LeetCode 84](https://leetcode.com/problems/largest-rectangle-in-histogram/) |
| 10 | Basic Calculator | Hard | Expression Eval | [LeetCode 224](https://leetcode.com/problems/basic-calculator/) |
| 11 | Trapping Rain Water | Hard | Monotonic Stack | [LeetCode 42](https://leetcode.com/problems/trapping-rain-water/) |

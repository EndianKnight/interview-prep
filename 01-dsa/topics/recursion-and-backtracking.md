# Recursion & Backtracking

The core technique for generating all combinations, permutations, subsets, and solving constraint-satisfaction problems.

---

## Core Concepts

### Recursion
- Function calls itself with a smaller input
- Must have a **base case** to stop
- Each call adds a frame to the call stack — O(depth) space
- Think: "if I solve the smaller version, how do I build the full answer?"

### Backtracking
- Systematic exploration of all candidates
- **Choose → Explore → Unchoose** (undo the choice)
- Prune invalid branches early to reduce search space
- Time complexity often O(2ⁿ) or O(n!) — but pruning helps

---

## Pattern 1: Subsets / Power Set

**When to use:** Generate all subsets, combinations, or subsequences.

### Example: All Subsets

**C++**
```cpp
vector<vector<int>> subsets(vector<int>& nums) {
    vector<vector<int>> result;
    vector<int> current;

    function<void(int)> backtrack = [&](int start) {
        result.push_back(current);
        for (int i = start; i < nums.size(); i++) {
            current.push_back(nums[i]);
            backtrack(i + 1);
            current.pop_back();
        }
    };
    backtrack(0);
    return result;
}
```

**Java**
```java
public List<List<Integer>> subsets(int[] nums) {
    List<List<Integer>> result = new ArrayList<>();
    backtrack(nums, 0, new ArrayList<>(), result);
    return result;
}

private void backtrack(int[] nums, int start, List<Integer> current,
                       List<List<Integer>> result) {
    result.add(new ArrayList<>(current));
    for (int i = start; i < nums.length; i++) {
        current.add(nums[i]);
        backtrack(nums, i + 1, current, result);
        current.remove(current.size() - 1);
    }
}
```

**Python**
```python
def subsets(nums: list[int]) -> list[list[int]]:
    result = []

    def backtrack(start: int, current: list[int]):
        result.append(current[:])  # copy
        for i in range(start, len(nums)):
            current.append(nums[i])
            backtrack(i + 1, current)
            current.pop()

    backtrack(0, [])
    return result
```

---

## Pattern 2: Permutations

**When to use:** Generate all orderings of elements.

### Example: All Permutations

**Python**
```python
def permute(nums: list[int]) -> list[list[int]]:
    result = []

    def backtrack(current: list[int], remaining: set):
        if not remaining:
            result.append(current[:])
            return
        for num in list(remaining):
            current.append(num)
            remaining.remove(num)
            backtrack(current, remaining)
            current.pop()
            remaining.add(num)

    backtrack([], set(nums))
    return result

# Swap-based (in-place, more efficient)
def permute_swap(nums: list[int]) -> list[list[int]]:
    result = []

    def backtrack(start: int):
        if start == len(nums):
            result.append(nums[:])
            return
        for i in range(start, len(nums)):
            nums[start], nums[i] = nums[i], nums[start]
            backtrack(start + 1)
            nums[start], nums[i] = nums[i], nums[start]

    backtrack(0)
    return result
```

**C++**
```cpp
vector<vector<int>> permute(vector<int>& nums) {
    vector<vector<int>> result;
    function<void(int)> backtrack = [&](int start) {
        if (start == nums.size()) {
            result.push_back(nums);
            return;
        }
        for (int i = start; i < nums.size(); i++) {
            swap(nums[start], nums[i]);
            backtrack(start + 1);
            swap(nums[start], nums[i]);
        }
    };
    backtrack(0);
    return result;
}
```

---

## Pattern 3: Combinations

**When to use:** Choose K elements from N, combination sum.

### Example: Combination Sum

Find all unique combinations that sum to target (elements reusable).

**Python**
```python
def combination_sum(candidates: list[int], target: int) -> list[list[int]]:
    result = []

    def backtrack(start: int, current: list[int], remaining: int):
        if remaining == 0:
            result.append(current[:])
            return
        if remaining < 0:
            return
        for i in range(start, len(candidates)):
            current.append(candidates[i])
            backtrack(i, current, remaining - candidates[i])  # i, not i+1 (reuse)
            current.pop()

    backtrack(0, [], target)
    return result
```

**C++**
```cpp
vector<vector<int>> combinationSum(vector<int>& candidates, int target) {
    vector<vector<int>> result;
    vector<int> current;

    function<void(int, int)> backtrack = [&](int start, int remaining) {
        if (remaining == 0) { result.push_back(current); return; }
        if (remaining < 0) return;
        for (int i = start; i < candidates.size(); i++) {
            current.push_back(candidates[i]);
            backtrack(i, remaining - candidates[i]);
            current.pop_back();
        }
    };
    backtrack(0, target);
    return result;
}
```

---

## Pattern 4: Grid / Board Search

**When to use:** Word search, Sudoku solver, N-Queens, maze solving.

### Example: N-Queens

**Python**
```python
def solve_n_queens(n: int) -> list[list[str]]:
    result = []
    cols, diag1, diag2 = set(), set(), set()
    board = [['.'] * n for _ in range(n)]

    def backtrack(row: int):
        if row == n:
            result.append([''.join(r) for r in board])
            return
        for col in range(n):
            if col in cols or (row - col) in diag1 or (row + col) in diag2:
                continue
            board[row][col] = 'Q'
            cols.add(col)
            diag1.add(row - col)
            diag2.add(row + col)

            backtrack(row + 1)

            board[row][col] = '.'
            cols.remove(col)
            diag1.remove(row - col)
            diag2.remove(row + col)

    backtrack(0)
    return result
```

### Example: Word Search

**Python**
```python
def exist(board: list[list[str]], word: str) -> bool:
    rows, cols = len(board), len(board[0])

    def dfs(r: int, c: int, i: int) -> bool:
        if i == len(word):
            return True
        if (r < 0 or r >= rows or c < 0 or c >= cols or
            board[r][c] != word[i]):
            return False

        temp = board[r][c]
        board[r][c] = '#'  # mark visited
        found = any(dfs(r + dr, c + dc, i + 1)
                     for dr, dc in [(0,1),(0,-1),(1,0),(-1,0)])
        board[r][c] = temp  # unmark
        return found

    return any(dfs(r, c, 0) for r in range(rows) for c in range(cols))
```

---

## Backtracking Template

```
def backtrack(state):
    if is_solution(state):
        record(state)
        return

    for choice in get_choices(state):
        if is_valid(choice, state):
            make_choice(choice, state)      # choose
            backtrack(state)                 # explore
            undo_choice(choice, state)       # unchoose
```

---

## Handling Duplicates

To avoid duplicate results when input has duplicates:
1. **Sort** the input first
2. **Skip** if `nums[i] == nums[i-1]` and `i > start`

**Python**
```python
def subsets_with_dup(nums: list[int]) -> list[list[int]]:
    nums.sort()
    result = []

    def backtrack(start, current):
        result.append(current[:])
        for i in range(start, len(nums)):
            if i > start and nums[i] == nums[i-1]:
                continue  # skip duplicate
            current.append(nums[i])
            backtrack(i + 1, current)
            current.pop()

    backtrack(0, [])
    return result
```

---

## Pitfalls & Edge Cases

| Pitfall | How to Handle |
|---------|--------------|
| Forgetting to copy `current` | `result.append(current[:])` not `result.append(current)` |
| Infinite recursion | Ensure base case is reachable; track visited cells |
| Duplicate results | Sort input + skip adjacent duplicates |
| Not undoing choice | Always undo after recursive call |
| TLE on large inputs | Add pruning (e.g., sort + early termination) |

---

## Practice Problems

| # | Problem | Difficulty | Pattern | Link |
|---|---------|-----------|---------|------|
| 1 | Subsets | Medium | Subsets | [LeetCode 78](https://leetcode.com/problems/subsets/) |
| 2 | Permutations | Medium | Permutations | [LeetCode 46](https://leetcode.com/problems/permutations/) |
| 3 | Combination Sum | Medium | Combinations | [LeetCode 39](https://leetcode.com/problems/combination-sum/) |
| 4 | Letter Combinations (Phone) | Medium | Combinations | [LeetCode 17](https://leetcode.com/problems/letter-combinations-of-a-phone-number/) |
| 5 | Word Search | Medium | Grid DFS | [LeetCode 79](https://leetcode.com/problems/word-search/) |
| 6 | Palindrome Partitioning | Medium | Subsets | [LeetCode 131](https://leetcode.com/problems/palindrome-partitioning/) |
| 7 | N-Queens | Hard | Board | [LeetCode 51](https://leetcode.com/problems/n-queens/) |
| 8 | Sudoku Solver | Hard | Board | [LeetCode 37](https://leetcode.com/problems/sudoku-solver/) |

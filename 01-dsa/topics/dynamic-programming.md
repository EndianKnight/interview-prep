# Dynamic Programming

The most feared interview topic — but it's just pattern recognition + optimal substructure + overlapping subproblems.

---

## Core Concepts

### When to Use DP
1. **Optimal substructure** — optimal solution built from optimal sub-solutions
2. **Overlapping subproblems** — same subproblems solved repeatedly
3. Key phrases: "minimum cost," "maximum profit," "number of ways," "can you reach"

### Two Approaches
| Approach | Description | Pros | Cons |
|----------|------------|------|------|
| **Top-Down (Memoization)** | Recursion + cache | Intuitive, only computes needed states | Stack overflow risk |
| **Bottom-Up (Tabulation)** | Iterative, fill table | No stack overflow, often faster | Must determine order |

### Framework
1. **Define state** — what variables uniquely describe a subproblem?
2. **Recurrence relation** — how does the current state relate to previous states?
3. **Base cases** — what are the trivial cases?
4. **Iteration order** — in what order to fill the table?
5. **Space optimization** — can we reduce from O(n²) to O(n)?

---

## Pattern 1: 1D DP (Linear Sequence)

**When to use:** Climbing stairs, house robber, coin change, decode ways.

### Example: Climbing Stairs

**C++**
```cpp
int climbStairs(int n) {
    if (n <= 2) return n;
    int prev2 = 1, prev1 = 2;
    for (int i = 3; i <= n; i++) {
        int curr = prev1 + prev2;
        prev2 = prev1;
        prev1 = curr;
    }
    return prev1;
}
```

**Java**
```java
public int climbStairs(int n) {
    if (n <= 2) return n;
    int prev2 = 1, prev1 = 2;
    for (int i = 3; i <= n; i++) {
        int curr = prev1 + prev2;
        prev2 = prev1;
        prev1 = curr;
    }
    return prev1;
}
```

**Python**
```python
def climb_stairs(n: int) -> int:
    if n <= 2:
        return n
    prev2, prev1 = 1, 2
    for _ in range(3, n + 1):
        prev2, prev1 = prev1, prev1 + prev2
    return prev1
```

### Example: House Robber

**C++**
```cpp
int rob(vector<int>& nums) {
    int prev2 = 0, prev1 = 0;
    for (int num : nums) {
        int curr = max(prev1, prev2 + num);
        prev2 = prev1;
        prev1 = curr;
    }
    return prev1;
}
```

**Java**
```java
public int rob(int[] nums) {
    int prev2 = 0, prev1 = 0;
    for (int num : nums) {
        int curr = Math.max(prev1, prev2 + num);
        prev2 = prev1;
        prev1 = curr;
    }
    return prev1;
}
```

**Python**
```python
def rob(nums: list[int]) -> int:
    if not nums:
        return 0
    prev2, prev1 = 0, 0
    for num in nums:
        prev2, prev1 = prev1, max(prev1, prev2 + num)
    return prev1
```

### Example: Coin Change (Minimum Coins)

**C++**
```cpp
int coinChange(vector<int>& coins, int amount) {
    vector<int> dp(amount + 1, amount + 1);
    dp[0] = 0;
    for (int i = 1; i <= amount; i++) {
        for (int coin : coins) {
            if (coin <= i) {
                dp[i] = min(dp[i], dp[i - coin] + 1);
            }
        }
    }
    return dp[amount] > amount ? -1 : dp[amount];
}
```

**Java**
```java
public int coinChange(int[] coins, int amount) {
    int[] dp = new int[amount + 1];
    Arrays.fill(dp, amount + 1);
    dp[0] = 0;
    for (int i = 1; i <= amount; i++) {
        for (int coin : coins) {
            if (coin <= i) {
                dp[i] = Math.min(dp[i], dp[i - coin] + 1);
            }
        }
    }
    return dp[amount] > amount ? -1 : dp[amount];
}
```

**Python**
```python
def coin_change(coins: list[int], amount: int) -> int:
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0
    for i in range(1, amount + 1):
        for coin in coins:
            if coin <= i:
                dp[i] = min(dp[i], dp[i - coin] + 1)
    return dp[amount] if dp[amount] != float('inf') else -1
```

---

## Pattern 2: 2D DP (Grid / Two Sequences)

**When to use:** Grid paths, longest common subsequence, edit distance, 0/1 knapsack.

### Example: Unique Paths

**C++**
```cpp
int uniquePaths(int m, int n) {
    vector<int> row(n, 1);
    for (int i = 1; i < m; i++)
        for (int j = 1; j < n; j++)
            row[j] += row[j-1];
    return row[n-1];
}
```

**Java**
```java
public int uniquePaths(int m, int n) {
    int[] row = new int[n];
    Arrays.fill(row, 1);
    for (int i = 1; i < m; i++)
        for (int j = 1; j < n; j++)
            row[j] += row[j-1];
    return row[n-1];
}
```

**Python**
```python
def unique_paths(m: int, n: int) -> int:
    row = [1] * n
    for i in range(1, m):
        for j in range(1, n):
            row[j] += row[j-1]
    return row[-1]
```

### Example: Longest Common Subsequence

**C++**
```cpp
int longestCommonSubsequence(string text1, string text2) {
    int m = text1.size(), n = text2.size();
    vector<vector<int>> dp(m + 1, vector<int>(n + 1, 0));

    for (int i = 1; i <= m; i++) {
        for (int j = 1; j <= n; j++) {
            if (text1[i-1] == text2[j-1]) {
                dp[i][j] = dp[i-1][j-1] + 1;
            } else {
                dp[i][j] = max(dp[i-1][j], dp[i][j-1]);
            }
        }
    }
    return dp[m][n];
}
```

**Python**
```python
def longest_common_subsequence(text1: str, text2: str) -> int:
    m, n = len(text1), len(text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i-1] == text2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return dp[m][n]
```

### Example: 0/1 Knapsack

**C++**
```cpp
int knapsack(vector<int>& weights, vector<int>& values, int capacity) {
    int n = weights.size();
    vector<int> dp(capacity + 1, 0);
    for (int i = 0; i < n; i++)
        for (int w = capacity; w >= weights[i]; w--) // reverse!
            dp[w] = max(dp[w], dp[w - weights[i]] + values[i]);
    return dp[capacity];
}
```

**Java**
```java
public int knapsack(int[] weights, int[] values, int capacity) {
    int[] dp = new int[capacity + 1];
    for (int i = 0; i < weights.length; i++)
        for (int w = capacity; w >= weights[i]; w--) // reverse!
            dp[w] = Math.max(dp[w], dp[w - weights[i]] + values[i]);
    return dp[capacity];
}
```

**Python**
```python
def knapsack(weights: list[int], values: list[int], capacity: int) -> int:
    dp = [0] * (capacity + 1)
    for i in range(len(weights)):
        for w in range(capacity, weights[i] - 1, -1):  # reverse!
            dp[w] = max(dp[w], dp[w - weights[i]] + values[i])
    return dp[capacity]
```

---

## Pattern 3: String DP

**When to use:** Edit distance, palindromic subsequences, word break, regex matching.

### Example: Edit Distance

**C++**
```cpp
int minDistance(string word1, string word2) {
    int m = word1.size(), n = word2.size();
    vector<vector<int>> dp(m + 1, vector<int>(n + 1, 0));
    for (int i = 0; i <= m; i++) dp[i][0] = i;
    for (int j = 0; j <= n; j++) dp[0][j] = j;

    for (int i = 1; i <= m; i++) {
        for (int j = 1; j <= n; j++) {
            if (word1[i-1] == word2[j-1])
                dp[i][j] = dp[i-1][j-1];
            else
                dp[i][j] = 1 + min({dp[i-1][j], dp[i][j-1], dp[i-1][j-1]});
        }
    }
    return dp[m][n];
}
```

**Java**
```java
public int minDistance(String word1, String word2) {
    int m = word1.length(), n = word2.length();
    int[][] dp = new int[m + 1][n + 1];
    for (int i = 0; i <= m; i++) dp[i][0] = i;
    for (int j = 0; j <= n; j++) dp[0][j] = j;

    for (int i = 1; i <= m; i++) {
        for (int j = 1; j <= n; j++) {
            if (word1.charAt(i-1) == word2.charAt(j-1))
                dp[i][j] = dp[i-1][j-1];
            else
                dp[i][j] = 1 + Math.min(dp[i-1][j-1],
                               Math.min(dp[i-1][j], dp[i][j-1]));
        }
    }
    return dp[m][n];
}
```

**Python**
```python
def min_distance(word1: str, word2: str) -> int:
    m, n = len(word1), len(word2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if word1[i-1] == word2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(
                    dp[i-1][j],    # delete
                    dp[i][j-1],    # insert
                    dp[i-1][j-1]   # replace
                )
    return dp[m][n]
```

---

## Pattern 4: Decision Making (Buy/Sell Stock)

**When to use:** Multiple transactions, with cooldown, with fees — state machine DP.

### Example: Best Time to Buy and Sell Stock with Cooldown

**C++**
```cpp
int maxProfit(vector<int>& prices) {
    if (prices.size() < 2) return 0;
    int hold = -prices[0], sold = 0, cooldown = 0;
    for (int i = 1; i < prices.size(); i++) {
        int prevHold = hold, prevSold = sold;
        hold = max(prevHold, cooldown - prices[i]);
        sold = prevHold + prices[i];
        cooldown = max(cooldown, prevSold);
    }
    return max(sold, cooldown);
}
```

**Java**
```java
public int maxProfit(int[] prices) {
    if (prices.length < 2) return 0;
    int hold = -prices[0], sold = 0, cooldown = 0;
    for (int i = 1; i < prices.length; i++) {
        int prevHold = hold, prevSold = sold;
        hold = Math.max(prevHold, cooldown - prices[i]);
        sold = prevHold + prices[i];
        cooldown = Math.max(cooldown, prevSold);
    }
    return Math.max(sold, cooldown);
}
```

**Python**
```python
def max_profit(prices: list[int]) -> int:
    if len(prices) < 2:
        return 0
    # States: hold, sold, cooldown
    hold, sold, cooldown = -prices[0], 0, 0

    for price in prices[1:]:
        prev_hold, prev_sold, prev_cooldown = hold, sold, cooldown
        hold = max(prev_hold, prev_cooldown - price)
        sold = prev_hold + price
        cooldown = max(prev_cooldown, prev_sold)

    return max(sold, cooldown)
```

---

## Pattern 5: Top-Down Memoization Template

**Python**
```python
from functools import lru_cache

def solve(args):
    @lru_cache(maxsize=None)
    def dp(state):
        # base case
        if base_condition(state):
            return base_value

        # recurrence
        result = initial_value
        for choice in choices(state):
            result = optimize(result, dp(next_state(state, choice)))
        return result

    return dp(initial_state)
```

---

## Pitfalls & Edge Cases

| Pitfall | How to Handle |
|---------|--------------|
| Wrong base case | Trace smallest inputs manually |
| Wrong iteration order | Bottom-up: ensure dependencies computed first |
| Off-by-one in table size | Usually `n+1` for 1D, `(m+1) x (n+1)` for 2D |
| Not considering "skip" option | Knapsack, LIS — always include "don't take" branch |
| Stack overflow (top-down) | Switch to bottom-up or increase recursion limit |
| Space optimization direction | 0/1 knapsack: iterate capacity in reverse |

---

## Practice Problems

| # | Problem | Difficulty | Pattern | Link |
|---|---------|-----------|---------|------|
| 1 | Climbing Stairs | Easy | 1D | [LeetCode 70](https://leetcode.com/problems/climbing-stairs/) |
| 2 | House Robber | Medium | 1D | [LeetCode 198](https://leetcode.com/problems/house-robber/) |
| 3 | Coin Change | Medium | 1D | [LeetCode 322](https://leetcode.com/problems/coin-change/) |
| 4 | Longest Increasing Subsequence | Medium | 1D + Binary Search | [LeetCode 300](https://leetcode.com/problems/longest-increasing-subsequence/) |
| 5 | Unique Paths | Medium | 2D Grid | [LeetCode 62](https://leetcode.com/problems/unique-paths/) |
| 6 | Longest Common Subsequence | Medium | 2D String | [LeetCode 1143](https://leetcode.com/problems/longest-common-subsequence/) |
| 7 | Word Break | Medium | 1D + Set | [LeetCode 139](https://leetcode.com/problems/word-break/) |
| 8 | Edit Distance | Medium | 2D String | [LeetCode 72](https://leetcode.com/problems/edit-distance/) |
| 9 | Best Time Buy/Sell Stock with Cooldown | Medium | State Machine | [LeetCode 309](https://leetcode.com/problems/best-time-to-buy-and-sell-stock-with-cooldown/) |
| 10 | Partition Equal Subset Sum | Medium | 0/1 Knapsack | [LeetCode 416](https://leetcode.com/problems/partition-equal-subset-sum/) |

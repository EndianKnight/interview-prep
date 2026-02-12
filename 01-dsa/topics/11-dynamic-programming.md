# Dynamic Programming

The most feared interview topic — but it's just pattern recognition + optimal substructure + overlapping subproblems.

---

## Core Concepts

### When to Use DP
1. **Optimal substructure** — optimal solution built from optimal sub-solutions
2. **Overlapping subproblems** — same subproblems solved repeatedly
3. Key phrases: "minimum cost," "maximum profit," "number of ways," "can you reach"

### Framework (5 Steps)
1. **Define state** — what variables uniquely describe a subproblem?
2. **Recurrence relation** — how does the current state relate to previous states?
3. **Base cases** — what are the trivial cases?
4. **Iteration order** — in what order to fill the table?
5. **Space optimization** — can we reduce from O(n²) to O(n)?

---

## Top-Down vs Bottom-Up — Deep Dive

This is the most important conceptual distinction in DP. Mastering both approaches (and knowing when to use each) is what separates average from great DP solvers.

### Top-Down (Memoization)

**Approach:** Write the recursive solution first, then add a cache.

```python
from functools import lru_cache

def fibonacci(n):
    @lru_cache(maxsize=None)
    def dp(i):
        if i <= 1:
            return i
        return dp(i - 1) + dp(i - 2)
    return dp(n)
```

**Why it's easier:**
- Start from the **big problem**, break it into smaller ones
- You only think about **"what do I need to solve this?"** — the recursion naturally explores only needed states
- Cache handles deduplication automatically

**When to prefer:**
- Problem has complex state space where not all states are reachable
- Tree/graph-based DP (e.g. tree diameter, path problems)
- You're under time pressure and need a correct solution fast
- State transitions are hard to enumerate in order

### Bottom-Up (Tabulation)

**Approach:** Build a table from smallest subproblems to largest.

```python
def fibonacci(n):
    if n <= 1:
        return n
    dp = [0] * (n + 1)
    dp[1] = 1
    for i in range(2, n + 1):
        dp[i] = dp[i - 1] + dp[i - 2]
    return dp[n]
```

**Why it's faster in practice:**
- No recursion overhead or stack limit
- Cache-friendly memory access (sequential iteration)
- Easier to apply space optimization (rolling arrays)

**When to prefer:**
- All states are computed anyway (grid problems, string matching)
- You need space optimization
- Problem has a natural "small → large" ordering
- In production code where stack overflow is a concern

### How to Develop Bottom-Up Intuition

This is the hardest part of DP. Here's a systematic process:

#### Step 1: Solve Top-Down First
Always start with the recursive solution. Don't try to jump straight to bottom-up.

```python
# Top-down for Coin Change
@lru_cache(maxsize=None)
def dp(amount):
    if amount == 0: return 0
    if amount < 0: return float('inf')
    return 1 + min(dp(amount - coin) for coin in coins)
```

#### Step 2: Identify the State Dimensions
What arguments does your recursive function take? Those become your DP table dimensions.
- `dp(i)` → 1D array `dp[i]`
- `dp(i, j)` → 2D array `dp[i][j]`
- `dp(i, j, k)` → 3D or use a hashmap

#### Step 3: Determine Iteration Order from Dependencies

**This is the key insight:** Look at which states each state depends on, then iterate so that dependencies are computed first.

```
If dp[i] depends on dp[i-1] and dp[i-2]
→ iterate i from small to large (left to right)

If dp[i][j] depends on dp[i-1][j-1], dp[i-1][j], dp[i][j-1]
→ iterate i from top to bottom, j from left to right

If dp[i] depends on dp[i+1] and dp[i+2]
→ iterate i from large to small (right to left)
```

**Visual: Dependency Direction → Iteration Direction**
```
Dependency    │ Iteration Order
──────────────┼─────────────────
dp[i-1]       │ i: 0 → n  (left to right)
dp[i+1]       │ i: n → 0  (right to left)
dp[i-1][j-1]  │ i: 0 → m, j: 0 → n  (top-left to bottom-right)
dp[i+1][j+1]  │ i: m → 0, j: n → 0  (bottom-right to top-left)
dp[i][j-1]    │ j: 0 → n  (left to right within each row)
```

#### Step 4: Translate Base Cases
Recursive base cases become initial values in your table:
```
if i == 0: return X       →     dp[0] = X
if i == n: return Y       →     dp[n] = Y
if i < 0: return INF      →     (handle bounds in loop)
```

#### Step 5: Apply Space Optimization
If `dp[i]` only depends on `dp[i-1]` (not `dp[i-3]` or earlier), keep only 2 rows instead of n:

```python
# Before: O(m × n) space
dp = [[0] * (n + 1) for _ in range(m + 1)]

# After: O(n) space — rolling two rows
prev = [0] * (n + 1)
curr = [0] * (n + 1)
for i in range(1, m + 1):
    for j in range(1, n + 1):
        curr[j] = ... # use prev[j], prev[j-1], curr[j-1]
    prev, curr = curr, [0] * (n + 1)
```

If `dp[i]` depends only on `dp[i-1]` at the same or earlier `j` index, you can use a single array iterating normally. If it depends on `dp[i-1]` at `j` or later, iterate `j` in **reverse** (this is why 0/1 knapsack iterates capacity in reverse).

### Side-by-Side Comparison — Coin Change

**Top-Down:**
```python
def coin_change_td(coins, amount):
    @lru_cache(maxsize=None)
    def dp(amt):
        if amt == 0: return 0
        if amt < 0: return float('inf')
        return 1 + min((dp(amt - c) for c in coins), default=float('inf'))
    res = dp(amount)
    return res if res != float('inf') else -1
```

**Bottom-Up:**
```python
def coin_change_bu(coins, amount):
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0
    for i in range(1, amount + 1):           # state: amount remaining
        for c in coins:                       # choice: which coin to use
            if c <= i:
                dp[i] = min(dp[i], dp[i - c] + 1)  # recurrence
    return dp[amount] if dp[amount] != float('inf') else -1
```

**Notice:** The top-down `dp(amt)` becomes `dp[i]` bottom-up. Dependencies go from `amt - c` → `amt`, so we iterate `i` left→right.

### Comparison Table

| Aspect | Top-Down | Bottom-Up |
|--------|----------|-----------|
| Thinking direction | "What do I need?" | "What can I build from?" |
| Code structure | Recursive + cache | Iterative + table |
| States computed | Only reachable | All states |
| Space optimization | Hard (need all states cached) | Easy (rolling arrays) |
| Stack overflow risk | Yes (deep recursion) | No |
| Debugging | Harder (trace recursion) | Easier (print table) |
| Implementation speed | Faster to write | Slower to write |
| Runtime constants | Slower (function call overhead) | Faster |

---

## Pattern 1: 1D DP (Linear Sequence)

**When to use:** Climbing stairs, house robber, coin change, decode ways, jump game.

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

### Example: Decode Ways

**C++**
```cpp
int numDecodings(string s) {
    int n = s.size();
    if (n == 0 || s[0] == '0') return 0;
    int prev2 = 1, prev1 = 1;  // dp[0]=1, dp[1]=1
    for (int i = 2; i <= n; i++) {
        int curr = 0;
        if (s[i-1] != '0') curr += prev1;                    // single digit
        int two = stoi(s.substr(i-2, 2));
        if (two >= 10 && two <= 26) curr += prev2;           // two digits
        prev2 = prev1;
        prev1 = curr;
    }
    return prev1;
}
```

**Java**
```java
public int numDecodings(String s) {
    if (s.isEmpty() || s.charAt(0) == '0') return 0;
    int prev2 = 1, prev1 = 1;
    for (int i = 2; i <= s.length(); i++) {
        int curr = 0;
        if (s.charAt(i-1) != '0') curr += prev1;
        int two = Integer.parseInt(s.substring(i-2, i));
        if (two >= 10 && two <= 26) curr += prev2;
        prev2 = prev1;
        prev1 = curr;
    }
    return prev1;
}
```

**Python**
```python
def num_decodings(s: str) -> int:
    if not s or s[0] == '0':
        return 0
    prev2, prev1 = 1, 1
    for i in range(2, len(s) + 1):
        curr = 0
        if s[i-1] != '0':
            curr += prev1                 # single digit valid
        two = int(s[i-2:i])
        if 10 <= two <= 26:
            curr += prev2                 # two digits valid
        prev2, prev1 = prev1, curr
    return prev1
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

**Java**
```java
public int longestCommonSubsequence(String text1, String text2) {
    int m = text1.length(), n = text2.length();
    int[][] dp = new int[m + 1][n + 1];
    for (int i = 1; i <= m; i++)
        for (int j = 1; j <= n; j++)
            if (text1.charAt(i-1) == text2.charAt(j-1))
                dp[i][j] = dp[i-1][j-1] + 1;
            else
                dp[i][j] = Math.max(dp[i-1][j], dp[i][j-1]);
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

> **Why reverse?** In 0/1 knapsack each item can be used once. Iterating forward would let us use item `i` multiple times (that's unbounded knapsack). Reverse ensures `dp[w - weights[i]]` still reflects "without item `i`".

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

### Example: Word Break

**C++**
```cpp
bool wordBreak(string s, vector<string>& wordDict) {
    unordered_set<string> dict(wordDict.begin(), wordDict.end());
    int n = s.size();
    vector<bool> dp(n + 1, false);
    dp[0] = true;
    for (int i = 1; i <= n; i++)
        for (int j = 0; j < i; j++)
            if (dp[j] && dict.count(s.substr(j, i - j))) {
                dp[i] = true;
                break;
            }
    return dp[n];
}
```

**Java**
```java
public boolean wordBreak(String s, List<String> wordDict) {
    Set<String> dict = new HashSet<>(wordDict);
    boolean[] dp = new boolean[s.length() + 1];
    dp[0] = true;
    for (int i = 1; i <= s.length(); i++)
        for (int j = 0; j < i; j++)
            if (dp[j] && dict.contains(s.substring(j, i))) {
                dp[i] = true;
                break;
            }
    return dp[s.length()];
}
```

**Python**
```python
def word_break(s: str, word_dict: list[str]) -> bool:
    words = set(word_dict)
    n = len(s)
    dp = [False] * (n + 1)
    dp[0] = True
    for i in range(1, n + 1):
        for j in range(i):
            if dp[j] and s[j:i] in words:
                dp[i] = True
                break
    return dp[n]
```

---

## Pattern 4: Decision Making (Buy/Sell Stock)

**When to use:** Multiple transactions, with cooldown, with fees — state machine DP.

### Concept: State Machine DP

Define states and transitions. For stock problems:
```
           buy              sell
[Rest] ─────────→ [Hold] ─────────→ [Sold]
  ↑                                    │
  └───────── cooldown ─────────────────┘
```

Each state tracks the max profit at that point.

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

## Pattern 5: Longest Increasing Subsequence (LIS)

**When to use:** Longest increasing/decreasing subsequence, Russian doll envelopes, patience sorting.

### O(n²) DP Solution

**State:** `dp[i]` = length of LIS ending at index `i`.

**C++**
```cpp
int lengthOfLIS(vector<int>& nums) {
    int n = nums.size();
    vector<int> dp(n, 1);
    int result = 1;
    for (int i = 1; i < n; i++) {
        for (int j = 0; j < i; j++)
            if (nums[j] < nums[i])
                dp[i] = max(dp[i], dp[j] + 1);
        result = max(result, dp[i]);
    }
    return result;
}
```

**Java**
```java
public int lengthOfLIS(int[] nums) {
    int n = nums.length;
    int[] dp = new int[n];
    Arrays.fill(dp, 1);
    int result = 1;
    for (int i = 1; i < n; i++) {
        for (int j = 0; j < i; j++)
            if (nums[j] < nums[i])
                dp[i] = Math.max(dp[i], dp[j] + 1);
        result = Math.max(result, dp[i]);
    }
    return result;
}
```

**Python**
```python
def length_of_lis(nums: list[int]) -> int:
    n = len(nums)
    dp = [1] * n
    for i in range(1, n):
        for j in range(i):
            if nums[j] < nums[i]:
                dp[i] = max(dp[i], dp[j] + 1)
    return max(dp)
```

### O(n log n) with Binary Search (Patience Sorting)

**Intuition:** Maintain a "tails" array — `tails[i]` = smallest tail element for increasing subsequence of length `i+1`.

**C++**
```cpp
int lengthOfLIS(vector<int>& nums) {
    vector<int> tails;
    for (int num : nums) {
        auto it = lower_bound(tails.begin(), tails.end(), num);
        if (it == tails.end()) tails.push_back(num);
        else *it = num;
    }
    return tails.size();
}
```

**Java**
```java
public int lengthOfLIS(int[] nums) {
    List<Integer> tails = new ArrayList<>();
    for (int num : nums) {
        int pos = Collections.binarySearch(tails, num);
        if (pos < 0) pos = -(pos + 1);
        if (pos == tails.size()) tails.add(num);
        else tails.set(pos, num);
    }
    return tails.size();
}
```

**Python**
```python
import bisect

def length_of_lis_fast(nums: list[int]) -> int:
    tails = []
    for num in nums:
        pos = bisect.bisect_left(tails, num)
        if pos == len(tails):
            tails.append(num)
        else:
            tails[pos] = num
    return len(tails)
```

---

## Pattern 6: Palindromic DP

**When to use:** Longest palindromic substring/subsequence, minimum insertions to make palindrome, palindrome partitioning.

### Longest Palindromic Subsequence

**State:** `dp[i][j]` = length of longest palindromic subsequence in `s[i..j]`.

**Recurrence:** If `s[i] == s[j]`, then `dp[i][j] = dp[i+1][j-1] + 2`, else `dp[i][j] = max(dp[i+1][j], dp[i][j-1])`.

**Iteration order:** Dependencies are `dp[i+1][...]` and `dp[...][j-1]`, so `i` goes from bottom to top, `j` from left to right.

**C++**
```cpp
int longestPalindromeSubseq(string s) {
    int n = s.size();
    vector<vector<int>> dp(n, vector<int>(n, 0));
    for (int i = 0; i < n; i++) dp[i][i] = 1;

    for (int i = n - 2; i >= 0; i--) {         // bottom → top
        for (int j = i + 1; j < n; j++) {       // left → right
            if (s[i] == s[j])
                dp[i][j] = dp[i+1][j-1] + 2;
            else
                dp[i][j] = max(dp[i+1][j], dp[i][j-1]);
        }
    }
    return dp[0][n-1];
}
```

**Java**
```java
public int longestPalindromeSubseq(String s) {
    int n = s.length();
    int[][] dp = new int[n][n];
    for (int i = 0; i < n; i++) dp[i][i] = 1;

    for (int i = n - 2; i >= 0; i--)
        for (int j = i + 1; j < n; j++)
            if (s.charAt(i) == s.charAt(j))
                dp[i][j] = dp[i+1][j-1] + 2;
            else
                dp[i][j] = Math.max(dp[i+1][j], dp[i][j-1]);
    return dp[0][n-1];
}
```

**Python**
```python
def longest_palindrome_subseq(s: str) -> int:
    n = len(s)
    dp = [[0] * n for _ in range(n)]
    for i in range(n):
        dp[i][i] = 1

    for i in range(n - 2, -1, -1):       # bottom → top
        for j in range(i + 1, n):         # left → right
            if s[i] == s[j]:
                dp[i][j] = dp[i+1][j-1] + 2
            else:
                dp[i][j] = max(dp[i+1][j], dp[i][j-1])
    return dp[0][n-1]
```

### Longest Palindromic Substring (Expand Around Center)

Not strictly DP, but a critical interview pattern. O(n²) time, O(1) space.

**C++**
```cpp
string longestPalindrome(string s) {
    int start = 0, maxLen = 1;
    auto expand = [&](int l, int r) {
        while (l >= 0 && r < s.size() && s[l] == s[r]) {
            if (r - l + 1 > maxLen) {
                start = l;
                maxLen = r - l + 1;
            }
            l--; r++;
        }
    };
    for (int i = 0; i < s.size(); i++) {
        expand(i, i);       // odd
        expand(i, i + 1);   // even
    }
    return s.substr(start, maxLen);
}
```

**Java**
```java
int start = 0, maxLen = 1;
public String longestPalindrome(String s) {
    for (int i = 0; i < s.length(); i++) {
        expand(s, i, i);       // odd
        expand(s, i, i + 1);   // even
    }
    return s.substring(start, start + maxLen);
}
private void expand(String s, int l, int r) {
    while (l >= 0 && r < s.length() && s.charAt(l) == s.charAt(r)) {
        if (r - l + 1 > maxLen) {
            start = l;
            maxLen = r - l + 1;
        }
        l--; r++;
    }
}
```

**Python**
```python
def longest_palindrome(s: str) -> str:
    start, max_len = 0, 1
    def expand(l, r):
        nonlocal start, max_len
        while l >= 0 and r < len(s) and s[l] == s[r]:
            if r - l + 1 > max_len:
                start, max_len = l, r - l + 1
            l -= 1
            r += 1
    for i in range(len(s)):
        expand(i, i)       # odd-length
        expand(i, i + 1)   # even-length
    return s[start:start + max_len]
```

### Palindrome Partitioning — Minimum Cuts

**C++**
```cpp
int minCut(string s) {
    int n = s.size();
    vector<vector<bool>> isPal(n, vector<bool>(n, false));
    for (int i = n - 1; i >= 0; i--)
        for (int j = i; j < n; j++)
            isPal[i][j] = (s[i] == s[j]) && (j - i < 3 || isPal[i+1][j-1]);

    vector<int> dp(n);
    for (int i = 0; i < n; i++) {
        if (isPal[0][i]) { dp[i] = 0; continue; }
        dp[i] = i;
        for (int j = 1; j <= i; j++)
            if (isPal[j][i])
                dp[i] = min(dp[i], dp[j-1] + 1);
    }
    return dp[n-1];
}
```

**Java**
```java
public int minCut(String s) {
    int n = s.length();
    boolean[][] isPal = new boolean[n][n];
    for (int i = n - 1; i >= 0; i--)
        for (int j = i; j < n; j++)
            isPal[i][j] = (s.charAt(i) == s.charAt(j)) && (j - i < 3 || isPal[i+1][j-1]);

    int[] dp = new int[n];
    for (int i = 0; i < n; i++) {
        if (isPal[0][i]) { dp[i] = 0; continue; }
        dp[i] = i;
        for (int j = 1; j <= i; j++)
            if (isPal[j][i])
                dp[i] = Math.min(dp[i], dp[j-1] + 1);
    }
    return dp[n-1];
}
```

**Python**
```python
def min_cut(s: str) -> int:
    n = len(s)
    # is_pal[i][j] = True if s[i..j] is palindrome
    is_pal = [[False] * n for _ in range(n)]
    for i in range(n - 1, -1, -1):
        for j in range(i, n):
            is_pal[i][j] = (s[i] == s[j]) and (j - i < 3 or is_pal[i+1][j-1])

    dp = [0] * n  # dp[i] = min cuts for s[0..i]
    for i in range(n):
        if is_pal[0][i]:
            dp[i] = 0
        else:
            dp[i] = i  # worst case: cut every character
            for j in range(1, i + 1):
                if is_pal[j][i]:
                    dp[i] = min(dp[i], dp[j-1] + 1)
    return dp[n-1]
```

---

## Pattern 7: Interval DP

**When to use:** Merging intervals, matrix chain multiplication, burst balloons, stone game — problems where you combine adjacent ranges.

**Key insight:** `dp[i][j]` represents the optimal answer for the subarray `[i..j]`. You try every possible split point `k` in between.

**Template:**
```python
for length in range(2, n + 1):       # subproblem size
    for i in range(n - length + 1):   # start index
        j = i + length - 1            # end index
        for k in range(i, j):         # split point
            dp[i][j] = optimize(dp[i][j], dp[i][k] + dp[k+1][j] + cost(i, j, k))
```

### Example: Burst Balloons

**C++**
```cpp
int maxCoins(vector<int>& nums) {
    int n = nums.size();
    nums.insert(nums.begin(), 1);
    nums.push_back(1);
    vector<vector<int>> dp(n + 2, vector<int>(n + 2, 0));

    for (int len = 1; len <= n; len++) {
        for (int i = 1; i <= n - len + 1; i++) {
            int j = i + len - 1;
            for (int k = i; k <= j; k++) {
                dp[i][j] = max(dp[i][j],
                    dp[i][k-1] + dp[k+1][j] +
                    nums[i-1] * nums[k] * nums[j+1]);
            }
        }
    }
    return dp[1][n];
}
```

**Java**
```java
public int maxCoins(int[] nums) {
    int n = nums.length;
    int[] arr = new int[n + 2];
    arr[0] = arr[n + 1] = 1;
    System.arraycopy(nums, 0, arr, 1, n);
    int[][] dp = new int[n + 2][n + 2];

    for (int len = 1; len <= n; len++)
        for (int i = 1; i <= n - len + 1; i++) {
            int j = i + len - 1;
            for (int k = i; k <= j; k++)
                dp[i][j] = Math.max(dp[i][j],
                    dp[i][k-1] + dp[k+1][j] +
                    arr[i-1] * arr[k] * arr[j+1]);
        }
    return dp[1][n];
}
```

**Python**
```python
def max_coins(nums: list[int]) -> int:
    nums = [1] + nums + [1]
    n = len(nums)
    dp = [[0] * n for _ in range(n)]

    for length in range(1, n - 1):              # subproblem size
        for i in range(1, n - length):          # start
            j = i + length - 1                  # end
            for k in range(i, j + 1):           # last balloon to burst
                dp[i][j] = max(dp[i][j],
                    dp[i][k-1] + dp[k+1][j] +
                    nums[i-1] * nums[k] * nums[j+1])
    return dp[1][n-2]
```

---

## Pattern 8: Bitmask DP

**When to use:** When n ≤ 20 and you need to track which elements have been used/visited. Traveling salesman, assignment problems, partition into groups.

**State:** `dp[mask]` or `dp[mask][i]` where `mask` is a bitmask of visited elements.

**Bit operations:**
```
Check if bit i is set:   mask & (1 << i)
Set bit i:               mask | (1 << i)
Clear bit i:             mask & ~(1 << i)
Count set bits:          bin(mask).count('1')  /  __builtin_popcount(mask)
```

### Example: Partition to K Equal Sum Subsets

**C++**
```cpp
bool canPartitionKSubsets(vector<int>& nums, int k) {
    int total = accumulate(nums.begin(), nums.end(), 0);
    if (total % k != 0) return false;
    int target = total / k, n = nums.size();
    vector<bool> dp(1 << n, false);
    vector<int> subsetSum(1 << n, 0);
    dp[0] = true;
    for (int mask = 0; mask < (1 << n); mask++) {
        if (!dp[mask]) continue;
        for (int i = 0; i < n; i++) {
            if (mask & (1 << i)) continue;
            int newMask = mask | (1 << i);
            if (!dp[newMask] && subsetSum[mask] % target + nums[i] <= target) {
                dp[newMask] = true;
                subsetSum[newMask] = subsetSum[mask] + nums[i];
            }
        }
    }
    return dp[(1 << n) - 1];
}
```

**Java**
```java
public boolean canPartitionKSubsets(int[] nums, int k) {
    int total = Arrays.stream(nums).sum();
    if (total % k != 0) return false;
    int target = total / k, n = nums.length;
    boolean[] dp = new boolean[1 << n];
    int[] subsetSum = new int[1 << n];
    dp[0] = true;
    for (int mask = 0; mask < (1 << n); mask++) {
        if (!dp[mask]) continue;
        for (int i = 0; i < n; i++) {
            if ((mask & (1 << i)) != 0) continue;
            int newMask = mask | (1 << i);
            if (!dp[newMask] && subsetSum[mask] % target + nums[i] <= target) {
                dp[newMask] = true;
                subsetSum[newMask] = subsetSum[mask] + nums[i];
            }
        }
    }
    return dp[(1 << n) - 1];
}
```

**Python**
```python
def can_partition_k(nums: list[int], k: int) -> bool:
    total = sum(nums)
    if total % k != 0:
        return False
    target = total // k
    n = len(nums)
    dp = [False] * (1 << n)
    subset_sum = [0] * (1 << n)
    dp[0] = True

    for mask in range(1 << n):
        if not dp[mask]:
            continue
        for i in range(n):
            if mask & (1 << i):
                continue    # already used
            new_mask = mask | (1 << i)
            if not dp[new_mask] and subset_sum[mask] % target + nums[i] <= target:
                dp[new_mask] = True
                subset_sum[new_mask] = subset_sum[mask] + nums[i]
    return dp[(1 << n) - 1]
```

### Example: Shortest Path Visiting All Nodes (Traveling Salesman Variant)

**Python**
```python
def shortest_path_all_nodes(graph: list[list[int]]) -> int:
    """BFS + bitmask: shortest path that visits all nodes"""
    n = len(graph)
    full_mask = (1 << n) - 1
    queue = deque()
    visited = set()

    for i in range(n):
        state = (1 << i, i)
        queue.append((state[0], state[1], 0))
        visited.add(state)

    while queue:
        mask, node, dist = queue.popleft()
        if mask == full_mask:
            return dist
        for neighbor in graph[node]:
            new_mask = mask | (1 << neighbor)
            if (new_mask, neighbor) not in visited:
                visited.add((new_mask, neighbor))
                queue.append((new_mask, neighbor, dist + 1))
    return -1
```

---

## Pattern 9: Partition / Subset Sum DP

**When to use:** "Can you partition into two equal subsets?", "Number of subsets that sum to target", "Target sum with +/−".

### Partition Equal Subset Sum

**C++**
```cpp
bool canPartition(vector<int>& nums) {
    int total = accumulate(nums.begin(), nums.end(), 0);
    if (total % 2 != 0) return false;
    int target = total / 2;
    vector<bool> dp(target + 1, false);
    dp[0] = true;
    for (int num : nums)
        for (int j = target; j >= num; j--)
            dp[j] = dp[j] || dp[j - num];
    return dp[target];
}
```

**Java**
```java
public boolean canPartition(int[] nums) {
    int total = Arrays.stream(nums).sum();
    if (total % 2 != 0) return false;
    int target = total / 2;
    boolean[] dp = new boolean[target + 1];
    dp[0] = true;
    for (int num : nums)
        for (int j = target; j >= num; j--)
            dp[j] = dp[j] || dp[j - num];
    return dp[target];
}
```

**Python**
```python
def can_partition(nums: list[int]) -> bool:
    total = sum(nums)
    if total % 2 != 0:
        return False
    target = total // 2
    dp = [False] * (target + 1)
    dp[0] = True
    for num in nums:
        for j in range(target, num - 1, -1):  # reverse for 0/1
            dp[j] = dp[j] or dp[j - num]
    return dp[target]
```

### Target Sum (Count Assignments of +/−)

**C++**
```cpp
int findTargetSumWays(vector<int>& nums, int target) {
    int total = accumulate(nums.begin(), nums.end(), 0);
    if ((total + target) % 2 != 0 || abs(target) > total) return 0;
    int newTarget = (total + target) / 2;
    vector<int> dp(newTarget + 1, 0);
    dp[0] = 1;
    for (int num : nums)
        for (int j = newTarget; j >= num; j--)
            dp[j] += dp[j - num];
    return dp[newTarget];
}
```

**Java**
```java
public int findTargetSumWays(int[] nums, int target) {
    int total = Arrays.stream(nums).sum();
    if ((total + target) % 2 != 0 || Math.abs(target) > total) return 0;
    int newTarget = (total + target) / 2;
    int[] dp = new int[newTarget + 1];
    dp[0] = 1;
    for (int num : nums)
        for (int j = newTarget; j >= num; j--)
            dp[j] += dp[j - num];
    return dp[newTarget];
}
```

**Python**
```python
def find_target_sum_ways(nums: list[int], target: int) -> int:
    """
    Transform: sum(P) - sum(N) = target, sum(P) + sum(N) = total
    → sum(P) = (target + total) / 2 → subset sum count
    """
    total = sum(nums)
    if (total + target) % 2 != 0 or abs(target) > total:
        return 0
    new_target = (total + target) // 2

    dp = [0] * (new_target + 1)
    dp[0] = 1
    for num in nums:
        for j in range(new_target, num - 1, -1):
            dp[j] += dp[j - num]
    return dp[new_target]
```

---

## Pattern 10: DP on Trees

**When to use:** Maximum path sum, house robber on tree, tree diameter, re-rooting.

### House Robber III (Binary Tree)

**C++**
```cpp
pair<int,int> dfs(TreeNode* root) {
    // returns {rob_this, skip_this}
    if (!root) return {0, 0};
    auto [leftRob, leftSkip] = dfs(root->left);
    auto [rightRob, rightSkip] = dfs(root->right);
    int rob = root->val + leftSkip + rightSkip;
    int skip = max(leftRob, leftSkip) + max(rightRob, rightSkip);
    return {rob, skip};
}
int rob(TreeNode* root) {
    auto [r, s] = dfs(root);
    return max(r, s);
}
```

**Java**
```java
public int rob(TreeNode root) {
    int[] res = dfs(root);
    return Math.max(res[0], res[1]);
}
private int[] dfs(TreeNode root) {
    if (root == null) return new int[]{0, 0};
    int[] left = dfs(root.left), right = dfs(root.right);
    int rob = root.val + left[1] + right[1];
    int skip = Math.max(left[0], left[1]) + Math.max(right[0], right[1]);
    return new int[]{rob, skip};
}
```

**Python**
```python
def rob_tree(root) -> int:
    def dfs(node):
        """Returns (rob_this, skip_this)"""
        if not node:
            return (0, 0)
        left = dfs(node.left)
        right = dfs(node.right)
        rob = node.val + left[1] + right[1]
        skip = max(left) + max(right)
        return (rob, skip)
    return max(dfs(root))
```

---

## Templates: Top-Down & Bottom-Up

### Top-Down Template

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

**C++ (with unordered_map)**
```cpp
unordered_map<int, int> memo;
int dp(int state) {
    if (base_condition(state)) return base_value;
    if (memo.count(state)) return memo[state];

    int result = initial_value;
    for (auto& choice : choices(state))
        result = optimize(result, dp(next_state(state, choice)));
    return memo[state] = result;
}
```

### Bottom-Up Template

```python
def solve(args):
    # 1. Create table
    dp = [base_value] * (n + 1)

    # 2. Set base cases
    dp[0] = ...

    # 3. Fill table (iterate so dependencies come first)
    for i in range(1, n + 1):
        for choice in choices(i):
            dp[i] = optimize(dp[i], dp[prev_state] + cost)

    # 4. Return answer
    return dp[n]
```

### Converting Top-Down → Bottom-Up Checklist

| Step | Action |
|------|--------|
| 1 | Identify recursive parameters → table dimensions |
| 2 | Find base cases → initial table values |
| 3 | Determine dependency direction → iteration order |
| 4 | Replace recursive calls with table lookups |
| 5 | Answer = what the initial recursive call returns |

---

## Pitfalls & Edge Cases

| Pitfall | How to Handle |
|---------|---------------|
| Wrong base case | Trace smallest inputs manually |
| Wrong iteration order | Bottom-up: ensure dependencies computed first |
| Off-by-one in table size | Usually `n+1` for 1D, `(m+1) x (n+1)` for 2D |
| Not considering "skip" option | Knapsack, LIS — always include "don't take" branch |
| Stack overflow (top-down) | Switch to bottom-up or increase recursion limit |
| Space optimization direction | 0/1 knapsack: iterate capacity in **reverse** |
| Using unbounded when 0/1 needed | Forward iteration = unbounded, reverse = 0/1 |
| Forgetting to handle impossible states | Use `INF`/`-INF` as sentinel, check before returning |
| Wrong answer location | Sometimes `max(dp[...])` not `dp[n]` (e.g., LIS) |

---

## Decision Guide: Which Pattern?

```
Start
  │
  ├─ Array/sequence? ──→ Is subsequence? ──→ LIS pattern
  │                   └─ Adjacent choices? ──→ 1D DP (House Robber)
  │                   └─ Sum/capacity? ──→ Knapsack / Subset Sum
  │
  ├─ Two strings? ──→ Edit distance / LCS pattern (2D String DP)
  │
  ├─ Grid? ──→ 2D DP (Unique Paths, Min Path Sum)
  │
  ├─ Merging ranges? ──→ Interval DP (Burst Balloons)
  │
  ├─ n ≤ 20 items? ──→ Bitmask DP
  │
  ├─ Tree structure? ──→ Tree DP (DFS returning tuples)
  │
  ├─ Palindrome? ──→ Palindromic DP (expand or `dp[i][j]`)
  │
  └─ Multiple states per step? ──→ State Machine DP (Stock)
```

---

## Practice Problems

| # | Problem | Difficulty | Pattern | Link |
|---|---------|-----------|---------|------|
| 1 | Climbing Stairs | Easy | 1D | [LC 70](https://leetcode.com/problems/climbing-stairs/) |
| 2 | House Robber | Medium | 1D | [LC 198](https://leetcode.com/problems/house-robber/) |
| 3 | House Robber II (circular) | Medium | 1D | [LC 213](https://leetcode.com/problems/house-robber-ii/) |
| 4 | Coin Change | Medium | 1D | [LC 322](https://leetcode.com/problems/coin-change/) |
| 5 | Decode Ways | Medium | 1D | [LC 91](https://leetcode.com/problems/decode-ways/) |
| 6 | Longest Increasing Subsequence | Medium | LIS | [LC 300](https://leetcode.com/problems/longest-increasing-subsequence/) |
| 7 | Unique Paths | Medium | 2D Grid | [LC 62](https://leetcode.com/problems/unique-paths/) |
| 8 | Minimum Path Sum | Medium | 2D Grid | [LC 64](https://leetcode.com/problems/minimum-path-sum/) |
| 9 | Longest Common Subsequence | Medium | 2D String | [LC 1143](https://leetcode.com/problems/longest-common-subsequence/) |
| 10 | Edit Distance | Medium | 2D String | [LC 72](https://leetcode.com/problems/edit-distance/) |
| 11 | Word Break | Medium | 1D + Set | [LC 139](https://leetcode.com/problems/word-break/) |
| 12 | Partition Equal Subset Sum | Medium | Subset Sum | [LC 416](https://leetcode.com/problems/partition-equal-subset-sum/) |
| 13 | Target Sum | Medium | Subset Sum | [LC 494](https://leetcode.com/problems/target-sum/) |
| 14 | 0/1 Knapsack | Medium | 2D → 1D | Classic |
| 15 | Longest Palindromic Subsequence | Medium | Palindromic | [LC 516](https://leetcode.com/problems/longest-palindromic-subsequence/) |
| 16 | Longest Palindromic Substring | Medium | Expand | [LC 5](https://leetcode.com/problems/longest-palindromic-substring/) |
| 17 | Palindrome Partitioning II | Hard | Palindromic | [LC 132](https://leetcode.com/problems/palindrome-partitioning-ii/) |
| 18 | Burst Balloons | Hard | Interval | [LC 312](https://leetcode.com/problems/burst-balloons/) |
| 19 | Best Time Buy/Sell Stock with Cooldown | Medium | State Machine | [LC 309](https://leetcode.com/problems/best-time-to-buy-and-sell-stock-with-cooldown/) |
| 20 | House Robber III | Medium | Tree DP | [LC 337](https://leetcode.com/problems/house-robber-iii/) |
| 21 | Shortest Path Visiting All Nodes | Hard | Bitmask | [LC 847](https://leetcode.com/problems/shortest-path-visiting-all-nodes/) |
| 22 | Matrix Chain Multiplication | Hard | Interval | Classic |
| 23 | Regular Expression Matching | Hard | 2D String | [LC 10](https://leetcode.com/problems/regular-expression-matching/) |
| 24 | Maximal Square | Medium | 2D Grid | [LC 221](https://leetcode.com/problems/maximal-square/) |
| 25 | Interleaving String | Medium | 2D String | [LC 97](https://leetcode.com/problems/interleaving-string/) |

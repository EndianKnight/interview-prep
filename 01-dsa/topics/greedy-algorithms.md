# Greedy Algorithms

Greedy algorithms make the locally optimal choice at each step, hoping to find the global optimum. They work when the problem has the **greedy-choice property** and **optimal substructure**.

---

## When Greedy Works

- **Greedy-choice property** — a locally optimal choice leads to a globally optimal solution
- **Optimal substructure** — optimal solution contains optimal solutions to subproblems
- Often provable by exchange argument or induction
- If unsure, try DP first — greedy is a special case of DP

---

## Pattern 1: Interval Scheduling

**When to use:** Maximum non-overlapping intervals, minimum intervals to remove, meeting rooms.

### Example: Non-Overlapping Intervals (Maximum Activities)

**C++**
```cpp
int eraseOverlapIntervals(vector<vector<int>>& intervals) {
    sort(intervals.begin(), intervals.end(),
         [](auto& a, auto& b) { return a[1] < b[1]; });

    int count = 0, prevEnd = INT_MIN;
    for (auto& iv : intervals) {
        if (iv[0] >= prevEnd) {
            prevEnd = iv[1];  // keep this interval
        } else {
            count++;  // remove this interval
        }
    }
    return count;
}
```

**Python**
```python
def erase_overlap_intervals(intervals: list[list[int]]) -> int:
    intervals.sort(key=lambda x: x[1])  # sort by end time
    count, prev_end = 0, float('-inf')
    for start, end in intervals:
        if start >= prev_end:
            prev_end = end
        else:
            count += 1
    return count
```

**Java**
```java
public int eraseOverlapIntervals(int[][] intervals) {
    Arrays.sort(intervals, (a, b) -> Integer.compare(a[1], b[1]));
    int count = 0, prevEnd = Integer.MIN_VALUE;
    for (int[] iv : intervals) {
        if (iv[0] >= prevEnd) {
            prevEnd = iv[1];
        } else {
            count++;
        }
    }
    return count;
}
```

---

## Pattern 2: Greedy Scheduling / Assignment

**When to use:** Task assignment, job scheduling, gas station.

### Example: Jump Game

**Python**
```python
def can_jump(nums: list[int]) -> bool:
    max_reach = 0
    for i, jump in enumerate(nums):
        if i > max_reach:
            return False
        max_reach = max(max_reach, i + jump)
    return True

def jump_game_ii(nums: list[int]) -> int:
    """Minimum jumps to reach end"""
    jumps = current_end = farthest = 0
    for i in range(len(nums) - 1):
        farthest = max(farthest, i + nums[i])
        if i == current_end:
            jumps += 1
            current_end = farthest
    return jumps
```

**C++**
```cpp
bool canJump(vector<int>& nums) {
    int maxReach = 0;
    for (int i = 0; i < nums.size(); i++) {
        if (i > maxReach) return false;
        maxReach = max(maxReach, i + nums[i]);
    }
    return true;
}

int jump(vector<int>& nums) {
    int jumps = 0, currentEnd = 0, farthest = 0;
    for (int i = 0; i < nums.size() - 1; i++) {
        farthest = max(farthest, i + nums[i]);
        if (i == currentEnd) { jumps++; currentEnd = farthest; }
    }
    return jumps;
}
```

**Java**
```java
public boolean canJump(int[] nums) {
    int maxReach = 0;
    for (int i = 0; i < nums.length; i++) {
        if (i > maxReach) return false;
        maxReach = Math.max(maxReach, i + nums[i]);
    }
    return true;
}

public int jump(int[] nums) {
    int jumps = 0, currentEnd = 0, farthest = 0;
    for (int i = 0; i < nums.length - 1; i++) {
        farthest = Math.max(farthest, i + nums[i]);
        if (i == currentEnd) { jumps++; currentEnd = farthest; }
    }
    return jumps;
}
```

---

## Pattern 3: Greedy + Sorting

**When to use:** Coin change (canonical systems), fractional knapsack, minimize cost.

### Example: Minimum Number of Platforms

**Python**
```python
def min_platforms(arrivals: list[int], departures: list[int]) -> int:
    arrivals.sort()
    departures.sort()
    platforms = max_platforms = 0
    i = j = 0
    while i < len(arrivals):
        if arrivals[i] <= departures[j]:
            platforms += 1
            max_platforms = max(max_platforms, platforms)
            i += 1
        else:
            platforms -= 1
            j += 1
    return max_platforms
```

### Example: Gas Station

**C++**
```cpp
int canCompleteCircuit(vector<int>& gas, vector<int>& cost) {
    int totalTank = 0, currTank = 0, start = 0;
    for (int i = 0; i < gas.size(); i++) {
        totalTank += gas[i] - cost[i];
        currTank += gas[i] - cost[i];
        if (currTank < 0) { start = i + 1; currTank = 0; }
    }
    return totalTank >= 0 ? start : -1;
}
```

**Java**
```java
public int canCompleteCircuit(int[] gas, int[] cost) {
    int totalTank = 0, currTank = 0, start = 0;
    for (int i = 0; i < gas.length; i++) {
        totalTank += gas[i] - cost[i];
        currTank += gas[i] - cost[i];
        if (currTank < 0) { start = i + 1; currTank = 0; }
    }
    return totalTank >= 0 ? start : -1;
}
```

**Python**
```python
def can_complete_circuit(gas: list[int], cost: list[int]) -> int:
    if sum(gas) < sum(cost):
        return -1

    start = tank = 0
    for i in range(len(gas)):
        tank += gas[i] - cost[i]
        if tank < 0:
            start = i + 1
            tank = 0
    return start
```

---

## Pattern 4: Greedy String Construction

**When to use:** Partition labels, reorganize string, remove K digits.

### Example: Partition Labels

**Python**
```python
def partition_labels(s: str) -> list[int]:
    last = {c: i for i, c in enumerate(s)}
    start = end = 0
    result = []
    for i, c in enumerate(s):
        end = max(end, last[c])
        if i == end:
            result.append(end - start + 1)
            start = end + 1
    return result
```

**Java**
```java
public List<Integer> partitionLabels(String s) {
    int[] last = new int[26];
    for (int i = 0; i < s.length(); i++) last[s.charAt(i) - 'a'] = i;

    List<Integer> result = new ArrayList<>();
    int start = 0, end = 0;
    for (int i = 0; i < s.length(); i++) {
        end = Math.max(end, last[s.charAt(i) - 'a']);
        if (i == end) {
            result.add(end - start + 1);
            start = end + 1;
        }
    }
    return result;
}
```

**C++**
```cpp
vector<int> partitionLabels(string s) {
    int last[26] = {};
    for (int i = 0; i < s.size(); i++) last[s[i] - 'a'] = i;

    vector<int> result;
    int start = 0, end = 0;
    for (int i = 0; i < s.size(); i++) {
        end = max(end, last[s[i] - 'a']);
        if (i == end) {
            result.push_back(end - start + 1);
            start = end + 1;
        }
    }
    return result;
}
```

---

## Pitfalls & Edge Cases

| Pitfall | How to Handle |
|---------|--------------|
| Greedy doesn't always work | Verify greedy-choice property; consider DP |
| Wrong sorting criterion | Interval: sort by end for max non-overlapping |
| Off-by-one with indices | Trace through small example |
| Negative numbers | May invalidate greedy choices |

---

## Practice Problems

| # | Problem | Difficulty | Pattern | Link |
|---|---------|-----------|---------|------|
| 1 | Jump Game | Medium | Greedy reach | [LeetCode 55](https://leetcode.com/problems/jump-game/) |
| 2 | Jump Game II | Medium | Greedy BFS | [LeetCode 45](https://leetcode.com/problems/jump-game-ii/) |
| 3 | Non-Overlapping Intervals | Medium | Interval | [LeetCode 435](https://leetcode.com/problems/non-overlapping-intervals/) |
| 4 | Gas Station | Medium | Greedy | [LeetCode 134](https://leetcode.com/problems/gas-station/) |
| 5 | Partition Labels | Medium | String | [LeetCode 763](https://leetcode.com/problems/partition-labels/) |
| 6 | Task Scheduler | Medium | Scheduling | [LeetCode 621](https://leetcode.com/problems/task-scheduler/) |
| 7 | Candy | Hard | Two-pass | [LeetCode 135](https://leetcode.com/problems/candy/) |

# Sliding Window & Two Pointers

Dedicated deep-dive — these patterns deserve their own reference since they appear in ~20% of interviews.

---

## Sliding Window Templates

### Fixed-Size Window

```python
def fixed_window(arr, k):
    # Initialize window with first k elements
    window_sum = sum(arr[:k])
    best = window_sum

    for i in range(k, len(arr)):
        window_sum += arr[i] - arr[i - k]  # slide
        best = max(best, window_sum)
    return best
```

### Variable-Size Window (Find Minimum)

```python
def variable_window_min(arr, condition):
    left = 0
    best = float('inf')
    # window state variables

    for right in range(len(arr)):
        # expand: add arr[right] to window

        while condition_met():
            best = min(best, right - left + 1)
            # shrink: remove arr[left] from window
            left += 1
    return best
```

### Variable-Size Window (Find Maximum)

```python
def variable_window_max(arr, condition):
    left = 0
    best = 0
    # window state variables

    for right in range(len(arr)):
        # expand: add arr[right] to window

        while condition_violated():
            # shrink: remove arr[left] from window
            left += 1
        best = max(best, right - left + 1)
    return best
```

---

## Example: Minimum Window Substring

**C++**
```cpp
string minWindow(string s, string t) {
    unordered_map<char, int> need, window;
    for (char c : t) need[c]++;

    int left = 0, valid = 0, start = 0, minLen = INT_MAX;

    for (int right = 0; right < s.size(); right++) {
        char c = s[right];
        if (need.count(c)) {
            window[c]++;
            if (window[c] == need[c]) valid++;
        }

        while (valid == need.size()) {
            if (right - left + 1 < minLen) {
                start = left;
                minLen = right - left + 1;
            }
            char d = s[left++];
            if (need.count(d)) {
                if (window[d] == need[d]) valid--;
                window[d]--;
            }
        }
    }
    return minLen == INT_MAX ? "" : s.substr(start, minLen);
}
```

**Python**
```python
from collections import Counter, defaultdict

def min_window(s: str, t: str) -> str:
    need = Counter(t)
    window = defaultdict(int)
    left = valid = 0
    start, min_len = 0, float('inf')

    for right, c in enumerate(s):
        if c in need:
            window[c] += 1
            if window[c] == need[c]:
                valid += 1

        while valid == len(need):
            if right - left + 1 < min_len:
                start = left
                min_len = right - left + 1
            d = s[left]
            left += 1
            if d in need:
                if window[d] == need[d]:
                    valid -= 1
                window[d] -= 1

    return "" if min_len == float('inf') else s[start:start + min_len]
```

---

## Two Pointer Patterns

### Converging Pointers (L→ ←R)

**Example: Container With Most Water**

```python
def max_area(height: list[int]) -> int:
    left, right = 0, len(height) - 1
    best = 0
    while left < right:
        area = min(height[left], height[right]) * (right - left)
        best = max(best, area)
        if height[left] < height[right]:
            left += 1
        else:
            right -= 1
    return best
```

### Same-Direction Pointers (→ →)

**Example: Remove Duplicates from Sorted Array**

```python
def remove_duplicates(nums: list[int]) -> int:
    if not nums:
        return 0
    write = 1
    for read in range(1, len(nums)):
        if nums[read] != nums[read - 1]:
            nums[write] = nums[read]
            write += 1
    return write
```

### Three Pointers (3Sum)

```python
def three_sum(nums: list[int]) -> list[list[int]]:
    nums.sort()
    result = []
    for i in range(len(nums) - 2):
        if i > 0 and nums[i] == nums[i-1]:
            continue
        left, right = i + 1, len(nums) - 1
        while left < right:
            total = nums[i] + nums[left] + nums[right]
            if total == 0:
                result.append([nums[i], nums[left], nums[right]])
                while left < right and nums[left] == nums[left+1]: left += 1
                while left < right and nums[right] == nums[right-1]: right -= 1
                left += 1
                right -= 1
            elif total < 0:
                left += 1
            else:
                right -= 1
    return result
```

---

## Practice Problems

| # | Problem | Difficulty | Pattern | Link |
|---|---------|-----------|---------|------|
| 1 | Maximum Average Subarray | Easy | Fixed Window | [LeetCode 643](https://leetcode.com/problems/maximum-average-subarray-i/) |
| 2 | Longest Substring Without Repeating | Medium | Variable Window | [LeetCode 3](https://leetcode.com/problems/longest-substring-without-repeating-characters/) |
| 3 | Permutation in String | Medium | Fixed Window + Freq | [LeetCode 567](https://leetcode.com/problems/permutation-in-string/) |
| 4 | Minimum Window Substring | Hard | Variable Window | [LeetCode 76](https://leetcode.com/problems/minimum-window-substring/) |
| 5 | 3Sum | Medium | Three Pointers | [LeetCode 15](https://leetcode.com/problems/3sum/) |
| 6 | Container With Most Water | Medium | Converging | [LeetCode 11](https://leetcode.com/problems/container-with-most-water/) |
| 7 | Trapping Rain Water | Hard | Converging | [LeetCode 42](https://leetcode.com/problems/trapping-rain-water/) |
| 8 | Sliding Window Maximum | Hard | Monotonic Deque | [LeetCode 239](https://leetcode.com/problems/sliding-window-maximum/) |

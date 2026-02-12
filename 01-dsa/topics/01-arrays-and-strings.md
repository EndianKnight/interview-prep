# Arrays & Strings

The most frequently tested topic in coding interviews. Mastering array/string patterns covers ~30% of all interview questions.

---

## Core Concepts

### Arrays
- **Contiguous memory** — elements stored sequentially, enabling O(1) random access
- **Fixed vs dynamic** — static arrays have fixed size; dynamic arrays (`vector`, `ArrayList`, `list`) resize automatically (amortized O(1) append)
- **Cache-friendly** — sequential access is fast due to spatial locality

### Strings
- **Immutable in Java/Python** — every modification creates a new string (O(n) per operation)
- **Mutable in C++** — `std::string` allows in-place modification
- **Character encoding** — ASCII (128 chars), Extended ASCII (256), Unicode (UTF-8/16/32)

---

## Complexity Quick Reference

| Operation | Array | Dynamic Array | String (immutable) |
|-----------|-------|--------------|-------------------|
| Access by index | O(1) | O(1) | O(1) |
| Search (unsorted) | O(n) | O(n) | O(n) |
| Insert at end | — | O(1)* | O(n) |
| Insert at index | O(n) | O(n) | O(n) |
| Delete at index | O(n) | O(n) | O(n) |
| Concatenation | — | — | O(n+m) |

\* amortized

---

## Pattern 1: Two Pointers

**When to use:** Sorted arrays, pair/triplet finding, partitioning, removing duplicates.

### Technique
Use two pointers (usually `left` and `right`) moving towards each other or in the same direction to reduce nested loops from O(n²) to O(n).

### Example: Two Sum (Sorted Array)

Given a sorted array, find two numbers that add up to a target.

**C++**
```cpp
#include <vector>
using namespace std;

vector<int> twoSum(vector<int>& nums, int target) {
    int left = 0, right = nums.size() - 1;
    while (left < right) {
        int sum = nums[left] + nums[right];
        if (sum == target) {
            return {left, right};
        } else if (sum < target) {
            left++;
        } else {
            right--;
        }
    }
    return {}; // no solution
}
```

**Java**
```java
public int[] twoSum(int[] nums, int target) {
    int left = 0, right = nums.length - 1;
    while (left < right) {
        int sum = nums[left] + nums[right];
        if (sum == target) {
            return new int[]{left, right};
        } else if (sum < target) {
            left++;
        } else {
            right--;
        }
    }
    return new int[]{}; // no solution
}
```

**Python**
```python
def two_sum(nums: list[int], target: int) -> list[int]:
    left, right = 0, len(nums) - 1
    while left < right:
        total = nums[left] + nums[right]
        if total == target:
            return [left, right]
        elif total < target:
            left += 1
        else:
            right -= 1
    return []  # no solution
```

**Complexity:** Time O(n), Space O(1)

### Other Two Pointer Problems
- Remove duplicates from sorted array
- Container with most water
- 3Sum / 4Sum
- Trapping rain water
- Sort colors (Dutch National Flag)

---

## Pattern 2: Sliding Window

**When to use:** Subarray/substring problems, finding optimal contiguous sequences, "at most K" constraints.

### Technique
Maintain a window `[left, right]` that expands and shrinks. Track window state (sum, count, frequency map) incrementally.

### Template (Variable-Size Window)

```
left = 0
for right in range(n):
    # expand: add nums[right] to window state
    while window_is_invalid():
        # shrink: remove nums[left] from window state
        left += 1
    # update answer
```

### Example: Longest Substring Without Repeating Characters

**C++**
```cpp
#include <string>
#include <unordered_set>
using namespace std;

int lengthOfLongestSubstring(string s) {
    unordered_set<char> window;
    int left = 0, maxLen = 0;

    for (int right = 0; right < s.size(); right++) {
        while (window.count(s[right])) {
            window.erase(s[left]);
            left++;
        }
        window.insert(s[right]);
        maxLen = max(maxLen, right - left + 1);
    }
    return maxLen;
}
```

**Java**
```java
import java.util.HashSet;
import java.util.Set;

public int lengthOfLongestSubstring(String s) {
    Set<Character> window = new HashSet<>();
    int left = 0, maxLen = 0;

    for (int right = 0; right < s.length(); right++) {
        while (window.contains(s.charAt(right))) {
            window.remove(s.charAt(left));
            left++;
        }
        window.add(s.charAt(right));
        maxLen = Math.max(maxLen, right - left + 1);
    }
    return maxLen;
}
```

**Python**
```python
def length_of_longest_substring(s: str) -> int:
    window = set()
    left = max_len = 0

    for right in range(len(s)):
        while s[right] in window:
            window.remove(s[left])
            left += 1
        window.add(s[right])
        max_len = max(max_len, right - left + 1)
    return max_len
```

**Complexity:** Time O(n), Space O(min(n, charset_size))

### Other Sliding Window Problems
- Minimum window substring
- Maximum sum subarray of size K (fixed window)
- Longest substring with at most K distinct characters
- Fruit into baskets
- Permutation in string

---

## Pattern 3: Prefix Sum

**When to use:** Range sum queries, subarray sums equal to K, equilibrium index.

### Technique
Precompute cumulative sums so any subarray sum can be computed in O(1): `sum(i, j) = prefix[j+1] - prefix[i]`.

### Example: Subarray Sum Equals K

**C++**
```cpp
#include <vector>
#include <unordered_map>
using namespace std;

int subarraySum(vector<int>& nums, int k) {
    unordered_map<int, int> prefixCount;
    prefixCount[0] = 1; // empty prefix
    int sum = 0, count = 0;

    for (int num : nums) {
        sum += num;
        if (prefixCount.count(sum - k)) {
            count += prefixCount[sum - k];
        }
        prefixCount[sum]++;
    }
    return count;
}
```

**Java**
```java
import java.util.HashMap;
import java.util.Map;

public int subarraySum(int[] nums, int k) {
    Map<Integer, Integer> prefixCount = new HashMap<>();
    prefixCount.put(0, 1);
    int sum = 0, count = 0;

    for (int num : nums) {
        sum += num;
        count += prefixCount.getOrDefault(sum - k, 0);
        prefixCount.merge(sum, 1, Integer::sum);
    }
    return count;
}
```

**Python**
```python
from collections import defaultdict

def subarray_sum(nums: list[int], k: int) -> int:
    prefix_count = defaultdict(int)
    prefix_count[0] = 1
    total = count = 0

    for num in nums:
        total += num
        count += prefix_count[total - k]
        prefix_count[total] += 1
    return count
```

**Complexity:** Time O(n), Space O(n)

---

## Pattern 4: Kadane's Algorithm

**When to use:** Maximum subarray sum, maximum product subarray.

### Technique
Track the best subarray ending at each position. At each step, either extend the previous subarray or start fresh.

### Example: Maximum Subarray Sum

**C++**
```cpp
#include <vector>
#include <algorithm>
using namespace std;

int maxSubArray(vector<int>& nums) {
    int currentMax = nums[0], globalMax = nums[0];

    for (int i = 1; i < nums.size(); i++) {
        currentMax = max(nums[i], currentMax + nums[i]);
        globalMax = max(globalMax, currentMax);
    }
    return globalMax;
}
```

**Java**
```java
public int maxSubArray(int[] nums) {
    int currentMax = nums[0], globalMax = nums[0];

    for (int i = 1; i < nums.length; i++) {
        currentMax = Math.max(nums[i], currentMax + nums[i]);
        globalMax = Math.max(globalMax, currentMax);
    }
    return globalMax;
}
```

**Python**
```python
def max_sub_array(nums: list[int]) -> int:
    current_max = global_max = nums[0]

    for num in nums[1:]:
        current_max = max(num, current_max + num)
        global_max = max(global_max, current_max)
    return global_max
```

**Complexity:** Time O(n), Space O(1)

---

## Pattern 5: In-Place Array Manipulation

**When to use:** Removing elements, rotating arrays, moving zeroes — when O(1) space is required.

### Example: Move Zeroes to End

**C++**
```cpp
#include <vector>
using namespace std;

void moveZeroes(vector<int>& nums) {
    int insertPos = 0;
    for (int i = 0; i < nums.size(); i++) {
        if (nums[i] != 0) {
            swap(nums[insertPos], nums[i]);
            insertPos++;
        }
    }
}
```

**Java**
```java
public void moveZeroes(int[] nums) {
    int insertPos = 0;
    for (int i = 0; i < nums.length; i++) {
        if (nums[i] != 0) {
            int temp = nums[insertPos];
            nums[insertPos] = nums[i];
            nums[i] = temp;
            insertPos++;
        }
    }
}
```

**Python**
```python
def move_zeroes(nums: list[int]) -> None:
    insert_pos = 0
    for i in range(len(nums)):
        if nums[i] != 0:
            nums[insert_pos], nums[i] = nums[i], nums[insert_pos]
            insert_pos += 1
```

**Complexity:** Time O(n), Space O(1)

---

## Pattern 6: String Manipulation

**When to use:** Palindromes, anagrams, string matching, character frequency problems.

### Example: Valid Anagram

**C++**
```cpp
#include <string>
#include <array>
using namespace std;

bool isAnagram(string s, string t) {
    if (s.size() != t.size()) return false;
    array<int, 26> freq{};

    for (int i = 0; i < s.size(); i++) {
        freq[s[i] - 'a']++;
        freq[t[i] - 'a']--;
    }
    for (int f : freq) {
        if (f != 0) return false;
    }
    return true;
}
```

**Java**
```java
public boolean isAnagram(String s, String t) {
    if (s.length() != t.length()) return false;
    int[] freq = new int[26];

    for (int i = 0; i < s.length(); i++) {
        freq[s.charAt(i) - 'a']++;
        freq[t.charAt(i) - 'a']--;
    }
    for (int f : freq) {
        if (f != 0) return false;
    }
    return true;
}
```

**Python**
```python
from collections import Counter

def is_anagram(s: str, t: str) -> bool:
    return Counter(s) == Counter(t)

# Without Counter:
def is_anagram_manual(s: str, t: str) -> bool:
    if len(s) != len(t):
        return False
    freq = [0] * 26
    for cs, ct in zip(s, t):
        freq[ord(cs) - ord('a')] += 1
        freq[ord(ct) - ord('a')] -= 1
    return all(f == 0 for f in freq)
```

**Complexity:** Time O(n), Space O(1) (fixed 26-char alphabet)

---

## Pitfalls & Edge Cases

| Pitfall | How to Handle |
|---------|--------------|
| Empty array/string | Check `size == 0` before processing |
| Single element | Often a valid answer by itself |
| All same elements | Test with `[1,1,1,1]` or `"aaaa"` |
| Negative numbers | Kadane's still works; prefix sum needs hash map |
| Integer overflow | Use `long long` (C++) or `long` (Java) for sums |
| Off-by-one in sliding window | `right - left + 1` for window size |
| String immutability (Java/Python) | Use `StringBuilder` / list of chars for building |

---

## Practice Problems

| # | Problem | Difficulty | Pattern | Link |
|---|---------|-----------|---------|------|
| 1 | Two Sum | Easy | Hash Map / Two Pointer | [LeetCode 1](https://leetcode.com/problems/two-sum/) |
| 2 | Best Time to Buy and Sell Stock | Easy | Kadane's variant | [LeetCode 121](https://leetcode.com/problems/best-time-to-buy-and-sell-stock/) |
| 3 | Contains Duplicate | Easy | Hash Set | [LeetCode 217](https://leetcode.com/problems/contains-duplicate/) |
| 4 | Product of Array Except Self | Medium | Prefix/Suffix | [LeetCode 238](https://leetcode.com/problems/product-of-array-except-self/) |
| 5 | Maximum Subarray | Medium | Kadane's | [LeetCode 53](https://leetcode.com/problems/maximum-subarray/) |
| 6 | 3Sum | Medium | Two Pointers | [LeetCode 15](https://leetcode.com/problems/3sum/) |
| 7 | Container With Most Water | Medium | Two Pointers | [LeetCode 11](https://leetcode.com/problems/container-with-most-water/) |
| 8 | Longest Substring Without Repeating | Medium | Sliding Window | [LeetCode 3](https://leetcode.com/problems/longest-substring-without-repeating-characters/) |
| 9 | Minimum Window Substring | Hard | Sliding Window | [LeetCode 76](https://leetcode.com/problems/minimum-window-substring/) |
| 10 | Trapping Rain Water | Hard | Two Pointers / Stack | [LeetCode 42](https://leetcode.com/problems/trapping-rain-water/) |

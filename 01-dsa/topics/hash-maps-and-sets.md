# Hash Maps & Sets

The go-to data structure for O(1) lookups — solves frequency counting, grouping, deduplication, and lookup problems.

---

## Core Concepts

### Hash Map (Dictionary)
- Key-value pairs with O(1) average lookup, insert, delete
- **Hash function** maps keys to bucket indices
- **Collision handling:** chaining (linked lists) or open addressing (probing)
- **Load factor:** ratio of entries to buckets; resize at ~0.75

### Hash Set
- Stores unique elements only
- O(1) average `add`, `contains`, `remove`

### Language Implementations

| Operation | C++ `unordered_map` | Java `HashMap` | Python `dict` |
|-----------|-------------------|---------------|--------------|
| Insert | `map[key] = val` | `map.put(key, val)` | `d[key] = val` |
| Lookup | `map[key]` / `map.at(key)` | `map.get(key)` | `d[key]` / `d.get(key)` |
| Check key | `map.count(key)` | `map.containsKey(key)` | `key in d` |
| Delete | `map.erase(key)` | `map.remove(key)` | `del d[key]` |
| Size | `map.size()` | `map.size()` | `len(d)` |
| Iterate | `for (auto& [k,v] : map)` | `for (var e : map.entrySet())` | `for k, v in d.items()` |

---

## Pattern 1: Frequency Counting

**When to use:** Anagrams, majority element, top-K frequent, character counts.

### Example: Top K Frequent Elements

**C++**
```cpp
#include <vector>
#include <unordered_map>
#include <queue>
using namespace std;

vector<int> topKFrequent(vector<int>& nums, int k) {
    unordered_map<int, int> freq;
    for (int n : nums) freq[n]++;

    // Min-heap of size k
    auto cmp = [](pair<int,int>& a, pair<int,int>& b) {
        return a.second > b.second;
    };
    priority_queue<pair<int,int>, vector<pair<int,int>>, decltype(cmp)> pq(cmp);

    for (auto& [num, count] : freq) {
        pq.push({num, count});
        if (pq.size() > k) pq.pop();
    }

    vector<int> result;
    while (!pq.empty()) {
        result.push_back(pq.top().first);
        pq.pop();
    }
    return result;
}
```

**Java**
```java
public int[] topKFrequent(int[] nums, int k) {
    Map<Integer, Integer> freq = new HashMap<>();
    for (int n : nums) freq.merge(n, 1, Integer::sum);

    // Min-heap by frequency
    PriorityQueue<Integer> pq = new PriorityQueue<>(
        Comparator.comparingInt(freq::get)
    );

    for (int num : freq.keySet()) {
        pq.offer(num);
        if (pq.size() > k) pq.poll();
    }

    int[] result = new int[k];
    for (int i = 0; i < k; i++) result[i] = pq.poll();
    return result;
}
```

**Python**
```python
from collections import Counter

def top_k_frequent(nums: list[int], k: int) -> list[int]:
    return [num for num, _ in Counter(nums).most_common(k)]

# Bucket sort approach — O(n) time
def top_k_frequent_bucket(nums: list[int], k: int) -> list[int]:
    freq = Counter(nums)
    buckets = [[] for _ in range(len(nums) + 1)]
    for num, count in freq.items():
        buckets[count].append(num)

    result = []
    for i in range(len(buckets) - 1, 0, -1):
        for num in buckets[i]:
            result.append(num)
            if len(result) == k:
                return result
    return result
```

---

## Pattern 2: Two Sum / Complement Lookup

**When to use:** Finding pairs with a given sum, difference, or product.

### Example: Two Sum (Unsorted)

**C++**
```cpp
vector<int> twoSum(vector<int>& nums, int target) {
    unordered_map<int, int> seen; // value → index
    for (int i = 0; i < nums.size(); i++) {
        int complement = target - nums[i];
        if (seen.count(complement)) {
            return {seen[complement], i};
        }
        seen[nums[i]] = i;
    }
    return {};
}
```

**Java**
```java
public int[] twoSum(int[] nums, int target) {
    Map<Integer, Integer> seen = new HashMap<>();
    for (int i = 0; i < nums.length; i++) {
        int complement = target - nums[i];
        if (seen.containsKey(complement)) {
            return new int[]{seen.get(complement), i};
        }
        seen.put(nums[i], i);
    }
    return new int[]{};
}
```

**Python**
```python
def two_sum(nums: list[int], target: int) -> list[int]:
    seen = {}  # value → index
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    return []
```

---

## Pattern 3: Grouping / Bucketing

**When to use:** Group anagrams, group by property, partition.

### Example: Group Anagrams

**C++**
```cpp
vector<vector<string>> groupAnagrams(vector<string>& strs) {
    unordered_map<string, vector<string>> groups;
    for (const string& s : strs) {
        string key = s;
        sort(key.begin(), key.end());
        groups[key].push_back(s);
    }
    vector<vector<string>> result;
    for (auto& [_, group] : groups) {
        result.push_back(group);
    }
    return result;
}
```

**Python**
```python
from collections import defaultdict

def group_anagrams(strs: list[str]) -> list[list[str]]:
    groups = defaultdict(list)
    for s in strs:
        key = tuple(sorted(s))  # or use character count tuple
        groups[key].append(s)
    return list(groups.values())

# O(n*k) approach using character count as key
def group_anagrams_optimal(strs: list[str]) -> list[list[str]]:
    groups = defaultdict(list)
    for s in strs:
        count = [0] * 26
        for c in s:
            count[ord(c) - ord('a')] += 1
        groups[tuple(count)].append(s)
    return list(groups.values())
```

---

## Pattern 4: Set Operations

**When to use:** Intersection, union, difference, deduplication, membership testing.

### Example: Intersection of Two Arrays

**Python**
```python
def intersection(nums1: list[int], nums2: list[int]) -> list[int]:
    return list(set(nums1) & set(nums2))

# With duplicates (frequency-aware)
from collections import Counter
def intersect(nums1: list[int], nums2: list[int]) -> list[int]:
    counts = Counter(nums1)
    result = []
    for num in nums2:
        if counts[num] > 0:
            result.append(num)
            counts[num] -= 1
    return result
```

---

## Pattern 5: Seen / Visited Tracking

**When to use:** Cycle detection, duplicate detection, graph traversal visited set.

### Example: Longest Consecutive Sequence

**C++**
```cpp
int longestConsecutive(vector<int>& nums) {
    unordered_set<int> numSet(nums.begin(), nums.end());
    int maxLen = 0;

    for (int num : numSet) {
        // Only start counting from sequence beginning
        if (!numSet.count(num - 1)) {
            int curr = num, len = 1;
            while (numSet.count(curr + 1)) {
                curr++;
                len++;
            }
            maxLen = max(maxLen, len);
        }
    }
    return maxLen;
}
```

**Python**
```python
def longest_consecutive(nums: list[int]) -> int:
    num_set = set(nums)
    max_len = 0

    for num in num_set:
        if num - 1 not in num_set:  # start of sequence
            curr, length = num, 1
            while curr + 1 in num_set:
                curr += 1
                length += 1
            max_len = max(max_len, length)
    return max_len
```

**Complexity:** Time O(n), Space O(n)

---

## Pitfalls & Edge Cases

| Pitfall | How to Handle |
|---------|--------------|
| C++ `unordered_map[key]` creates entry | Use `count()` or `find()` to check first |
| Hash collisions → O(n) worst case | Rare in practice; use good hash functions |
| Unhashable types (Python lists, dicts) | Convert to `tuple` for dict keys |
| Java: `Integer` caching (−128 to 127) | Use `.equals()` not `==` for Integer comparison |
| Ordering not preserved | C++: use `map` for ordered; Python 3.7+: `dict` preserves insertion order |
| Empty input | Check for empty array before processing |

---

## Practice Problems

| # | Problem | Difficulty | Pattern | Link |
|---|---------|-----------|---------|------|
| 1 | Two Sum | Easy | Complement Lookup | [LeetCode 1](https://leetcode.com/problems/two-sum/) |
| 2 | Contains Duplicate | Easy | Set | [LeetCode 217](https://leetcode.com/problems/contains-duplicate/) |
| 3 | Valid Anagram | Easy | Frequency Count | [LeetCode 242](https://leetcode.com/problems/valid-anagram/) |
| 4 | Group Anagrams | Medium | Grouping | [LeetCode 49](https://leetcode.com/problems/group-anagrams/) |
| 5 | Top K Frequent Elements | Medium | Frequency + Heap | [LeetCode 347](https://leetcode.com/problems/top-k-frequent-elements/) |
| 6 | Longest Consecutive Sequence | Medium | Set | [LeetCode 128](https://leetcode.com/problems/longest-consecutive-sequence/) |
| 7 | Subarray Sum Equals K | Medium | Prefix Sum + Map | [LeetCode 560](https://leetcode.com/problems/subarray-sum-equals-k/) |
| 8 | LRU Cache | Medium | Map + Doubly LL | [LeetCode 146](https://leetcode.com/problems/lru-cache/) |

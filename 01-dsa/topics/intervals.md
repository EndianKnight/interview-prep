# Intervals

A focused category that appears frequently — merging, inserting, overlapping, and scheduling.

---

## Core Technique

Most interval problems follow the same recipe:
1. **Sort** by start time (or end time for scheduling)
2. **Iterate** and compare current interval with the previous
3. **Merge/split/count** based on overlap condition

**Overlap condition:** two intervals `[a, b]` and `[c, d]` overlap if `a < d && c < b` (assuming `a ≤ b`, `c ≤ d`).

---

## Pattern 1: Merge Intervals

**C++**
```cpp
vector<vector<int>> merge(vector<vector<int>>& intervals) {
    sort(intervals.begin(), intervals.end());
    vector<vector<int>> merged;

    for (auto& iv : intervals) {
        if (merged.empty() || merged.back()[1] < iv[0]) {
            merged.push_back(iv);
        } else {
            merged.back()[1] = max(merged.back()[1], iv[1]);
        }
    }
    return merged;
}
```

**Java**
```java
public int[][] merge(int[][] intervals) {
    Arrays.sort(intervals, Comparator.comparingInt(a -> a[0]));
    List<int[]> merged = new ArrayList<>();

    for (int[] iv : intervals) {
        if (merged.isEmpty() || merged.getLast()[1] < iv[0]) {
            merged.add(iv);
        } else {
            merged.getLast()[1] = Math.max(merged.getLast()[1], iv[1]);
        }
    }
    return merged.toArray(new int[0][]);
}
```

**Python**
```python
def merge(intervals: list[list[int]]) -> list[list[int]]:
    intervals.sort()
    merged = []
    for start, end in intervals:
        if merged and merged[-1][1] >= start:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged
```

---

## Pattern 2: Insert Interval

**Python**
```python
def insert(intervals: list[list[int]], new: list[int]) -> list[list[int]]:
    result = []
    i = 0
    # Add all intervals before new
    while i < len(intervals) and intervals[i][1] < new[0]:
        result.append(intervals[i])
        i += 1
    # Merge overlapping
    while i < len(intervals) and intervals[i][0] <= new[1]:
        new = [min(new[0], intervals[i][0]), max(new[1], intervals[i][1])]
        i += 1
    result.append(new)
    # Add remaining
    result.extend(intervals[i:])
    return result
```

---

## Pattern 3: Line Sweep (Event-Based)

**When to use:** Meeting rooms, overlapping count, skyline problem.

**Python**
```python
def min_meeting_rooms(intervals: list[list[int]]) -> int:
    events = []
    for start, end in intervals:
        events.append((start, 1))   # meeting starts
        events.append((end, -1))    # meeting ends
    events.sort()

    max_rooms = current = 0
    for _, delta in events:
        current += delta
        max_rooms = max(max_rooms, current)
    return max_rooms
```

---

## Practice Problems

| # | Problem | Difficulty | Pattern | Link |
|---|---------|-----------|---------|------|
| 1 | Merge Intervals | Medium | Merge | [LeetCode 56](https://leetcode.com/problems/merge-intervals/) |
| 2 | Insert Interval | Medium | Insert | [LeetCode 57](https://leetcode.com/problems/insert-interval/) |
| 3 | Non-Overlapping Intervals | Medium | Greedy | [LeetCode 435](https://leetcode.com/problems/non-overlapping-intervals/) |
| 4 | Meeting Rooms II | Medium | Line Sweep | [LeetCode 253](https://leetcode.com/problems/meeting-rooms-ii/) |
| 5 | Interval List Intersections | Medium | Two Pointers | [LeetCode 986](https://leetcode.com/problems/interval-list-intersections/) |

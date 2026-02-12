# Sorting & Searching

Mastering sorting algorithms and binary search variants is essential — they underpin many interview solutions.

---

## Sorting Algorithms Comparison

| Algorithm | Best | Average | Worst | Space | Stable |
|-----------|------|---------|-------|-------|--------|
| Bubble Sort | O(n) | O(n²) | O(n²) | O(1) | ✅ |
| Selection Sort | O(n²) | O(n²) | O(n²) | O(1) | ❌ |
| Insertion Sort | O(n) | O(n²) | O(n²) | O(1) | ✅ |
| Merge Sort | O(n log n) | O(n log n) | O(n log n) | O(n) | ✅ |
| Quick Sort | O(n log n) | O(n log n) | O(n²) | O(log n) | ❌ |
| Heap Sort | O(n log n) | O(n log n) | O(n log n) | O(1) | ❌ |
| Counting Sort | O(n+k) | O(n+k) | O(n+k) | O(k) | ✅ |
| Radix Sort | O(d·(n+k)) | O(d·(n+k)) | O(d·(n+k)) | O(n+k) | ✅ |

### Quick Sort Implementation

**C++**
```cpp
#include <vector>
#include <cstdlib>
using namespace std;

int partition(vector<int>& arr, int lo, int hi) {
    int pivotIdx = lo + rand() % (hi - lo + 1);
    swap(arr[pivotIdx], arr[hi]);
    int pivot = arr[hi], i = lo;

    for (int j = lo; j < hi; j++) {
        if (arr[j] <= pivot) swap(arr[i++], arr[j]);
    }
    swap(arr[i], arr[hi]);
    return i;
}

void quicksort(vector<int>& arr, int lo, int hi) {
    if (lo >= hi) return;
    int p = partition(arr, lo, hi);
    quicksort(arr, lo, p - 1);
    quicksort(arr, p + 1, hi);
}
```

**Java**
```java
import java.util.Random;

public void quicksort(int[] arr, int lo, int hi) {
    if (lo >= hi) return;
    int pivotIdx = lo + new Random().nextInt(hi - lo + 1);
    int temp = arr[pivotIdx]; arr[pivotIdx] = arr[hi]; arr[hi] = temp;
    int pivot = arr[hi], i = lo;

    for (int j = lo; j < hi; j++) {
        if (arr[j] <= pivot) {
            temp = arr[i]; arr[i] = arr[j]; arr[j] = temp;
            i++;
        }
    }
    temp = arr[i]; arr[i] = arr[hi]; arr[hi] = temp;
    quicksort(arr, lo, i - 1);
    quicksort(arr, i + 1, hi);
}
```

**Python**
```python
import random

def quicksort(arr: list[int], lo: int = 0, hi: int = None) -> None:
    if hi is None:
        hi = len(arr) - 1
    if lo >= hi:
        return

    pivot_idx = random.randint(lo, hi)
    arr[pivot_idx], arr[hi] = arr[hi], arr[pivot_idx]
    pivot = arr[hi]

    i = lo
    for j in range(lo, hi):
        if arr[j] <= pivot:
            arr[i], arr[j] = arr[j], arr[i]
            i += 1
    arr[i], arr[hi] = arr[hi], arr[i]

    quicksort(arr, lo, i - 1)
    quicksort(arr, i + 1, hi)
```

### Merge Sort Implementation

**C++**
```cpp
void merge(vector<int>& arr, int lo, int mid, int hi) {
    vector<int> left(arr.begin() + lo, arr.begin() + mid + 1);
    vector<int> right(arr.begin() + mid + 1, arr.begin() + hi + 1);
    int i = 0, j = 0, k = lo;

    while (i < left.size() && j < right.size())
        arr[k++] = (left[i] <= right[j]) ? left[i++] : right[j++];
    while (i < left.size()) arr[k++] = left[i++];
    while (j < right.size()) arr[k++] = right[j++];
}

void mergeSort(vector<int>& arr, int lo, int hi) {
    if (lo >= hi) return;
    int mid = lo + (hi - lo) / 2;
    mergeSort(arr, lo, mid);
    mergeSort(arr, mid + 1, hi);
    merge(arr, lo, mid, hi);
}
```

**Java**
```java
public void mergeSort(int[] arr, int lo, int hi) {
    if (lo >= hi) return;
    int mid = lo + (hi - lo) / 2;
    mergeSort(arr, lo, mid);
    mergeSort(arr, mid + 1, hi);
    merge(arr, lo, mid, hi);
}

private void merge(int[] arr, int lo, int mid, int hi) {
    int[] left = Arrays.copyOfRange(arr, lo, mid + 1);
    int[] right = Arrays.copyOfRange(arr, mid + 1, hi + 1);
    int i = 0, j = 0, k = lo;

    while (i < left.length && j < right.length)
        arr[k++] = (left[i] <= right[j]) ? left[i++] : right[j++];
    while (i < left.length) arr[k++] = left[i++];
    while (j < right.length) arr[k++] = right[j++];
}
```

**Python**
```python
def merge_sort(arr: list[int]) -> list[int]:
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])

    merged, i, j = [], 0, 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            merged.append(left[i]); i += 1
        else:
            merged.append(right[j]); j += 1
    merged.extend(left[i:])
    merged.extend(right[j:])
    return merged
```

### Language Built-in Sorting

| | C++ | Java | Python |
|-|-----|------|--------|
| Sort | `sort(v.begin(), v.end())` | `Arrays.sort(arr)` | `arr.sort()` / `sorted(arr)` |
| Custom | `sort(v.begin(), v.end(), cmp)` | `Arrays.sort(arr, Comparator)` | `sort(key=lambda)` |
| Stable | `stable_sort(...)` | `Arrays.sort()` (objects) | `sort()` (Timsort is stable) |
| Algorithm | Introsort | Dual-pivot quicksort / Timsort | Timsort |

---

## Pattern 1: Standard Binary Search

**When to use:** Finding an element or its position in a sorted array.

### Template

**C++**
```cpp
int binarySearch(vector<int>& nums, int target) {
    int lo = 0, hi = nums.size() - 1;
    while (lo <= hi) {
        int mid = lo + (hi - lo) / 2;
        if (nums[mid] == target) return mid;
        else if (nums[mid] < target) lo = mid + 1;
        else hi = mid - 1;
    }
    return -1;
}
```

**Java**
```java
public int binarySearch(int[] nums, int target) {
    int lo = 0, hi = nums.length - 1;
    while (lo <= hi) {
        int mid = lo + (hi - lo) / 2;
        if (nums[mid] == target) return mid;
        else if (nums[mid] < target) lo = mid + 1;
        else hi = mid - 1;
    }
    return -1;
}
```

**Python**
```python
def binary_search(nums: list[int], target: int) -> int:
    lo, hi = 0, len(nums) - 1
    while lo <= hi:
        mid = lo + (hi - lo) // 2  # avoid overflow
        if nums[mid] == target:
            return mid
        elif nums[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1  # not found
```

---

## Pattern 2: Binary Search on Answer (Bisect)

**When to use:** Minimize/maximize a value that satisfies a condition — capacity, speed, distance, days.

### Technique
Binary search on the answer space. For each candidate, check if it's feasible in O(n).

### Example: Koko Eating Bananas

**C++**
```cpp
int minEatingSpeed(vector<int>& piles, int h) {
    int lo = 1, hi = *max_element(piles.begin(), piles.end());
    while (lo < hi) {
        int mid = lo + (hi - lo) / 2;
        int hours = 0;
        for (int p : piles) hours += (p + mid - 1) / mid;
        if (hours <= h) hi = mid;
        else lo = mid + 1;
    }
    return lo;
}
```

**Java**
```java
public int minEatingSpeed(int[] piles, int h) {
    int lo = 1, hi = Arrays.stream(piles).max().getAsInt();
    while (lo < hi) {
        int mid = lo + (hi - lo) / 2;
        int hours = 0;
        for (int p : piles) hours += (p + mid - 1) / mid;
        if (hours <= h) hi = mid;
        else lo = mid + 1;
    }
    return lo;
}
```

**Python**
```python
import math

def min_eating_speed(piles: list[int], h: int) -> int:
    lo, hi = 1, max(piles)

    while lo < hi:
        mid = lo + (hi - lo) // 2
        hours = sum(math.ceil(p / mid) for p in piles)
        if hours <= h:
            hi = mid  # try slower
        else:
            lo = mid + 1  # need faster
    return lo
```

---

## Pattern 3: Finding Boundaries (Lower/Upper Bound)

**When to use:** First/last occurrence, insertion point, count of target.

### Example: First and Last Position of Target

**C++**
```cpp
vector<int> searchRange(vector<int>& nums, int target) {
    int left = findLeft(nums, target);
    int right = findRight(nums, target);
    if (left <= right) return {left, right};
    return {-1, -1};
}

int findLeft(vector<int>& nums, int target) {
    int lo = 0, hi = nums.size() - 1;
    while (lo <= hi) {
        int mid = lo + (hi - lo) / 2;
        if (nums[mid] < target) lo = mid + 1;
        else hi = mid - 1;
    }
    return lo;
}

int findRight(vector<int>& nums, int target) {
    int lo = 0, hi = nums.size() - 1;
    while (lo <= hi) {
        int mid = lo + (hi - lo) / 2;
        if (nums[mid] <= target) lo = mid + 1;
        else hi = mid - 1;
    }
    return hi;
}
```

**Java**
```java
public int[] searchRange(int[] nums, int target) {
    int left = findLeft(nums, target);
    int right = findRight(nums, target);
    if (left <= right) return new int[]{left, right};
    return new int[]{-1, -1};
}

private int findLeft(int[] nums, int target) {
    int lo = 0, hi = nums.length - 1;
    while (lo <= hi) {
        int mid = lo + (hi - lo) / 2;
        if (nums[mid] < target) lo = mid + 1;
        else hi = mid - 1;
    }
    return lo;
}

private int findRight(int[] nums, int target) {
    int lo = 0, hi = nums.length - 1;
    while (lo <= hi) {
        int mid = lo + (hi - lo) / 2;
        if (nums[mid] <= target) lo = mid + 1;
        else hi = mid - 1;
    }
    return hi;
}
```

**Python**
```python
def search_range(nums: list[int], target: int) -> list[int]:
    def find_left():
        lo, hi = 0, len(nums) - 1
        while lo <= hi:
            mid = lo + (hi - lo) // 2
            if nums[mid] < target:
                lo = mid + 1
            else:
                hi = mid - 1
        return lo

    def find_right():
        lo, hi = 0, len(nums) - 1
        while lo <= hi:
            mid = lo + (hi - lo) // 2
            if nums[mid] <= target:
                lo = mid + 1
            else:
                hi = mid - 1
        return hi

    left, right = find_left(), find_right()
    if left <= right:
        return [left, right]
    return [-1, -1]
```

### Using Language Built-ins

**C++**
```cpp
auto [lo, hi] = equal_range(nums.begin(), nums.end(), target);
// lo = lower_bound (first >=), hi = upper_bound (first >)
if (lo != hi) return {(int)(lo - nums.begin()), (int)(hi - nums.begin() - 1)};
```

**Java**
```java
int lo = Collections.binarySearch(list, target); // only finds one occurrence
// For bounds: use Arrays.binarySearch or implement manually as above
```

**Python**
```python
from bisect import bisect_left, bisect_right
lo = bisect_left(nums, target)
hi = bisect_right(nums, target) - 1
```

---

## Pattern 4: Search in Rotated/Modified Array

**When to use:** Rotated sorted array, mountain array, bitonic sequence.

### Example: Search in Rotated Sorted Array

**C++**
```cpp
int search(vector<int>& nums, int target) {
    int lo = 0, hi = nums.size() - 1;
    while (lo <= hi) {
        int mid = lo + (hi - lo) / 2;
        if (nums[mid] == target) return mid;

        if (nums[lo] <= nums[mid]) { // left half sorted
            if (nums[lo] <= target && target < nums[mid])
                hi = mid - 1;
            else
                lo = mid + 1;
        } else { // right half sorted
            if (nums[mid] < target && target <= nums[hi])
                lo = mid + 1;
            else
                hi = mid - 1;
        }
    }
    return -1;
}
```

**Java**
```java
public int search(int[] nums, int target) {
    int lo = 0, hi = nums.length - 1;
    while (lo <= hi) {
        int mid = lo + (hi - lo) / 2;
        if (nums[mid] == target) return mid;

        if (nums[lo] <= nums[mid]) { // left half sorted
            if (nums[lo] <= target && target < nums[mid])
                hi = mid - 1;
            else
                lo = mid + 1;
        } else { // right half sorted
            if (nums[mid] < target && target <= nums[hi])
                lo = mid + 1;
            else
                hi = mid - 1;
        }
    }
    return -1;
}
```

**Python**
```python
def search(nums: list[int], target: int) -> int:
    lo, hi = 0, len(nums) - 1
    while lo <= hi:
        mid = lo + (hi - lo) // 2
        if nums[mid] == target:
            return mid

        if nums[lo] <= nums[mid]:  # left half is sorted
            if nums[lo] <= target < nums[mid]:
                hi = mid - 1
            else:
                lo = mid + 1
        else:  # right half is sorted
            if nums[mid] < target <= nums[hi]:
                lo = mid + 1
            else:
                hi = mid - 1
    return -1
```

---

## Pitfalls & Edge Cases

| Pitfall | How to Handle |
|---------|--------------|
| Integer overflow in `(lo + hi) / 2` | Use `lo + (hi - lo) / 2` |
| Off-by-one in `lo < hi` vs `lo <= hi` | `lo <= hi` when searching for exact; `lo < hi` when converging |
| Empty array | Check `nums.length == 0` |
| Single element | Works correctly with proper bounds |
| Duplicates in rotated array | May need O(n) worst case |
| Wrong search space for bisect problems | Verify `lo` and `hi` bounds carefully |

---

## Practice Problems

| # | Problem | Difficulty | Pattern | Link |
|---|---------|-----------|---------|------|
| 1 | Binary Search | Easy | Standard | [LeetCode 704](https://leetcode.com/problems/binary-search/) |
| 2 | Search Insert Position | Easy | Lower Bound | [LeetCode 35](https://leetcode.com/problems/search-insert-position/) |
| 3 | Find First and Last Position | Medium | Boundaries | [LeetCode 34](https://leetcode.com/problems/find-first-and-last-position-of-element-in-sorted-array/) |
| 4 | Search in Rotated Array | Medium | Modified | [LeetCode 33](https://leetcode.com/problems/search-in-rotated-sorted-array/) |
| 5 | Find Peak Element | Medium | Modified | [LeetCode 162](https://leetcode.com/problems/find-peak-element/) |
| 6 | Koko Eating Bananas | Medium | Bisect on Answer | [LeetCode 875](https://leetcode.com/problems/koko-eating-bananas/) |
| 7 | Median of Two Sorted Arrays | Hard | Binary Search | [LeetCode 4](https://leetcode.com/problems/median-of-two-sorted-arrays/) |

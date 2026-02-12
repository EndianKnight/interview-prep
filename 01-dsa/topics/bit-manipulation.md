# Bit Manipulation

Low-level operations that unlock O(1) space solutions and clever tricks for specific problem classes.

---

## Essential Bit Operations

| Operation | Syntax (C++/Java) | Python | Result |
|-----------|-------------------|--------|--------|
| AND | `a & b` | `a & b` | 1 if both bits are 1 |
| OR | `a \| b` | `a \| b` | 1 if either bit is 1 |
| XOR | `a ^ b` | `a ^ b` | 1 if bits differ |
| NOT | `~a` | `~a` | Flip all bits |
| Left shift | `a << n` | `a << n` | Multiply by 2ⁿ |
| Right shift | `a >> n` | `a >> n` | Divide by 2ⁿ |

## Key Identities

```
a ^ 0 = a           // XOR with 0 → unchanged
a ^ a = 0           // XOR with self → 0
a ^ b ^ a = b       // XOR is self-inverse
a & (a - 1)          // Clear lowest set bit
a & (-a)             // Isolate lowest set bit
a | (a + 1)          // Set lowest unset bit
```

---

## Pattern 1: XOR Tricks

### Example: Single Number (find unique in pairs)

**C++**
```cpp
int singleNumber(vector<int>& nums) {
    int result = 0;
    for (int n : nums) result ^= n;
    return result;
}
```

**Python**
```python
from functools import reduce
from operator import xor

def single_number(nums: list[int]) -> int:
    return reduce(xor, nums)
```

### Example: Missing Number

**Python**
```python
def missing_number(nums: list[int]) -> int:
    result = len(nums)
    for i, num in enumerate(nums):
        result ^= i ^ num
    return result
```

---

## Pattern 2: Bit Counting

### Count Set Bits (Brian Kernighan's)

**C++**
```cpp
int countBits(int n) {
    int count = 0;
    while (n) {
        n &= (n - 1);  // clear lowest set bit
        count++;
    }
    return count;
}
// Or: __builtin_popcount(n)
```

**Java**
```java
// Integer.bitCount(n) — built-in
int countBits(int n) {
    int count = 0;
    while (n != 0) {
        n &= (n - 1);
        count++;
    }
    return count;
}
```

**Python**
```python
def count_bits(n: int) -> int:
    return bin(n).count('1')
    # Or: n.bit_count()  (Python 3.10+)
```

### Power of Two Check

```python
def is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0
```

---

## Pattern 3: Bit Masking

**When to use:** Subset representation, permission flags, DP with bitmask.

### Example: Subsets Using Bitmask

**Python**
```python
def subsets_bitmask(nums: list[int]) -> list[list[int]]:
    n = len(nums)
    result = []
    for mask in range(1 << n):  # 0 to 2^n - 1
        subset = [nums[i] for i in range(n) if mask & (1 << i)]
        result.append(subset)
    return result
```

**C++**
```cpp
vector<vector<int>> subsets(vector<int>& nums) {
    int n = nums.size();
    vector<vector<int>> result;
    for (int mask = 0; mask < (1 << n); mask++) {
        vector<int> subset;
        for (int i = 0; i < n; i++) {
            if (mask & (1 << i)) subset.push_back(nums[i]);
        }
        result.push_back(subset);
    }
    return result;
}
```

---

## Pitfalls

| Pitfall | How to Handle |
|---------|--------------|
| Python integers are arbitrary precision | No overflow, but `~` behaves differently than C++ |
| Signed vs unsigned right shift | Java: `>>>` unsigned, `>>` signed. C++: implementation-defined for negative |
| Operator precedence | `a & b == 0` parses as `a & (b == 0)` — use parentheses! |

---

## Practice Problems

| # | Problem | Difficulty | Pattern | Link |
|---|---------|-----------|---------|------|
| 1 | Single Number | Easy | XOR | [LeetCode 136](https://leetcode.com/problems/single-number/) |
| 2 | Number of 1 Bits | Easy | Counting | [LeetCode 191](https://leetcode.com/problems/number-of-1-bits/) |
| 3 | Missing Number | Easy | XOR | [LeetCode 268](https://leetcode.com/problems/missing-number/) |
| 4 | Reverse Bits | Easy | Bit manipulation | [LeetCode 190](https://leetcode.com/problems/reverse-bits/) |
| 5 | Counting Bits | Easy | DP + Bits | [LeetCode 338](https://leetcode.com/problems/counting-bits/) |
| 6 | Sum of Two Integers | Medium | Bit math | [LeetCode 371](https://leetcode.com/problems/sum-of-two-integers/) |

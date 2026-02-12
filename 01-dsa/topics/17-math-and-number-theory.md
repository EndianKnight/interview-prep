# Math & Number Theory

Essential mathematical patterns and number theory concepts for coding interviews.

---

## Core Concepts

### Modular Arithmetic
- `(a + b) % m = ((a % m) + (b % m)) % m`
- `(a * b) % m = ((a % m) * (b % m)) % m`
- Subtraction: `(a - b) % m = ((a % m) - (b % m) + m) % m`
- Division requires **modular inverse** (not simple `%`)

### Common Constants
```python
MOD = 10**9 + 7   # prime, fits in 32-bit signed int
```

---

## Pattern 1: Exponentiation & Powers

### Fast Power (Binary Exponentiation)

**C++**
```cpp
long long power(long long base, long long exp, long long mod) {
    long long result = 1;
    base %= mod;
    while (exp > 0) {
        if (exp & 1) result = result * base % mod;
        base = base * base % mod;
        exp >>= 1;
    }
    return result;
}
```

**Java**
```java
public long power(long base, long exp, long mod) {
    long result = 1;
    base %= mod;
    while (exp > 0) {
        if ((exp & 1) == 1) result = result * base % mod;
        base = base * base % mod;
        exp >>= 1;
    }
    return result;
}
```

**Python**
```python
# Built-in is best:
pow(base, exp, mod)  # O(log exp), handles large numbers

# Manual implementation:
def power(base, exp, mod):
    result = 1
    base %= mod
    while exp > 0:
        if exp & 1:
            result = result * base % mod
        base = base * base % mod
        exp >>= 1
    return result
```

---

## Pattern 2: GCD, LCM & Extended Euclidean

**C++**
```cpp
// C++17: std::gcd, std::lcm in <numeric>
#include <numeric>
int g = gcd(a, b);
int l = lcm(a, b);

// Manual Euclidean
int gcd(int a, int b) {
    while (b) { a %= b; swap(a, b); }
    return a;
}
int lcm(int a, int b) { return a / gcd(a, b) * b; }  // avoid overflow
```

**Java**
```java
// Manual (Java doesn't have built-in until BigInteger)
public int gcd(int a, int b) {
    while (b != 0) { int t = b; b = a % b; a = t; }
    return a;
}
public int lcm(int a, int b) { return a / gcd(a, b) * b; }
```

**Python**
```python
from math import gcd, lcm  # lcm available Python 3.9+

# Multiple values
from functools import reduce
g = reduce(gcd, nums)
l = reduce(lcm, nums)

# Extended Euclidean: ax + by = gcd(a, b)
def extended_gcd(a, b):
    if b == 0:
        return a, 1, 0
    g, x, y = extended_gcd(b, a % b)
    return g, y, x - (a // b) * y

# Modular inverse: a^(-1) mod m (when gcd(a, m) = 1)
def mod_inverse(a, m):
    return pow(a, m - 2, m)  # Fermat's little theorem (m must be prime)
```

---

## Pattern 3: Prime Numbers

### Sieve of Eratosthenes

**C++**
```cpp
vector<bool> sieve(int n) {
    vector<bool> is_prime(n + 1, true);
    is_prime[0] = is_prime[1] = false;
    for (int i = 2; i * i <= n; i++) {
        if (is_prime[i]) {
            for (int j = i * i; j <= n; j += i)
                is_prime[j] = false;
        }
    }
    return is_prime;
}
```

**Java**
```java
public boolean[] sieve(int n) {
    boolean[] isPrime = new boolean[n + 1];
    Arrays.fill(isPrime, true);
    isPrime[0] = isPrime[1] = false;
    for (int i = 2; i * i <= n; i++) {
        if (isPrime[i]) {
            for (int j = i * i; j <= n; j += i)
                isPrime[j] = false;
        }
    }
    return isPrime;
}
```

**Python**
```python
def sieve(n):
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            for j in range(i * i, n + 1, i):
                is_prime[j] = False
    return is_prime

# Get list of primes
primes = [i for i, v in enumerate(sieve(n)) if v]
```

### Primality Check

**C++**
```cpp
bool isPrime(int n) {
    if (n < 2) return false;
    if (n < 4) return true;
    if (n % 2 == 0 || n % 3 == 0) return false;
    for (int i = 5; (long)i * i <= n; i += 6)
        if (n % i == 0 || n % (i + 2) == 0) return false;
    return true;
}
```

**Java**
```java
public boolean isPrime(int n) {
    if (n < 2) return false;
    if (n < 4) return true;
    if (n % 2 == 0 || n % 3 == 0) return false;
    for (int i = 5; (long)i * i <= n; i += 6)
        if (n % i == 0 || n % (i + 2) == 0) return false;
    return true;
}
```

**Python**
```python
def is_prime(n):
    if n < 2: return False
    if n < 4: return True
    if n % 2 == 0 or n % 3 == 0: return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True
```

### Prime Factorization

**C++**
```cpp
vector<int> primeFactors(int n) {
    vector<int> factors;
    for (int d = 2; (long)d * d <= n; d++) {
        while (n % d == 0) {
            factors.push_back(d);
            n /= d;
        }
    }
    if (n > 1) factors.push_back(n);
    return factors;
}
```

**Java**
```java
public List<Integer> primeFactors(int n) {
    List<Integer> factors = new ArrayList<>();
    for (int d = 2; (long)d * d <= n; d++) {
        while (n % d == 0) {
            factors.add(d);
            n /= d;
        }
    }
    if (n > 1) factors.add(n);
    return factors;
}
```

**Python**
```python
def prime_factors(n):
    factors = []
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.append(d)
            n //= d
        d += 1
    if n > 1:
        factors.append(n)
    return factors
```

---

## Pattern 4: Combinatorics

### nCr (Combinations) with Modular Arithmetic

**C++**
```cpp
long long nCr(int n, int k, long long mod) {
    if (k > n) return 0;
    vector<long long> fact(n + 1), inv_fact(n + 1);
    fact[0] = 1;
    for (int i = 1; i <= n; i++) fact[i] = fact[i-1] * i % mod;
    inv_fact[n] = power(fact[n], mod - 2, mod);  // uses fast power
    for (int i = n - 1; i >= 0; i--) inv_fact[i] = inv_fact[i+1] * (i+1) % mod;
    return fact[n] % mod * inv_fact[k] % mod * inv_fact[n-k] % mod;
}
```

**Java**
```java
public long nCr(int n, int k, long mod) {
    if (k > n) return 0;
    long[] fact = new long[n + 1], invFact = new long[n + 1];
    fact[0] = 1;
    for (int i = 1; i <= n; i++) fact[i] = fact[i-1] * i % mod;
    invFact[n] = power(fact[n], mod - 2, mod);  // uses fast power
    for (int i = n - 1; i >= 0; i--) invFact[i] = invFact[i+1] * (i+1) % mod;
    return fact[n] % mod * invFact[k] % mod * invFact[n-k] % mod;
}
```

**Python**
```python
from math import comb  # Python 3.8+
comb(n, k)  # n choose k

# With mod using Pascal's Triangle
def build_pascal(n, mod):
    C = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        C[i][0] = 1
        for j in range(1, i + 1):
            C[i][j] = (C[i-1][j-1] + C[i-1][j]) % mod
    return C

# With mod using factorials and inverse
def nCr_mod(n, k, mod):
    if k > n: return 0
    # Precompute factorials
    fact = [1] * (n + 1)
    for i in range(1, n + 1):
        fact[i] = fact[i-1] * i % mod
    inv_fact = [1] * (n + 1)
    inv_fact[n] = pow(fact[n], mod - 2, mod)
    for i in range(n - 1, -1, -1):
        inv_fact[i] = inv_fact[i + 1] * (i + 1) % mod
    return fact[n] * inv_fact[k] % mod * inv_fact[n - k] % mod
```

### Catalan Numbers

```python
# C(n) = C(2n, n) / (n+1)
# Applications: valid parentheses, BST count, triangulation
# C(0)=1, C(1)=1, C(2)=2, C(3)=5, C(4)=14, ...
def catalan(n):
    return comb(2 * n, n) // (n + 1)

# DP approach
def catalan_dp(n):
    dp = [0] * (n + 1)
    dp[0] = dp[1] = 1
    for i in range(2, n + 1):
        for j in range(i):
            dp[i] += dp[j] * dp[i - 1 - j]
    return dp[n]
```

---

## Pattern 5: Number Properties

### Count Digits / Digit Manipulation

**Python**
```python
# Count digits
num_digits = len(str(n))       # or: int(log10(n)) + 1

# Sum of digits
digit_sum = sum(int(d) for d in str(n))

# Reverse a number
reversed_num = int(str(n)[::-1])

# Check palindrome number
def is_palindrome(n):
    s = str(n)
    return s == s[::-1]
```

### Integer Square Root

**Python**
```python
from math import isqrt  # Python 3.8+
isqrt(16)  # 4
isqrt(20)  # 4 (floor)

# Binary search approach
def isqrt_bs(n):
    lo, hi = 0, n
    while lo <= hi:
        mid = (lo + hi) // 2
        if mid * mid <= n:
            lo = mid + 1
        else:
            hi = mid - 1
    return hi
```

### Fibonacci (Matrix Exponentiation for O(log n))

**Python**
```python
def fib(n):
    """O(log n) using matrix exponentiation"""
    if n <= 1: return n
    def mat_mult(A, B):
        return [
            [A[0][0]*B[0][0] + A[0][1]*B[1][0],
             A[0][0]*B[0][1] + A[0][1]*B[1][1]],
            [A[1][0]*B[0][0] + A[1][1]*B[1][0],
             A[1][0]*B[0][1] + A[1][1]*B[1][1]]
        ]
    def mat_pow(M, p):
        result = [[1,0],[0,1]]  # identity
        while p:
            if p & 1: result = mat_mult(result, M)
            M = mat_mult(M, M)
            p >>= 1
        return result
    return mat_pow([[1,1],[1,0]], n)[0][1]
```

---

## Pattern 6: Counting & Frequency

### Count Trailing Zeros in n!

**C++**
```cpp
int trailingZeroes(int n) {
    int count = 0;
    while (n >= 5) { n /= 5; count += n; }
    return count;
}
```

**Java**
```java
public int trailingZeroes(int n) {
    int count = 0;
    while (n >= 5) { n /= 5; count += n; }
    return count;
}
```

**Python**
```python
def trailing_zeros(n):
    count = 0
    while n >= 5:
        n //= 5
        count += n
    return count
```

### Happy Number

**C++**
```cpp
bool isHappy(int n) {
    unordered_set<int> seen;
    while (n != 1 && !seen.count(n)) {
        seen.insert(n);
        int sum = 0;
        while (n > 0) { int d = n % 10; sum += d * d; n /= 10; }
        n = sum;
    }
    return n == 1;
}
```

**Java**
```java
public boolean isHappy(int n) {
    Set<Integer> seen = new HashSet<>();
    while (n != 1 && seen.add(n)) {
        int sum = 0;
        while (n > 0) { int d = n % 10; sum += d * d; n /= 10; }
        n = sum;
    }
    return n == 1;
}
```

**Python**
```python
def is_happy(n):
    seen = set()
    while n != 1 and n not in seen:
        seen.add(n)
        n = sum(int(d) ** 2 for d in str(n))
    return n == 1
```

---

## Pitfalls & Edge Cases

| Pitfall | How to Handle |
|---------|---------------|
| Integer overflow in multiplication | Use `long long` (C++), `long` (Java), or Python (auto) |
| Modular subtraction going negative | Add mod before taking `%`: `(a - b + mod) % mod` |
| Division with mod | Use modular inverse, not regular division |
| `0` and `1` are not prime | Handle edge cases in primality checks |
| GCD with zero | `gcd(a, 0) = a` |
| Factorial of large numbers | Precompute with mod, or use Python's arbitrary precision |
| Off-by-one in sieve | Allocate `n+1` entries |

---

## Practice Problems

| # | Problem | Difficulty | Pattern | Link |
|---|---------|-----------|---------|------|
| 1 | Count Primes | Medium | Sieve | [LC 204](https://leetcode.com/problems/count-primes/) |
| 2 | Power of Two/Three | Easy | Bit/Math | [LC 231](https://leetcode.com/problems/power-of-two/) |
| 3 | Happy Number | Easy | Cycle Detection | [LC 202](https://leetcode.com/problems/happy-number/) |
| 4 | Factorial Trailing Zeroes | Medium | Math | [LC 172](https://leetcode.com/problems/factorial-trailing-zeroes/) |
| 5 | Pow(x, n) | Medium | Fast Power | [LC 50](https://leetcode.com/problems/powx-n/) |
| 6 | Sqrt(x) | Easy | Binary Search | [LC 69](https://leetcode.com/problems/sqrtx/) |
| 7 | Unique Paths | Medium | Combinatorics/DP | [LC 62](https://leetcode.com/problems/unique-paths/) |
| 8 | Largest Number After Digit Swaps | Medium | Math | [LC 670](https://leetcode.com/problems/maximum-swap/) |
| 9 | Multiply Strings | Medium | Math/Simulation | [LC 43](https://leetcode.com/problems/multiply-strings/) |
| 10 | Super Pow | Medium | Modular Exp | [LC 372](https://leetcode.com/problems/super-pow/) |

# Matrix / 2D Array Patterns

Grid and matrix manipulation patterns — spiral traversal, rotation, search, and DP on grids.

---

## Core Concepts

### Matrix Setup
```python
# Create m x n matrix filled with zeros
grid = [[0] * cols for _ in range(rows)]

# Direction arrays
dirs4 = [(0,1),(0,-1),(1,0),(-1,0)]
dirs8 = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]

# Bounds check
def in_bounds(r, c, rows, cols):
    return 0 <= r < rows and 0 <= c < cols
```

---

## Pattern 1: Spiral Traversal

**C++**
```cpp
vector<int> spiralOrder(vector<vector<int>>& matrix) {
    vector<int> result;
    if (matrix.empty()) return result;
    int top = 0, bottom = matrix.size() - 1;
    int left = 0, right = matrix[0].size() - 1;

    while (top <= bottom && left <= right) {
        for (int c = left; c <= right; c++) result.push_back(matrix[top][c]);
        top++;
        for (int r = top; r <= bottom; r++) result.push_back(matrix[r][right]);
        right--;
        if (top <= bottom) {
            for (int c = right; c >= left; c--) result.push_back(matrix[bottom][c]);
            bottom--;
        }
        if (left <= right) {
            for (int r = bottom; r >= top; r--) result.push_back(matrix[r][left]);
            left++;
        }
    }
    return result;
}
```

**Java**
```java
public List<Integer> spiralOrder(int[][] matrix) {
    List<Integer> result = new ArrayList<>();
    int top = 0, bottom = matrix.length - 1;
    int left = 0, right = matrix[0].length - 1;

    while (top <= bottom && left <= right) {
        for (int c = left; c <= right; c++) result.add(matrix[top][c]);
        top++;
        for (int r = top; r <= bottom; r++) result.add(matrix[r][right]);
        right--;
        if (top <= bottom) {
            for (int c = right; c >= left; c--) result.add(matrix[bottom][c]);
            bottom--;
        }
        if (left <= right) {
            for (int r = bottom; r >= top; r--) result.add(matrix[r][left]);
            left++;
        }
    }
    return result;
}
```

**Python**
```python
def spiral_order(matrix):
    result = []
    top, bottom = 0, len(matrix) - 1
    left, right = 0, len(matrix[0]) - 1

    while top <= bottom and left <= right:
        for c in range(left, right + 1):
            result.append(matrix[top][c])
        top += 1
        for r in range(top, bottom + 1):
            result.append(matrix[r][right])
        right -= 1
        if top <= bottom:
            for c in range(right, left - 1, -1):
                result.append(matrix[bottom][c])
            bottom -= 1
        if left <= right:
            for r in range(bottom, top - 1, -1):
                result.append(matrix[r][left])
            left += 1
    return result
```

---

## Pattern 2: Rotate Matrix (In-Place)

### 90° Clockwise: Transpose → Reverse Each Row

**C++**
```cpp
void rotate(vector<vector<int>>& matrix) {
    int n = matrix.size();
    // Transpose
    for (int i = 0; i < n; i++)
        for (int j = i + 1; j < n; j++)
            swap(matrix[i][j], matrix[j][i]);
    // Reverse each row
    for (auto& row : matrix)
        reverse(row.begin(), row.end());
}
```

**Java**
```java
public void rotate(int[][] matrix) {
    int n = matrix.length;
    // Transpose
    for (int i = 0; i < n; i++)
        for (int j = i + 1; j < n; j++) {
            int temp = matrix[i][j];
            matrix[i][j] = matrix[j][i];
            matrix[j][i] = temp;
        }
    // Reverse each row
    for (int[] row : matrix) {
        int l = 0, r = n - 1;
        while (l < r) {
            int temp = row[l]; row[l] = row[r]; row[r] = temp;
            l++; r--;
        }
    }
}
```

**Python**
```python
def rotate(matrix):
    n = len(matrix)
    # Transpose
    for i in range(n):
        for j in range(i + 1, n):
            matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
    # Reverse each row
    for row in matrix:
        row.reverse()

# One-liner (creates new matrix):
# rotated = [list(row) for row in zip(*matrix[::-1])]
```

### Rotation Variants
```
90° CW:  Transpose → Reverse rows
90° CCW: Reverse rows → Transpose  (or Transpose → Reverse cols)
180°:    Reverse rows → Reverse each row
```

---

## Pattern 3: Search in Sorted Matrix

### Search in Row-wise & Column-wise Sorted Matrix (Staircase)

Each row sorted left-to-right, each column sorted top-to-bottom.

**C++**
```cpp
bool searchMatrix(vector<vector<int>>& matrix, int target) {
    int rows = matrix.size(), cols = matrix[0].size();
    int r = 0, c = cols - 1;  // start top-right
    while (r < rows && c >= 0) {
        if (matrix[r][c] == target) return true;
        else if (matrix[r][c] > target) c--;
        else r++;
    }
    return false;
}
```

**Java**
```java
public boolean searchMatrix(int[][] matrix, int target) {
    int r = 0, c = matrix[0].length - 1;
    while (r < matrix.length && c >= 0) {
        if (matrix[r][c] == target) return true;
        else if (matrix[r][c] > target) c--;
        else r++;
    }
    return false;
}
```

**Python**
```python
def search_matrix(matrix, target):
    r, c = 0, len(matrix[0]) - 1
    while r < len(matrix) and c >= 0:
        if matrix[r][c] == target:
            return True
        elif matrix[r][c] > target:
            c -= 1
        else:
            r += 1
    return False
```

### Search in Fully Sorted Matrix (Binary Search)

Rows sorted, first element of each row > last element of previous row.

**C++**
```cpp
bool searchMatrix(vector<vector<int>>& matrix, int target) {
    int rows = matrix.size(), cols = matrix[0].size();
    int lo = 0, hi = rows * cols - 1;
    while (lo <= hi) {
        int mid = lo + (hi - lo) / 2;
        int val = matrix[mid / cols][mid % cols];
        if (val == target) return true;
        else if (val < target) lo = mid + 1;
        else hi = mid - 1;
    }
    return false;
}
```

**Java**
```java
public boolean searchMatrix(int[][] matrix, int target) {
    int rows = matrix.length, cols = matrix[0].length;
    int lo = 0, hi = rows * cols - 1;
    while (lo <= hi) {
        int mid = lo + (hi - lo) / 2;
        int val = matrix[mid / cols][mid % cols];
        if (val == target) return true;
        else if (val < target) lo = mid + 1;
        else hi = mid - 1;
    }
    return false;
}
```

**Python**
```python
def search_sorted_matrix(matrix, target):
    rows, cols = len(matrix), len(matrix[0])
    lo, hi = 0, rows * cols - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        val = matrix[mid // cols][mid % cols]
        if val == target: return True
        elif val < target: lo = mid + 1
        else: hi = mid - 1
    return False
```

---

## Pattern 4: Set Matrix Zeroes

### O(1) Space — Use First Row/Col as Markers

**C++**
```cpp
void setZeroes(vector<vector<int>>& matrix) {
    int rows = matrix.size(), cols = matrix[0].size();
    bool firstRow = false, firstCol = false;
    for (int c = 0; c < cols; c++) if (matrix[0][c] == 0) firstRow = true;
    for (int r = 0; r < rows; r++) if (matrix[r][0] == 0) firstCol = true;
    for (int r = 1; r < rows; r++)
        for (int c = 1; c < cols; c++)
            if (matrix[r][c] == 0) { matrix[r][0] = 0; matrix[0][c] = 0; }
    for (int r = 1; r < rows; r++)
        for (int c = 1; c < cols; c++)
            if (matrix[r][0] == 0 || matrix[0][c] == 0) matrix[r][c] = 0;
    if (firstRow) for (int c = 0; c < cols; c++) matrix[0][c] = 0;
    if (firstCol) for (int r = 0; r < rows; r++) matrix[r][0] = 0;
}
```

**Java**
```java
public void setZeroes(int[][] matrix) {
    int rows = matrix.length, cols = matrix[0].length;
    boolean firstRow = false, firstCol = false;
    for (int c = 0; c < cols; c++) if (matrix[0][c] == 0) firstRow = true;
    for (int r = 0; r < rows; r++) if (matrix[r][0] == 0) firstCol = true;
    for (int r = 1; r < rows; r++)
        for (int c = 1; c < cols; c++)
            if (matrix[r][c] == 0) { matrix[r][0] = 0; matrix[0][c] = 0; }
    for (int r = 1; r < rows; r++)
        for (int c = 1; c < cols; c++)
            if (matrix[r][0] == 0 || matrix[0][c] == 0) matrix[r][c] = 0;
    if (firstRow) for (int c = 0; c < cols; c++) matrix[0][c] = 0;
    if (firstCol) for (int r = 0; r < rows; r++) matrix[r][0] = 0;
}
```

**Python**
```python
def set_zeroes(matrix):
    rows, cols = len(matrix), len(matrix[0])
    first_row_zero = any(matrix[0][c] == 0 for c in range(cols))
    first_col_zero = any(matrix[r][0] == 0 for r in range(rows))

    # Mark zeroes in first row/col
    for r in range(1, rows):
        for c in range(1, cols):
            if matrix[r][c] == 0:
                matrix[r][0] = 0
                matrix[0][c] = 0

    # Zero out based on marks
    for r in range(1, rows):
        for c in range(1, cols):
            if matrix[r][0] == 0 or matrix[0][c] == 0:
                matrix[r][c] = 0

    # Handle first row/col
    if first_row_zero:
        for c in range(cols): matrix[0][c] = 0
    if first_col_zero:
        for r in range(rows): matrix[r][0] = 0
```

---

## Pattern 5: Matrix DP

### Minimum Path Sum

**C++**
```cpp
int minPathSum(vector<vector<int>>& grid) {
    int rows = grid.size(), cols = grid[0].size();
    vector<vector<int>> dp(rows, vector<int>(cols, 0));
    dp[0][0] = grid[0][0];
    for (int r = 1; r < rows; r++) dp[r][0] = dp[r-1][0] + grid[r][0];
    for (int c = 1; c < cols; c++) dp[0][c] = dp[0][c-1] + grid[0][c];
    for (int r = 1; r < rows; r++)
        for (int c = 1; c < cols; c++)
            dp[r][c] = grid[r][c] + min(dp[r-1][c], dp[r][c-1]);
    return dp[rows-1][cols-1];
}
```

**Java**
```java
public int minPathSum(int[][] grid) {
    int rows = grid.length, cols = grid[0].length;
    int[][] dp = new int[rows][cols];
    dp[0][0] = grid[0][0];
    for (int r = 1; r < rows; r++) dp[r][0] = dp[r-1][0] + grid[r][0];
    for (int c = 1; c < cols; c++) dp[0][c] = dp[0][c-1] + grid[0][c];
    for (int r = 1; r < rows; r++)
        for (int c = 1; c < cols; c++)
            dp[r][c] = grid[r][c] + Math.min(dp[r-1][c], dp[r][c-1]);
    return dp[rows-1][cols-1];
}
```

**Python**
```python
def min_path_sum(grid):
    rows, cols = len(grid), len(grid[0])
    dp = [[0] * cols for _ in range(rows)]
    dp[0][0] = grid[0][0]

    for r in range(1, rows): dp[r][0] = dp[r-1][0] + grid[r][0]
    for c in range(1, cols): dp[0][c] = dp[0][c-1] + grid[0][c]

    for r in range(1, rows):
        for c in range(1, cols):
            dp[r][c] = grid[r][c] + min(dp[r-1][c], dp[r][c-1])

    return dp[-1][-1]
```

### Unique Paths (with Obstacles)

**C++**
```cpp
int uniquePathsWithObstacles(vector<vector<int>>& grid) {
    int rows = grid.size(), cols = grid[0].size();
    vector<vector<int>> dp(rows, vector<int>(cols, 0));
    dp[0][0] = grid[0][0] == 0 ? 1 : 0;
    for (int r = 0; r < rows; r++)
        for (int c = 0; c < cols; c++) {
            if (grid[r][c] == 1) { dp[r][c] = 0; continue; }
            if (r > 0) dp[r][c] += dp[r-1][c];
            if (c > 0) dp[r][c] += dp[r][c-1];
        }
    return dp[rows-1][cols-1];
}
```

**Java**
```java
public int uniquePathsWithObstacles(int[][] grid) {
    int rows = grid.length, cols = grid[0].length;
    int[][] dp = new int[rows][cols];
    dp[0][0] = grid[0][0] == 0 ? 1 : 0;
    for (int r = 0; r < rows; r++)
        for (int c = 0; c < cols; c++) {
            if (grid[r][c] == 1) { dp[r][c] = 0; continue; }
            if (r > 0) dp[r][c] += dp[r-1][c];
            if (c > 0) dp[r][c] += dp[r][c-1];
        }
    return dp[rows-1][cols-1];
}
```

**Python**
```python
def unique_paths_with_obstacles(grid):
    rows, cols = len(grid), len(grid[0])
    dp = [[0] * cols for _ in range(rows)]
    dp[0][0] = 1 if grid[0][0] == 0 else 0

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 1:
                dp[r][c] = 0
            else:
                if r > 0: dp[r][c] += dp[r-1][c]
                if c > 0: dp[r][c] += dp[r][c-1]
    return dp[-1][-1]
```

### Maximal Square

**C++**
```cpp
int maximalSquare(vector<vector<char>>& matrix) {
    int rows = matrix.size(), cols = matrix[0].size(), maxSide = 0;
    vector<vector<int>> dp(rows, vector<int>(cols, 0));
    for (int r = 0; r < rows; r++)
        for (int c = 0; c < cols; c++) {
            if (matrix[r][c] == '1') {
                dp[r][c] = (r == 0 || c == 0) ? 1
                    : 1 + min({dp[r-1][c], dp[r][c-1], dp[r-1][c-1]});
                maxSide = max(maxSide, dp[r][c]);
            }
        }
    return maxSide * maxSide;
}
```

**Java**
```java
public int maximalSquare(char[][] matrix) {
    int rows = matrix.length, cols = matrix[0].length, maxSide = 0;
    int[][] dp = new int[rows][cols];
    for (int r = 0; r < rows; r++)
        for (int c = 0; c < cols; c++) {
            if (matrix[r][c] == '1') {
                dp[r][c] = (r == 0 || c == 0) ? 1
                    : 1 + Math.min(dp[r-1][c-1], Math.min(dp[r-1][c], dp[r][c-1]));
                maxSide = Math.max(maxSide, dp[r][c]);
            }
        }
    return maxSide * maxSide;
}
```

**Python**
```python
def maximal_square(matrix):
    if not matrix: return 0
    rows, cols = len(matrix), len(matrix[0])
    dp = [[0] * cols for _ in range(rows)]
    max_side = 0

    for r in range(rows):
        for c in range(cols):
            if matrix[r][c] == '1':
                if r == 0 or c == 0:
                    dp[r][c] = 1
                else:
                    dp[r][c] = 1 + min(dp[r-1][c], dp[r][c-1], dp[r-1][c-1])
                max_side = max(max_side, dp[r][c])
    return max_side * max_side
```

---

## Pattern 6: Transpose & Reshape

**Python**
```python
# Transpose
def transpose(matrix):
    return [list(row) for row in zip(*matrix)]

# Reshape (flatten then refill)
def reshape(matrix, r, c):
    flat = [v for row in matrix for v in row]
    if len(flat) != r * c:
        return matrix
    return [flat[i*c:(i+1)*c] for i in range(r)]

# Diagonal traversal
def diagonal_order(matrix):
    if not matrix: return []
    rows, cols = len(matrix), len(matrix[0])
    result = []
    for d in range(rows + cols - 1):
        if d % 2 == 0:
            r, c = min(d, rows - 1), max(0, d - rows + 1)
            while r >= 0 and c < cols:
                result.append(matrix[r][c])
                r -= 1; c += 1
        else:
            r, c = max(0, d - cols + 1), min(d, cols - 1)
            while r < rows and c >= 0:
                result.append(matrix[r][c])
                r += 1; c -= 1
    return result
```

---

## Pitfalls & Edge Cases

| Pitfall | How to Handle |
|---------|---------------|
| Single row / single column | Handle boundary conditions |
| Non-square matrix (m ≠ n) | Rotation is only defined for square matrices |
| Empty matrix | Check `len(matrix) == 0` or `len(matrix[0]) == 0` |
| Mutating input during traversal | Use copy or visit-tracking if needed |
| Row-major vs column-major indexing | `matrix[row][col]`, row = `idx // cols`, col = `idx % cols` |
| Off-by-one in spiral | Use boundary checks after each direction |

---

## Practice Problems

| # | Problem | Difficulty | Pattern | Link |
|---|---------|-----------|---------|------|
| 1 | Spiral Matrix | Medium | Spiral | [LC 54](https://leetcode.com/problems/spiral-matrix/) |
| 2 | Spiral Matrix II | Medium | Spiral | [LC 59](https://leetcode.com/problems/spiral-matrix-ii/) |
| 3 | Rotate Image | Medium | In-Place Rotate | [LC 48](https://leetcode.com/problems/rotate-image/) |
| 4 | Search a 2D Matrix | Medium | Binary Search | [LC 74](https://leetcode.com/problems/search-a-2d-matrix/) |
| 5 | Search a 2D Matrix II | Medium | Staircase | [LC 240](https://leetcode.com/problems/search-a-2d-matrix-ii/) |
| 6 | Set Matrix Zeroes | Medium | In-Place Markers | [LC 73](https://leetcode.com/problems/set-matrix-zeroes/) |
| 7 | Minimum Path Sum | Medium | Grid DP | [LC 64](https://leetcode.com/problems/minimum-path-sum/) |
| 8 | Unique Paths II | Medium | Grid DP | [LC 63](https://leetcode.com/problems/unique-paths-ii/) |
| 9 | Maximal Square | Medium | Grid DP | [LC 221](https://leetcode.com/problems/maximal-square/) |
| 10 | Diagonal Traverse | Medium | Traversal | [LC 498](https://leetcode.com/problems/diagonal-traverse/) |
| 11 | Word Search | Medium | Grid DFS | [LC 79](https://leetcode.com/problems/word-search/) |
| 12 | Game of Life | Medium | In-Place Simulation | [LC 289](https://leetcode.com/problems/game-of-life/) |

# Trees & Graphs

The most versatile interview topic — covers binary trees, BSTs, n-ary trees, directed/undirected graphs, and a wide range of traversal algorithms.

---

## Core Concepts

### Binary Tree
- Each node has at most 2 children (`left`, `right`)
- **Height** — longest path from root to leaf: O(log n) balanced, O(n) worst
- **Complete** — all levels filled except possibly the last
- **Perfect** — all internal nodes have 2 children, all leaves at same level

### Binary Search Tree (BST)
- Left subtree values < node < right subtree values
- In-order traversal yields sorted order
- Operations O(log n) average, O(n) worst (skewed)

### Graph
- **Vertices (V)** + **Edges (E)**
- **Directed** vs **Undirected**, **Weighted** vs **Unweighted**
- **Representations:** adjacency list (sparse, O(V+E) space) vs adjacency matrix (dense, O(V²) space)

---

## Pattern 1: DFS (Depth-First Search)

**When to use:** Path finding, cycle detection, topological sort, connected components, tree traversals.

### Tree DFS Traversals

**C++**
```cpp
struct TreeNode {
    int val;
    TreeNode *left, *right;
    TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
};

// Preorder: Root → Left → Right
void preorder(TreeNode* root, vector<int>& result) {
    if (!root) return;
    result.push_back(root->val);
    preorder(root->left, result);
    preorder(root->right, result);
}

// Inorder: Left → Root → Right (sorted for BST)
void inorder(TreeNode* root, vector<int>& result) {
    if (!root) return;
    inorder(root->left, result);
    result.push_back(root->val);
    inorder(root->right, result);
}

// Postorder: Left → Right → Root
void postorder(TreeNode* root, vector<int>& result) {
    if (!root) return;
    postorder(root->left, result);
    postorder(root->right, result);
    result.push_back(root->val);
}
```

**Java**
```java
public class TreeNode {
    int val;
    TreeNode left, right;
    TreeNode(int val) { this.val = val; }
}

public void inorder(TreeNode root, List<Integer> result) {
    if (root == null) return;
    inorder(root.left, result);
    result.add(root.val);
    inorder(root.right, result);
}
```

**Python**
```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def inorder(root: TreeNode | None) -> list[int]:
    if not root:
        return []
    return inorder(root.left) + [root.val] + inorder(root.right)

# Iterative inorder (using stack)
def inorder_iterative(root: TreeNode | None) -> list[int]:
    result, stack = [], []
    curr = root
    while curr or stack:
        while curr:
            stack.append(curr)
            curr = curr.left
        curr = stack.pop()
        result.append(curr.val)
        curr = curr.right
    return result
```

### Graph DFS

**C++**
```cpp
void dfs(unordered_map<int, vector<int>>& graph, int node,
         unordered_set<int>& visited, vector<int>& result) {
    visited.insert(node);
    result.push_back(node);
    for (int neighbor : graph[node]) {
        if (!visited.count(neighbor))
            dfs(graph, neighbor, visited, result);
    }
}
```

**Java**
```java
public void dfs(Map<Integer, List<Integer>> graph, int node,
                Set<Integer> visited, List<Integer> result) {
    visited.add(node);
    result.add(node);
    for (int neighbor : graph.getOrDefault(node, List.of())) {
        if (!visited.contains(neighbor))
            dfs(graph, neighbor, visited, result);
    }
}
```

**Python**
```python
def dfs(graph: dict[int, list[int]], start: int) -> list[int]:
    visited = set()
    result = []

    def explore(node):
        visited.add(node)
        result.append(node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                explore(neighbor)

    explore(start)
    return result
```

---

## Pattern 2: BFS (Breadth-First Search)

**When to use:** Level-order traversal, shortest path (unweighted), nearest neighbors.

### Example: Level-Order Traversal

**C++**
```cpp
#include <queue>
#include <vector>
using namespace std;

vector<vector<int>> levelOrder(TreeNode* root) {
    vector<vector<int>> result;
    if (!root) return result;

    queue<TreeNode*> q;
    q.push(root);

    while (!q.empty()) {
        int size = q.size();
        vector<int> level;
        for (int i = 0; i < size; i++) {
            TreeNode* node = q.front(); q.pop();
            level.push_back(node->val);
            if (node->left) q.push(node->left);
            if (node->right) q.push(node->right);
        }
        result.push_back(level);
    }
    return result;
}
```

**Java**
```java
public List<List<Integer>> levelOrder(TreeNode root) {
    List<List<Integer>> result = new ArrayList<>();
    if (root == null) return result;

    Queue<TreeNode> queue = new LinkedList<>();
    queue.offer(root);

    while (!queue.isEmpty()) {
        int size = queue.size();
        List<Integer> level = new ArrayList<>();
        for (int i = 0; i < size; i++) {
            TreeNode node = queue.poll();
            level.add(node.val);
            if (node.left != null) queue.offer(node.left);
            if (node.right != null) queue.offer(node.right);
        }
        result.add(level);
    }
    return result;
}
```

**Python**
```python
from collections import deque

def level_order(root: TreeNode | None) -> list[list[int]]:
    if not root:
        return []
    result, queue = [], deque([root])
    while queue:
        level = []
        for _ in range(len(queue)):
            node = queue.popleft()
            level.append(node.val)
            if node.left:  queue.append(node.left)
            if node.right: queue.append(node.right)
        result.append(level)
    return result
```

### Graph BFS (Shortest Path)

**C++**
```cpp
int bfsShortestPath(unordered_map<int, vector<int>>& graph, int start, int end) {
    queue<pair<int, int>> q;
    unordered_set<int> visited;
    q.push({start, 0});
    visited.insert(start);

    while (!q.empty()) {
        auto [node, dist] = q.front(); q.pop();
        if (node == end) return dist;
        for (int neighbor : graph[node]) {
            if (!visited.count(neighbor)) {
                visited.insert(neighbor);
                q.push({neighbor, dist + 1});
            }
        }
    }
    return -1;
}
```

**Java**
```java
public int bfsShortestPath(Map<Integer, List<Integer>> graph, int start, int end) {
    Queue<int[]> queue = new ArrayDeque<>();
    Set<Integer> visited = new HashSet<>();
    queue.offer(new int[]{start, 0});
    visited.add(start);

    while (!queue.isEmpty()) {
        int[] curr = queue.poll();
        if (curr[0] == end) return curr[1];
        for (int neighbor : graph.getOrDefault(curr[0], List.of())) {
            if (visited.add(neighbor))
                queue.offer(new int[]{neighbor, curr[1] + 1});
        }
    }
    return -1;
}
```

**Python**
```python
from collections import deque

def bfs_shortest_path(graph: dict, start: int, end: int) -> int:
    queue = deque([(start, 0)])
    visited = {start}

    while queue:
        node, dist = queue.popleft()
        if node == end:
            return dist
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, dist + 1))
    return -1  # unreachable
```

---

## Pattern 3: Recursive Tree Problems

**When to use:** Height, diameter, path sum, LCA, validate BST, subtree problems.

### Example: Maximum Depth

**C++**
```cpp
int maxDepth(TreeNode* root) {
    if (!root) return 0;
    return 1 + max(maxDepth(root->left), maxDepth(root->right));
}
```

**Java**
```java
public int maxDepth(TreeNode root) {
    if (root == null) return 0;
    return 1 + Math.max(maxDepth(root.left), maxDepth(root.right));
}
```

**Python**
```python
def max_depth(root: TreeNode | None) -> int:
    if not root:
        return 0
    return 1 + max(max_depth(root.left), max_depth(root.right))
```

### Example: Validate BST

**C++**
```cpp
bool isValidBST(TreeNode* root, long minVal = LONG_MIN, long maxVal = LONG_MAX) {
    if (!root) return true;
    if (root->val <= minVal || root->val >= maxVal) return false;
    return isValidBST(root->left, minVal, root->val) &&
           isValidBST(root->right, root->val, maxVal);
}
```

**Java**
```java
public boolean isValidBST(TreeNode root) {
    return validate(root, Long.MIN_VALUE, Long.MAX_VALUE);
}

private boolean validate(TreeNode node, long min, long max) {
    if (node == null) return true;
    if (node.val <= min || node.val >= max) return false;
    return validate(node.left, min, node.val) &&
           validate(node.right, node.val, max);
}
```

**Python**
```python
def is_valid_bst(root: TreeNode | None,
                 lo: float = float('-inf'),
                 hi: float = float('inf')) -> bool:
    if not root:
        return True
    if root.val <= lo or root.val >= hi:
        return False
    return (is_valid_bst(root.left, lo, root.val) and
            is_valid_bst(root.right, root.val, hi))
```

### Example: Lowest Common Ancestor (BST)

**C++**
```cpp
TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
    while (root) {
        if (p->val < root->val && q->val < root->val) root = root->left;
        else if (p->val > root->val && q->val > root->val) root = root->right;
        else return root;
    }
    return nullptr;
}
```

**Java**
```java
public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
    while (root != null) {
        if (p.val < root.val && q.val < root.val) root = root.left;
        else if (p.val > root.val && q.val > root.val) root = root.right;
        else return root;
    }
    return null;
}
```

**Python**
```python
def lowest_common_ancestor(root: TreeNode, p: TreeNode, q: TreeNode) -> TreeNode:
    while root:
        if p.val < root.val and q.val < root.val:
            root = root.left
        elif p.val > root.val and q.val > root.val:
            root = root.right
        else:
            return root
```

---

## Pattern 4: Topological Sort

**When to use:** Task scheduling, course prerequisites, build dependencies — any DAG ordering.

### Example: Course Schedule (Kahn's BFS)

**C++**
```cpp
bool canFinish(int numCourses, vector<vector<int>>& prerequisites) {
    vector<vector<int>> graph(numCourses);
    vector<int> inDegree(numCourses, 0);

    for (auto& p : prerequisites) {
        graph[p[1]].push_back(p[0]);
        inDegree[p[0]]++;
    }

    queue<int> q;
    for (int i = 0; i < numCourses; i++)
        if (inDegree[i] == 0) q.push(i);

    int completed = 0;
    while (!q.empty()) {
        int node = q.front(); q.pop();
        completed++;
        for (int next : graph[node])
            if (--inDegree[next] == 0) q.push(next);
    }
    return completed == numCourses;
}
```

**Java**
```java
public boolean canFinish(int numCourses, int[][] prerequisites) {
    List<List<Integer>> graph = new ArrayList<>();
    int[] inDegree = new int[numCourses];
    for (int i = 0; i < numCourses; i++) graph.add(new ArrayList<>());

    for (int[] p : prerequisites) {
        graph.get(p[1]).add(p[0]);
        inDegree[p[0]]++;
    }

    Queue<Integer> q = new ArrayDeque<>();
    for (int i = 0; i < numCourses; i++)
        if (inDegree[i] == 0) q.offer(i);

    int completed = 0;
    while (!q.isEmpty()) {
        int node = q.poll();
        completed++;
        for (int next : graph.get(node))
            if (--inDegree[next] == 0) q.offer(next);
    }
    return completed == numCourses;
}
```

**Python**
```python
from collections import deque, defaultdict

def can_finish(num_courses: int, prerequisites: list[list[int]]) -> bool:
    graph = defaultdict(list)
    in_degree = [0] * num_courses

    for course, prereq in prerequisites:
        graph[prereq].append(course)
        in_degree[course] += 1

    queue = deque(i for i in range(num_courses) if in_degree[i] == 0)
    completed = 0

    while queue:
        node = queue.popleft()
        completed += 1
        for neighbor in graph[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    return completed == num_courses
```

---

## Pattern 5: Union-Find (Disjoint Set)

**When to use:** Connected components, cycle detection in undirected graphs, Kruskal's MST.

**C++**
```cpp
class UnionFind {
public:
    vector<int> parent, rank_;
    int components;

    UnionFind(int n) : parent(n), rank_(n, 0), components(n) {
        iota(parent.begin(), parent.end(), 0);
    }

    int find(int x) {
        if (parent[x] != x) parent[x] = find(parent[x]);
        return parent[x];
    }

    bool unite(int x, int y) {
        int px = find(x), py = find(y);
        if (px == py) return false;
        if (rank_[px] < rank_[py]) swap(px, py);
        parent[py] = px;
        if (rank_[px] == rank_[py]) rank_[px]++;
        components--;
        return true;
    }
};
```

**Java**
```java
class UnionFind {
    int[] parent, rank;
    int components;

    UnionFind(int n) {
        parent = new int[n];
        rank = new int[n];
        components = n;
        for (int i = 0; i < n; i++) parent[i] = i;
    }

    int find(int x) {
        if (parent[x] != x) parent[x] = find(parent[x]);
        return parent[x];
    }

    boolean union(int x, int y) {
        int px = find(x), py = find(y);
        if (px == py) return false;
        if (rank[px] < rank[py]) { int t = px; px = py; py = t; }
        parent[py] = px;
        if (rank[px] == rank[py]) rank[px]++;
        components--;
        return true;
    }
}
```

**Python**
```python
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.components = n

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # path compression
        return self.parent[x]

    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px == py:
            return False
        # union by rank
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1
        self.components -= 1
        return True
```

---

## Pitfalls & Edge Cases

| Pitfall | How to Handle |
|---------|--------------|
| Null/empty tree | Always check `root == null` |
| Single-node tree | Valid tree — height = 0 or 1 (clarify definition) |
| Skewed tree (all left/right) | Degenerates to linked list — O(n) height |
| Disconnected graph | Loop over all nodes, not just start |
| Self-loops and parallel edges | Check problem constraints carefully |
| BST with duplicate values | Clarify: left < root ≤ right? |
| Infinite loops in graph DFS/BFS | Always use `visited` set |

---

## Practice Problems

| # | Problem | Difficulty | Pattern | Link |
|---|---------|-----------|---------|------|
| 1 | Maximum Depth of Binary Tree | Easy | Recursion | [LeetCode 104](https://leetcode.com/problems/maximum-depth-of-binary-tree/) |
| 2 | Invert Binary Tree | Easy | DFS | [LeetCode 226](https://leetcode.com/problems/invert-binary-tree/) |
| 3 | Same Tree | Easy | DFS | [LeetCode 100](https://leetcode.com/problems/same-tree/) |
| 4 | Validate BST | Medium | Recursion + Bounds | [LeetCode 98](https://leetcode.com/problems/validate-binary-search-tree/) |
| 5 | Level Order Traversal | Medium | BFS | [LeetCode 102](https://leetcode.com/problems/binary-tree-level-order-traversal/) |
| 6 | Course Schedule | Medium | Topological Sort | [LeetCode 207](https://leetcode.com/problems/course-schedule/) |
| 7 | Number of Islands | Medium | DFS/BFS | [LeetCode 200](https://leetcode.com/problems/number-of-islands/) |
| 8 | Lowest Common Ancestor | Medium | Recursion | [LeetCode 236](https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-tree/) |
| 9 | Word Ladder | Hard | BFS | [LeetCode 127](https://leetcode.com/problems/word-ladder/) |
| 10 | Serialize/Deserialize Tree | Hard | DFS/BFS | [LeetCode 297](https://leetcode.com/problems/serialize-and-deserialize-binary-tree/) |

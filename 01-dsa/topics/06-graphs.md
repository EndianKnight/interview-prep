# Graphs

Directed & undirected graphs, weighted & unweighted — covering traversals, shortest paths, connectivity, and advanced algorithms for interviews.

---

## Core Concepts

### Representations

```
Adjacency List (preferred for sparse graphs):
  0: [1, 2]
  1: [0, 3]
  2: [0]
  3: [1]

Adjacency Matrix (preferred for dense graphs):
  [[0,1,1,0],
   [1,0,0,1],
   [1,0,0,0],
   [0,1,0,0]]
```

| Representation | Space | Check Edge | Iterate Neighbors |
|---------------|-------|------------|-------------------|
| Adjacency List | O(V + E) | O(degree) | O(degree) |
| Adjacency Matrix | O(V²) | O(1) | O(V) |

### Terminology
- **Directed** vs **Undirected**
- **Weighted** vs **Unweighted**
- **Cyclic** vs **Acyclic** (DAG = Directed Acyclic Graph)
- **Connected** (undirected) / **Strongly Connected** (directed)
- **Degree** — number of edges (in-degree, out-degree for directed)
- **Dense**: E ≈ V², **Sparse**: E ≈ V

---

## Pattern 1: DFS (Depth-First Search)

**When to use:** Path finding, cycle detection, connected components, topological sort, flood fill.

**Time:** O(V + E) | **Space:** O(V)

### Recursive DFS

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

### Iterative DFS (Stack)

**C++**
```cpp
vector<int> dfsIterative(unordered_map<int, vector<int>>& graph, int start) {
    vector<int> result;
    unordered_set<int> visited;
    stack<int> stk;
    stk.push(start);
    while (!stk.empty()) {
        int node = stk.top(); stk.pop();
        if (visited.count(node)) continue;
        visited.insert(node);
        result.push_back(node);
        for (int neighbor : graph[node])
            if (!visited.count(neighbor))
                stk.push(neighbor);
    }
    return result;
}
```

**Java**
```java
public List<Integer> dfsIterative(Map<Integer, List<Integer>> graph, int start) {
    List<Integer> result = new ArrayList<>();
    Set<Integer> visited = new HashSet<>();
    Deque<Integer> stack = new ArrayDeque<>();
    stack.push(start);
    while (!stack.isEmpty()) {
        int node = stack.pop();
        if (!visited.add(node)) continue;
        result.add(node);
        for (int neighbor : graph.getOrDefault(node, List.of()))
            if (!visited.contains(neighbor))
                stack.push(neighbor);
    }
    return result;
}
```

**Python**
```python
def dfs_iterative(graph, start):
    visited = set()
    stack = [start]
    result = []
    while stack:
        node = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        result.append(node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                stack.append(neighbor)
    return result
```

### Grid DFS (Flood Fill / Number of Islands)

**C++**
```cpp
int numIslands(vector<vector<char>>& grid) {
    int count = 0;
    int rows = grid.size(), cols = grid[0].size();
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            if (grid[r][c] == '1') {
                count++;
                dfs(grid, r, c, rows, cols);
            }
        }
    }
    return count;
}

void dfs(vector<vector<char>>& grid, int r, int c, int rows, int cols) {
    if (r < 0 || r >= rows || c < 0 || c >= cols || grid[r][c] != '1') return;
    grid[r][c] = '0';
    int dirs[] = {0, 1, 0, -1, 0};
    for (int d = 0; d < 4; d++)
        dfs(grid, r + dirs[d], c + dirs[d+1], rows, cols);
}
```

**Java**
```java
public int numIslands(char[][] grid) {
    int count = 0;
    for (int r = 0; r < grid.length; r++) {
        for (int c = 0; c < grid[0].length; c++) {
            if (grid[r][c] == '1') {
                count++;
                dfs(grid, r, c);
            }
        }
    }
    return count;
}

private void dfs(char[][] grid, int r, int c) {
    if (r < 0 || r >= grid.length || c < 0 || c >= grid[0].length || grid[r][c] != '1') return;
    grid[r][c] = '0';
    dfs(grid, r+1, c); dfs(grid, r-1, c);
    dfs(grid, r, c+1); dfs(grid, r, c-1);
}
```

**Python**
```python
def num_islands(grid):
    rows, cols = len(grid), len(grid[0])
    count = 0

    def dfs(r, c):
        if r < 0 or r >= rows or c < 0 or c >= cols or grid[r][c] != '1':
            return
        grid[r][c] = '0'
        for dr, dc in [(0,1),(0,-1),(1,0),(-1,0)]:
            dfs(r + dr, c + dc)

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == '1':
                count += 1
                dfs(r, c)
    return count
```

---

## Pattern 2: BFS (Breadth-First Search)

**When to use:** Shortest path (unweighted), level-by-level exploration, nearest neighbor.

**Time:** O(V + E) | **Space:** O(V)

### Shortest Path (Unweighted)

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

def bfs_shortest_path(graph, start, end):
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
    return -1
```

### Multi-Source BFS (e.g. Rotten Oranges)

**C++**
```cpp
int orangesRotting(vector<vector<int>>& grid) {
    int rows = grid.size(), cols = grid[0].size(), fresh = 0;
    queue<tuple<int,int,int>> q;
    for (int r = 0; r < rows; r++)
        for (int c = 0; c < cols; c++)
            if (grid[r][c] == 2) q.push({r, c, 0});
            else if (grid[r][c] == 1) fresh++;

    int time = 0;
    int dirs[] = {0, 1, 0, -1, 0};
    while (!q.empty() && fresh) {
        auto [r, c, t] = q.front(); q.pop();
        for (int d = 0; d < 4; d++) {
            int nr = r + dirs[d], nc = c + dirs[d+1];
            if (nr >= 0 && nr < rows && nc >= 0 && nc < cols && grid[nr][nc] == 1) {
                grid[nr][nc] = 2;
                fresh--;
                time = t + 1;
                q.push({nr, nc, t + 1});
            }
        }
    }
    return fresh == 0 ? time : -1;
}
```

**Java**
```java
public int orangesRotting(int[][] grid) {
    int rows = grid.length, cols = grid[0].length, fresh = 0;
    Queue<int[]> queue = new ArrayDeque<>();
    for (int r = 0; r < rows; r++)
        for (int c = 0; c < cols; c++)
            if (grid[r][c] == 2) queue.offer(new int[]{r, c, 0});
            else if (grid[r][c] == 1) fresh++;

    int time = 0;
    int[][] dirs = {{0,1},{0,-1},{1,0},{-1,0}};
    while (!queue.isEmpty() && fresh > 0) {
        int[] curr = queue.poll();
        for (int[] d : dirs) {
            int nr = curr[0] + d[0], nc = curr[1] + d[1];
            if (nr >= 0 && nr < rows && nc >= 0 && nc < cols && grid[nr][nc] == 1) {
                grid[nr][nc] = 2;
                fresh--;
                time = curr[2] + 1;
                queue.offer(new int[]{nr, nc, curr[2] + 1});
            }
        }
    }
    return fresh == 0 ? time : -1;
}
```

**Python**
```python
def oranges_rotting(grid):
    rows, cols = len(grid), len(grid[0])
    queue = deque()
    fresh = 0

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 2:
                queue.append((r, c, 0))
            elif grid[r][c] == 1:
                fresh += 1

    time = 0
    while queue and fresh:
        r, c, t = queue.popleft()
        for dr, dc in [(0,1),(0,-1),(1,0),(-1,0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == 1:
                grid[nr][nc] = 2
                fresh -= 1
                time = t + 1
                queue.append((nr, nc, t + 1))

    return time if fresh == 0 else -1
```

### 0-1 BFS (Deque — edges with weights 0 or 1)

**Python**
```python
def shortest_path_01(graph, start, n):
    """graph[u] = [(v, weight)] where weight is 0 or 1"""
    dist = [float('inf')] * n
    dist[start] = 0
    dq = deque([start])

    while dq:
        u = dq.popleft()
        for v, w in graph[u]:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                if w == 0:
                    dq.appendleft(v)  # 0-weight → front
                else:
                    dq.append(v)      # 1-weight → back
    return dist
```

---

## Pattern 3: Topological Sort

**When to use:** Task scheduling, course prerequisites, build dependencies — any DAG ordering.

### Kahn's Algorithm (BFS)

**C++**
```cpp
vector<int> topologicalSort(int n, vector<vector<int>>& edges) {
    vector<vector<int>> graph(n);
    vector<int> inDegree(n, 0);

    for (auto& e : edges) {
        graph[e[0]].push_back(e[1]);
        inDegree[e[1]]++;
    }

    queue<int> q;
    for (int i = 0; i < n; i++)
        if (inDegree[i] == 0) q.push(i);

    vector<int> order;
    while (!q.empty()) {
        int node = q.front(); q.pop();
        order.push_back(node);
        for (int next : graph[node])
            if (--inDegree[next] == 0) q.push(next);
    }

    return order.size() == n ? order : vector<int>();  // empty = cycle
}
```

**Java**
```java
public int[] topologicalSort(int n, int[][] edges) {
    List<List<Integer>> graph = new ArrayList<>();
    int[] inDegree = new int[n];
    for (int i = 0; i < n; i++) graph.add(new ArrayList<>());

    for (int[] e : edges) {
        graph.get(e[0]).add(e[1]);
        inDegree[e[1]]++;
    }

    Queue<Integer> q = new ArrayDeque<>();
    for (int i = 0; i < n; i++)
        if (inDegree[i] == 0) q.offer(i);

    int[] order = new int[n];
    int idx = 0;
    while (!q.isEmpty()) {
        int node = q.poll();
        order[idx++] = node;
        for (int next : graph.get(node))
            if (--inDegree[next] == 0) q.offer(next);
    }
    return idx == n ? order : new int[0];
}
```

**Python**
```python
from collections import deque, defaultdict

def topological_sort(n, edges):
    graph = defaultdict(list)
    in_degree = [0] * n

    for u, v in edges:
        graph[u].append(v)
        in_degree[v] += 1

    queue = deque(i for i in range(n) if in_degree[i] == 0)
    order = []

    while queue:
        node = queue.popleft()
        order.append(node)
        for neighbor in graph[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    return order if len(order) == n else []  # empty = cycle
```

### DFS-Based Topological Sort

**Python**
```python
def topo_sort_dfs(n, edges):
    graph = defaultdict(list)
    for u, v in edges:
        graph[u].append(v)

    WHITE, GRAY, BLACK = 0, 1, 2
    color = [WHITE] * n
    order = []
    has_cycle = False

    def dfs(node):
        nonlocal has_cycle
        color[node] = GRAY
        for neighbor in graph[node]:
            if color[neighbor] == GRAY:
                has_cycle = True
                return
            if color[neighbor] == WHITE:
                dfs(neighbor)
        color[node] = BLACK
        order.append(node)

    for i in range(n):
        if color[i] == WHITE:
            dfs(i)

    return order[::-1] if not has_cycle else []
```

---

## Pattern 4: Union-Find (Disjoint Set)

**When to use:** Connected components, cycle detection in undirected graphs, Kruskal's MST, dynamic connectivity.

**Time:** O(α(n)) per operation ≈ O(1) amortized

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
        if (parent[x] != x) parent[x] = find(parent[x]);  // path compression
        return parent[x];
    }

    bool unite(int x, int y) {
        int px = find(x), py = find(y);
        if (px == py) return false;
        if (rank_[px] < rank_[py]) swap(px, py);   // union by rank
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
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px == py:
            return False
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1
        self.components -= 1
        return True
```

---

## Pattern 5: Cycle Detection

### Undirected Graph — DFS

**C++**
```cpp
bool hasCycleUndirected(vector<vector<int>>& graph, int n) {
    vector<bool> visited(n, false);
    function<bool(int, int)> dfs = [&](int node, int parent) -> bool {
        visited[node] = true;
        for (int neighbor : graph[node]) {
            if (!visited[neighbor]) {
                if (dfs(neighbor, node)) return true;
            } else if (neighbor != parent) return true;
        }
        return false;
    };
    for (int i = 0; i < n; i++)
        if (!visited[i] && dfs(i, -1)) return true;
    return false;
}
```

**Java**
```java
public boolean hasCycleUndirected(List<List<Integer>> graph, int n) {
    boolean[] visited = new boolean[n];
    for (int i = 0; i < n; i++)
        if (!visited[i] && dfs(graph, i, -1, visited)) return true;
    return false;
}
private boolean dfs(List<List<Integer>> graph, int node, int parent, boolean[] visited) {
    visited[node] = true;
    for (int neighbor : graph.get(node)) {
        if (!visited[neighbor]) {
            if (dfs(graph, neighbor, node, visited)) return true;
        } else if (neighbor != parent) return true;
    }
    return false;
}
```

**Python**
```python
def has_cycle_undirected(graph, n):
    visited = set()

    def dfs(node, parent):
        visited.add(node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                if dfs(neighbor, node):
                    return True
            elif neighbor != parent:
                return True
        return False

    for i in range(n):
        if i not in visited:
            if dfs(i, -1):
                return True
    return False
```

### Directed Graph — DFS (Three-Color)

**C++**
```cpp
bool hasCycleDirected(vector<vector<int>>& graph, int n) {
    vector<int> color(n, 0);  // 0=WHITE, 1=GRAY, 2=BLACK
    function<bool(int)> dfs = [&](int node) -> bool {
        color[node] = 1;
        for (int neighbor : graph[node]) {
            if (color[neighbor] == 1) return true;
            if (color[neighbor] == 0 && dfs(neighbor)) return true;
        }
        color[node] = 2;
        return false;
    };
    for (int i = 0; i < n; i++)
        if (color[i] == 0 && dfs(i)) return true;
    return false;
}
```

**Java**
```java
public boolean hasCycleDirected(List<List<Integer>> graph, int n) {
    int[] color = new int[n]; // 0=WHITE, 1=GRAY, 2=BLACK
    for (int i = 0; i < n; i++)
        if (color[i] == 0 && dfsCycle(graph, i, color)) return true;
    return false;
}
private boolean dfsCycle(List<List<Integer>> graph, int node, int[] color) {
    color[node] = 1;
    for (int neighbor : graph.get(node)) {
        if (color[neighbor] == 1) return true;
        if (color[neighbor] == 0 && dfsCycle(graph, neighbor, color)) return true;
    }
    color[node] = 2;
    return false;
}
```

**Python**
```python
def has_cycle_directed(graph, n):
    WHITE, GRAY, BLACK = 0, 1, 2
    color = [WHITE] * n

    def dfs(node):
        color[node] = GRAY
        for neighbor in graph[node]:
            if color[neighbor] == GRAY:
                return True   # back edge → cycle
            if color[neighbor] == WHITE and dfs(neighbor):
                return True
        color[node] = BLACK
        return False

    return any(color[i] == WHITE and dfs(i) for i in range(n))
```

---

## Pattern 6: Bipartite Check

**When to use:** Two-coloring, odd-cycle detection.

**C++**
```cpp
bool isBipartite(vector<vector<int>>& graph, int n) {
    vector<int> color(n, -1);
    for (int i = 0; i < n; i++) {
        if (color[i] != -1) continue;
        queue<int> q;
        q.push(i);
        color[i] = 0;
        while (!q.empty()) {
            int node = q.front(); q.pop();
            for (int neighbor : graph[node]) {
                if (color[neighbor] == -1) {
                    color[neighbor] = 1 - color[node];
                    q.push(neighbor);
                } else if (color[neighbor] == color[node]) return false;
            }
        }
    }
    return true;
}
```

**Java**
```java
public boolean isBipartite(int[][] graph) {
    int n = graph.length;
    int[] color = new int[n];
    Arrays.fill(color, -1);
    for (int i = 0; i < n; i++) {
        if (color[i] != -1) continue;
        Queue<Integer> queue = new ArrayDeque<>();
        queue.offer(i);
        color[i] = 0;
        while (!queue.isEmpty()) {
            int node = queue.poll();
            for (int neighbor : graph[node]) {
                if (color[neighbor] == -1) {
                    color[neighbor] = 1 - color[node];
                    queue.offer(neighbor);
                } else if (color[neighbor] == color[node]) return false;
            }
        }
    }
    return true;
}
```

**Python**
```python
def is_bipartite(graph, n):
    color = [-1] * n

    def bfs(start):
        queue = deque([start])
        color[start] = 0
        while queue:
            node = queue.popleft()
            for neighbor in graph[node]:
                if color[neighbor] == -1:
                    color[neighbor] = 1 - color[node]
                    queue.append(neighbor)
                elif color[neighbor] == color[node]:
                    return False
        return True

    return all(color[i] != -1 or bfs(i) for i in range(n))
```

---

## Pattern 7: Shortest Path — Weighted Graphs

### Dijkstra's Algorithm (Non-Negative Weights)

**Time:** O((V + E) log V) with min-heap

**C++**
```cpp
vector<int> dijkstra(vector<vector<pair<int,int>>>& graph, int src, int n) {
    vector<int> dist(n, INT_MAX);
    priority_queue<pair<int,int>, vector<pair<int,int>>, greater<>> pq;
    dist[src] = 0;
    pq.push({0, src});

    while (!pq.empty()) {
        auto [d, u] = pq.top(); pq.pop();
        if (d > dist[u]) continue;  // skip stale entries
        for (auto [v, w] : graph[u]) {
            if (dist[u] + w < dist[v]) {
                dist[v] = dist[u] + w;
                pq.push({dist[v], v});
            }
        }
    }
    return dist;
}
```

**Java**
```java
public int[] dijkstra(List<List<int[]>> graph, int src, int n) {
    int[] dist = new int[n];
    Arrays.fill(dist, Integer.MAX_VALUE);
    dist[src] = 0;
    // min-heap: [distance, node]
    PriorityQueue<int[]> pq = new PriorityQueue<>((a, b) -> a[0] - b[0]);
    pq.offer(new int[]{0, src});

    while (!pq.isEmpty()) {
        int[] curr = pq.poll();
        int d = curr[0], u = curr[1];
        if (d > dist[u]) continue;
        for (int[] edge : graph.get(u)) {
            int v = edge[0], w = edge[1];
            if (dist[u] + w < dist[v]) {
                dist[v] = dist[u] + w;
                pq.offer(new int[]{dist[v], v});
            }
        }
    }
    return dist;
}
```

**Python**
```python
import heapq

def dijkstra(graph, src, n):
    """graph[u] = [(v, weight), ...]"""
    dist = [float('inf')] * n
    dist[src] = 0
    heap = [(0, src)]

    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]:
            continue
        for v, w in graph[u]:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                heapq.heappush(heap, (dist[v], v))
    return dist
```

### Bellman-Ford (Handles Negative Weights)

**Time:** O(V × E) — also detects negative cycles.

**C++**
```cpp
vector<int> bellmanFord(vector<tuple<int,int,int>>& edges, int src, int n) {
    vector<int> dist(n, INT_MAX);
    dist[src] = 0;
    for (int i = 0; i < n - 1; i++)
        for (auto& [u, v, w] : edges)
            if (dist[u] != INT_MAX && dist[u] + w < dist[v])
                dist[v] = dist[u] + w;
    // Check for negative cycles
    for (auto& [u, v, w] : edges)
        if (dist[u] != INT_MAX && dist[u] + w < dist[v])
            return {};  // negative cycle
    return dist;
}
```

**Java**
```java
public int[] bellmanFord(int[][] edges, int src, int n) {
    int[] dist = new int[n];
    Arrays.fill(dist, Integer.MAX_VALUE);
    dist[src] = 0;
    for (int i = 0; i < n - 1; i++)
        for (int[] e : edges)
            if (dist[e[0]] != Integer.MAX_VALUE && dist[e[0]] + e[2] < dist[e[1]])
                dist[e[1]] = dist[e[0]] + e[2];
    for (int[] e : edges)
        if (dist[e[0]] != Integer.MAX_VALUE && dist[e[0]] + e[2] < dist[e[1]])
            return new int[0];
    return dist;
}
```

**Python**
```python
def bellman_ford(edges, src, n):
    """edges = [(u, v, weight), ...]"""
    dist = [float('inf')] * n
    dist[src] = 0

    for _ in range(n - 1):
        for u, v, w in edges:
            if dist[u] != float('inf') and dist[u] + w < dist[v]:
                dist[v] = dist[u] + w

    # Check for negative cycles
    for u, v, w in edges:
        if dist[u] != float('inf') and dist[u] + w < dist[v]:
            return None  # negative cycle detected

    return dist
```

### Floyd-Warshall (All-Pairs Shortest Path)

**Time:** O(V³) | **Space:** O(V²)

**Python**
```python
def floyd_warshall(n, edges):
    dist = [[float('inf')] * n for _ in range(n)]
    for i in range(n):
        dist[i][i] = 0
    for u, v, w in edges:
        dist[u][v] = w

    for k in range(n):
        for i in range(n):
            for j in range(n):
                dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])
    return dist
```

---

## Pattern 8: Minimum Spanning Tree

### Kruskal's Algorithm (Union-Find)

**Time:** O(E log E) — sort edges, then union-find.

**C++**
```cpp
int kruskal(int n, vector<tuple<int,int,int>>& edges) {
    // edges: (weight, u, v)
    sort(edges.begin(), edges.end());
    UnionFind uf(n);
    int mstWeight = 0, mstEdges = 0;
    for (auto& [w, u, v] : edges) {
        if (uf.unite(u, v)) {
            mstWeight += w;
            if (++mstEdges == n - 1) break;
        }
    }
    return mstEdges == n - 1 ? mstWeight : -1;
}
```

**Java**
```java
public int kruskal(int n, int[][] edges) {
    // edges[i] = {weight, u, v}
    Arrays.sort(edges, (a, b) -> a[0] - b[0]);
    UnionFind uf = new UnionFind(n);
    int mstWeight = 0, mstEdges = 0;
    for (int[] e : edges) {
        if (uf.union(e[1], e[2])) {
            mstWeight += e[0];
            if (++mstEdges == n - 1) break;
        }
    }
    return mstEdges == n - 1 ? mstWeight : -1;
}
```

**Python**
```python
def kruskal(n, edges):
    """edges = [(weight, u, v), ...] — returns MST weight"""
    edges.sort()
    uf = UnionFind(n)
    mst_weight = 0
    mst_edges = 0

    for w, u, v in edges:
        if uf.union(u, v):
            mst_weight += w
            mst_edges += 1
            if mst_edges == n - 1:
                break

    return mst_weight if mst_edges == n - 1 else -1  # -1 = not connected
```

### Prim's Algorithm (Heap)

**Time:** O(E log V)

**Python**
```python
import heapq

def prim(graph, n):
    """graph[u] = [(v, weight), ...]"""
    visited = set()
    heap = [(0, 0)]  # (weight, node) — start from node 0
    mst_weight = 0

    while heap and len(visited) < n:
        w, u = heapq.heappop(heap)
        if u in visited:
            continue
        visited.add(u)
        mst_weight += w
        for v, weight in graph[u]:
            if v not in visited:
                heapq.heappush(heap, (weight, v))

    return mst_weight if len(visited) == n else -1
```

---

## Pattern 9: Strongly Connected Components

### Kosaraju's Algorithm

**Time:** O(V + E)

**Python**
```python
def kosaraju(n, graph):
    # Step 1: Fill order by finish time (DFS on original graph)
    visited = set()
    order = []

    def dfs1(node):
        visited.add(node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                dfs1(neighbor)
        order.append(node)

    for i in range(n):
        if i not in visited:
            dfs1(i)

    # Step 2: Build reverse graph
    rev_graph = defaultdict(list)
    for u in graph:
        for v in graph[u]:
            rev_graph[v].append(u)

    # Step 3: DFS on reverse graph in reverse finish order
    visited.clear()
    sccs = []

    def dfs2(node, component):
        visited.add(node)
        component.append(node)
        for neighbor in rev_graph[node]:
            if neighbor not in visited:
                dfs2(neighbor, component)

    for node in reversed(order):
        if node not in visited:
            component = []
            dfs2(node, component)
            sccs.append(component)

    return sccs
```

### Tarjan's Algorithm

**Python**
```python
def tarjan(n, graph):
    disc = [-1] * n
    low = [-1] * n
    on_stack = [False] * n
    stack = []
    timer = [0]
    sccs = []

    def dfs(u):
        disc[u] = low[u] = timer[0]
        timer[0] += 1
        stack.append(u)
        on_stack[u] = True

        for v in graph[u]:
            if disc[v] == -1:
                dfs(v)
                low[u] = min(low[u], low[v])
            elif on_stack[v]:
                low[u] = min(low[u], disc[v])

        if low[u] == disc[u]:
            component = []
            while True:
                v = stack.pop()
                on_stack[v] = False
                component.append(v)
                if v == u:
                    break
            sccs.append(component)

    for i in range(n):
        if disc[i] == -1:
            dfs(i)

    return sccs
```

---

## Pattern 10: Other Advanced Patterns

### Clone Graph

**Python**
```python
def clone_graph(node):
    if not node:
        return None
    clones = {}

    def dfs(n):
        if n in clones:
            return clones[n]
        clone = Node(n.val)
        clones[n] = clone
        for neighbor in n.neighbors:
            clone.neighbors.append(dfs(neighbor))
        return clone

    return dfs(node)
```

### Word Ladder (BFS + Implicit Graph)

**Python**
```python
def ladder_length(begin_word, end_word, word_list):
    word_set = set(word_list)
    if end_word not in word_set:
        return 0
    queue = deque([(begin_word, 1)])
    visited = {begin_word}

    while queue:
        word, steps = queue.popleft()
        for i in range(len(word)):
            for c in 'abcdefghijklmnopqrstuvwxyz':
                new_word = word[:i] + c + word[i+1:]
                if new_word == end_word:
                    return steps + 1
                if new_word in word_set and new_word not in visited:
                    visited.add(new_word)
                    queue.append((new_word, steps + 1))
    return 0
```

### Alien Dictionary (Topological Sort from Constraints)

**Python**
```python
def alien_order(words):
    graph = defaultdict(set)
    in_degree = {c: 0 for word in words for c in word}

    for i in range(len(words) - 1):
        w1, w2 = words[i], words[i + 1]
        min_len = min(len(w1), len(w2))
        if len(w1) > len(w2) and w1[:min_len] == w2[:min_len]:
            return ""  # invalid: prefix comes after longer word
        for j in range(min_len):
            if w1[j] != w2[j]:
                if w2[j] not in graph[w1[j]]:
                    graph[w1[j]].add(w2[j])
                    in_degree[w2[j]] += 1
                break

    queue = deque(c for c in in_degree if in_degree[c] == 0)
    result = []
    while queue:
        c = queue.popleft()
        result.append(c)
        for neighbor in graph[c]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    return "".join(result) if len(result) == len(in_degree) else ""
```

---

## Algorithm Comparison

| Algorithm | Use Case | Time | Negative Weights? |
|-----------|----------|------|-------------------|
| BFS | Unweighted shortest path | O(V + E) | N/A |
| 0-1 BFS | Weights 0 or 1 only | O(V + E) | No |
| Dijkstra | Non-negative weighted | O((V+E) log V) | No |
| Bellman-Ford | Any weights, negative cycle detection | O(V × E) | Yes |
| Floyd-Warshall | All-pairs shortest path | O(V³) | Yes |
| Kruskal (MST) | Minimum spanning tree | O(E log E) | N/A |
| Prim (MST) | Minimum spanning tree (dense) | O(E log V) | N/A |
| Topological Sort | DAG ordering | O(V + E) | N/A |
| Kosaraju / Tarjan | Strongly connected components | O(V + E) | N/A |
| Union-Find | Dynamic connectivity, components | O(α(n)) per op | N/A |

---

## Pitfalls & Edge Cases

| Pitfall | How to Handle |
|---------|---------------|
| Disconnected graph | Loop over all nodes, not just start |
| Self-loops and parallel edges | Check problem constraints carefully |
| Infinite loops in DFS/BFS | Always use `visited` set |
| Grid problems: bounds checking | Check `0 ≤ r < rows` and `0 ≤ c < cols` |
| Directed vs undirected confusion | Undirected = add edge both ways |
| Dijkstra with negative weights | Use Bellman-Ford instead |
| Topological sort on cyclic graph | Detect cycle (order.size ≠ n) |
| Node labels not 0-indexed | Use hashmap instead of array for visited |
| Implicit graphs (word transformation) | Generate neighbors on-the-fly |

---

## Practice Problems

| # | Problem | Difficulty | Pattern | Link |
|---|---------|-----------|---------|------|
| 1 | Number of Islands | Medium | DFS/BFS Grid | [LC 200](https://leetcode.com/problems/number-of-islands/) |
| 2 | Clone Graph | Medium | DFS + HashMap | [LC 133](https://leetcode.com/problems/clone-graph/) |
| 3 | Course Schedule | Medium | Topological Sort | [LC 207](https://leetcode.com/problems/course-schedule/) |
| 4 | Course Schedule II | Medium | Topological Sort | [LC 210](https://leetcode.com/problems/course-schedule-ii/) |
| 5 | Pacific Atlantic Water Flow | Medium | Multi-Source DFS | [LC 417](https://leetcode.com/problems/pacific-atlantic-water-flow/) |
| 6 | Number of Connected Components | Medium | Union-Find/DFS | [LC 323](https://leetcode.com/problems/number-of-connected-components-in-an-undirected-graph/) |
| 7 | Graph Valid Tree | Medium | Union-Find + Cycle | [LC 261](https://leetcode.com/problems/graph-valid-tree/) |
| 8 | Redundant Connection | Medium | Union-Find | [LC 684](https://leetcode.com/problems/redundant-connection/) |
| 9 | Word Ladder | Hard | BFS | [LC 127](https://leetcode.com/problems/word-ladder/) |
| 10 | Alien Dictionary | Hard | Topological Sort | [LC 269](https://leetcode.com/problems/alien-dictionary/) |
| 11 | Network Delay Time | Medium | Dijkstra | [LC 743](https://leetcode.com/problems/network-delay-time/) |
| 12 | Cheapest Flights K Stops | Medium | Bellman-Ford/BFS | [LC 787](https://leetcode.com/problems/cheapest-flights-within-k-stops/) |
| 13 | Is Graph Bipartite? | Medium | BFS/DFS Coloring | [LC 785](https://leetcode.com/problems/is-graph-bipartite/) |
| 14 | Minimum Cost to Connect All Points | Medium | MST (Prim/Kruskal) | [LC 1584](https://leetcode.com/problems/min-cost-to-connect-all-points/) |
| 15 | Rotting Oranges | Medium | Multi-Source BFS | [LC 994](https://leetcode.com/problems/rotting-oranges/) |
| 16 | Surrounded Regions | Medium | DFS/BFS Border | [LC 130](https://leetcode.com/problems/surrounded-regions/) |
| 17 | Critical Connections | Hard | Tarjan's Bridges | [LC 1192](https://leetcode.com/problems/critical-connections-in-a-network/) |

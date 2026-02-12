# Trees

Binary trees, BSTs, and n-ary trees — the most recursive topic in interviews. Master the patterns, not just the traversals.

---

## Core Concepts

### Binary Tree
- Each node has at most 2 children (`left`, `right`)
- **Height** — longest path from root to leaf: O(log n) balanced, O(n) worst
- **Complete** — all levels filled except possibly the last (fills left to right)
- **Full** — every node has 0 or 2 children
- **Perfect** — all internal nodes have 2 children, all leaves at same level
- **Balanced** — height difference between left/right subtrees ≤ 1

### Binary Search Tree (BST)
- Left subtree values < node < right subtree values
- In-order traversal yields sorted order
- Operations O(log n) average, O(n) worst (skewed)
- Self-balancing variants: AVL, Red-Black, B-Trees

### N-ary Tree
- Each node can have any number of children
- Represented as `List<Node>` children
- Trie is a special case of n-ary tree

---

## Pattern 1: DFS Traversals

**When to use:** Visiting all nodes, path-based problems, building results from subtrees.

### Three Orders (Recursive)

**C++**
```cpp
struct TreeNode {
    int val;
    TreeNode *left, *right;
    TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
};

// Preorder: Root → Left → Right (copy/serialize a tree)
void preorder(TreeNode* root, vector<int>& result) {
    if (!root) return;
    result.push_back(root->val);
    preorder(root->left, result);
    preorder(root->right, result);
}

// Inorder: Left → Root → Right (sorted order for BST)
void inorder(TreeNode* root, vector<int>& result) {
    if (!root) return;
    inorder(root->left, result);
    result.push_back(root->val);
    inorder(root->right, result);
}

// Postorder: Left → Right → Root (delete tree, evaluate expression)
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

public void preorder(TreeNode root, List<Integer> result) {
    if (root == null) return;
    result.add(root.val);
    preorder(root.left, result);
    preorder(root.right, result);
}

public void inorder(TreeNode root, List<Integer> result) {
    if (root == null) return;
    inorder(root.left, result);
    result.add(root.val);
    inorder(root.right, result);
}

public void postorder(TreeNode root, List<Integer> result) {
    if (root == null) return;
    postorder(root.left, result);
    postorder(root.right, result);
    result.add(root.val);
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
```

### Iterative Inorder (Stack)

**C++**
```cpp
vector<int> inorderIterative(TreeNode* root) {
    vector<int> result;
    stack<TreeNode*> stk;
    TreeNode* curr = root;
    while (curr || !stk.empty()) {
        while (curr) {
            stk.push(curr);
            curr = curr->left;
        }
        curr = stk.top(); stk.pop();
        result.push_back(curr->val);
        curr = curr->right;
    }
    return result;
}
```

**Java**
```java
public List<Integer> inorderIterative(TreeNode root) {
    List<Integer> result = new ArrayList<>();
    Deque<TreeNode> stack = new ArrayDeque<>();
    TreeNode curr = root;
    while (curr != null || !stack.isEmpty()) {
        while (curr != null) {
            stack.push(curr);
            curr = curr.left;
        }
        curr = stack.pop();
        result.add(curr.val);
        curr = curr.right;
    }
    return result;
}
```

**Python**
```python
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

### Morris Traversal (O(1) Space Inorder)

**Python**
```python
def morris_inorder(root: TreeNode | None) -> list[int]:
    result = []
    curr = root
    while curr:
        if not curr.left:
            result.append(curr.val)
            curr = curr.right
        else:
            pred = curr.left
            while pred.right and pred.right != curr:
                pred = pred.right
            if not pred.right:  # make thread
                pred.right = curr
                curr = curr.left
            else:               # remove thread
                pred.right = None
                result.append(curr.val)
                curr = curr.right
    return result
```

---

## Pattern 2: BFS / Level-Order Traversal

**When to use:** Level-by-level processing, shortest path in tree, zigzag traversal, right/left side view.

### Level-Order Traversal

**C++**
```cpp
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

### Zigzag Level Order

**Python**
```python
def zigzag_level_order(root: TreeNode | None) -> list[list[int]]:
    if not root:
        return []
    result, queue, left_to_right = [], deque([root]), True
    while queue:
        level = []
        for _ in range(len(queue)):
            node = queue.popleft()
            level.append(node.val)
            if node.left:  queue.append(node.left)
            if node.right: queue.append(node.right)
        result.append(level if left_to_right else level[::-1])
        left_to_right = not left_to_right
    return result
```

### Right Side View

**Python**
```python
def right_side_view(root: TreeNode | None) -> list[int]:
    if not root:
        return []
    result, queue = [], deque([root])
    while queue:
        for i in range(len(queue)):
            node = queue.popleft()
            if i == len(queue):  # wait, we need the last node per level
                pass
            if node.left:  queue.append(node.left)
            if node.right: queue.append(node.right)
        result.append(node.val)  # last node in the level
    return result
```

---

## Pattern 3: Recursive Tree Problems

**When to use:** Height, diameter, path sum, subtree checks, tree construction problems.

### Maximum Depth

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

### Diameter of Binary Tree

**C++**
```cpp
int diameterOfBinaryTree(TreeNode* root) {
    int diameter = 0;
    function<int(TreeNode*)> dfs = [&](TreeNode* node) -> int {
        if (!node) return 0;
        int left = dfs(node->left);
        int right = dfs(node->right);
        diameter = max(diameter, left + right);
        return 1 + max(left, right);
    };
    dfs(root);
    return diameter;
}
```

**Java**
```java
int diameter = 0;
public int diameterOfBinaryTree(TreeNode root) {
    dfs(root);
    return diameter;
}
private int dfs(TreeNode node) {
    if (node == null) return 0;
    int left = dfs(node.left), right = dfs(node.right);
    diameter = Math.max(diameter, left + right);
    return 1 + Math.max(left, right);
}
```

**Python**
```python
def diameter_of_binary_tree(root: TreeNode | None) -> int:
    diameter = 0
    def dfs(node):
        nonlocal diameter
        if not node:
            return 0
        left, right = dfs(node.left), dfs(node.right)
        diameter = max(diameter, left + right)
        return 1 + max(left, right)
    dfs(root)
    return diameter
```

### Path Sum (Root to Leaf)

**Python**
```python
def has_path_sum(root: TreeNode | None, target: int) -> bool:
    if not root:
        return False
    if not root.left and not root.right:
        return root.val == target
    return (has_path_sum(root.left, target - root.val) or
            has_path_sum(root.right, target - root.val))
```

### Path Sum III (Any-to-Any Downward, Prefix Sum)

**Python**
```python
from collections import defaultdict

def path_sum_iii(root: TreeNode | None, target: int) -> int:
    count = 0
    prefix = defaultdict(int)
    prefix[0] = 1

    def dfs(node, curr_sum):
        nonlocal count
        if not node:
            return
        curr_sum += node.val
        count += prefix[curr_sum - target]
        prefix[curr_sum] += 1
        dfs(node.left, curr_sum)
        dfs(node.right, curr_sum)
        prefix[curr_sum] -= 1  # backtrack

    dfs(root, 0)
    return count
```

### Invert Binary Tree

**C++**
```cpp
TreeNode* invertTree(TreeNode* root) {
    if (!root) return nullptr;
    swap(root->left, root->right);
    invertTree(root->left);
    invertTree(root->right);
    return root;
}
```

**Java**
```java
public TreeNode invertTree(TreeNode root) {
    if (root == null) return null;
    TreeNode temp = root.left;
    root.left = invertTree(root.right);
    root.right = invertTree(temp);
    return root;
}
```

**Python**
```python
def invert_tree(root: TreeNode | None) -> TreeNode | None:
    if not root:
        return None
    root.left, root.right = invert_tree(root.right), invert_tree(root.left)
    return root
```

### Symmetric Tree

**Python**
```python
def is_symmetric(root: TreeNode | None) -> bool:
    def mirror(a, b):
        if not a and not b:
            return True
        if not a or not b:
            return False
        return (a.val == b.val and
                mirror(a.left, b.right) and
                mirror(a.right, b.left))
    return mirror(root, root)
```

### Subtree of Another Tree

**Python**
```python
def is_subtree(root: TreeNode | None, sub: TreeNode | None) -> bool:
    if not root:
        return False
    if same_tree(root, sub):
        return True
    return is_subtree(root.left, sub) or is_subtree(root.right, sub)

def same_tree(a, b):
    if not a and not b: return True
    if not a or not b: return False
    return a.val == b.val and same_tree(a.left, b.left) and same_tree(a.right, b.right)
```

---

## Pattern 4: BST-Specific Problems

**When to use:** Sorted-order queries, range queries, validation, successor/predecessor.

### Validate BST

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
def is_valid_bst(root, lo=float('-inf'), hi=float('inf')):
    if not root:
        return True
    if root.val <= lo or root.val >= hi:
        return False
    return is_valid_bst(root.left, lo, root.val) and \
           is_valid_bst(root.right, root.val, hi)
```

### Lowest Common Ancestor (BST)

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
def lowest_common_ancestor(root, p, q):
    while root:
        if p.val < root.val and q.val < root.val:
            root = root.left
        elif p.val > root.val and q.val > root.val:
            root = root.right
        else:
            return root
```

### LCA of Binary Tree (not BST)

**Python**
```python
def lca(root, p, q):
    if not root or root == p or root == q:
        return root
    left = lca(root.left, p, q)
    right = lca(root.right, p, q)
    if left and right:
        return root
    return left or right
```

### Kth Smallest in BST (Inorder)

**Python**
```python
def kth_smallest(root: TreeNode, k: int) -> int:
    stack = []
    curr = root
    while curr or stack:
        while curr:
            stack.append(curr)
            curr = curr.left
        curr = stack.pop()
        k -= 1
        if k == 0:
            return curr.val
        curr = curr.right
```

### BST Insert / Delete

**Python**
```python
def insert_bst(root, val):
    if not root:
        return TreeNode(val)
    if val < root.val:
        root.left = insert_bst(root.left, val)
    else:
        root.right = insert_bst(root.right, val)
    return root

def delete_bst(root, key):
    if not root:
        return None
    if key < root.val:
        root.left = delete_bst(root.left, key)
    elif key > root.val:
        root.right = delete_bst(root.right, key)
    else:
        if not root.left: return root.right
        if not root.right: return root.left
        # find inorder successor
        succ = root.right
        while succ.left:
            succ = succ.left
        root.val = succ.val
        root.right = delete_bst(root.right, succ.val)
    return root
```

---

## Pattern 5: Tree Construction

**When to use:** Build tree from traversals, serialization/deserialization, array-to-BST.

### Build from Preorder + Inorder

**Python**
```python
def build_tree(preorder, inorder):
    if not preorder or not inorder:
        return None
    root = TreeNode(preorder[0])
    mid = inorder.index(preorder[0])
    root.left = build_tree(preorder[1:mid+1], inorder[:mid])
    root.right = build_tree(preorder[mid+1:], inorder[mid+1:])
    return root
```

### Serialize / Deserialize (Preorder)

**Python**
```python
def serialize(root):
    vals = []
    def dfs(node):
        if not node:
            vals.append('#')
            return
        vals.append(str(node.val))
        dfs(node.left)
        dfs(node.right)
    dfs(root)
    return ','.join(vals)

def deserialize(data):
    vals = iter(data.split(','))
    def dfs():
        val = next(vals)
        if val == '#':
            return None
        node = TreeNode(int(val))
        node.left = dfs()
        node.right = dfs()
        return node
    return dfs()
```

### Sorted Array to Balanced BST

**Python**
```python
def sorted_array_to_bst(nums):
    if not nums:
        return None
    mid = len(nums) // 2
    root = TreeNode(nums[mid])
    root.left = sorted_array_to_bst(nums[:mid])
    root.right = sorted_array_to_bst(nums[mid+1:])
    return root
```

---

## Pattern 6: Tree Views & Boundaries

**When to use:** Right/left side view, boundary traversal, vertical order.

### Vertical Order Traversal

**Python**
```python
from collections import defaultdict, deque

def vertical_order(root):
    if not root:
        return []
    col_map = defaultdict(list)
    queue = deque([(root, 0)])  # (node, column)
    min_col = max_col = 0

    while queue:
        node, col = queue.popleft()
        col_map[col].append(node.val)
        min_col = min(min_col, col)
        max_col = max(max_col, col)
        if node.left:  queue.append((node.left, col - 1))
        if node.right: queue.append((node.right, col + 1))

    return [col_map[c] for c in range(min_col, max_col + 1)]
```

### Binary Tree Maximum Path Sum

**Python**
```python
def max_path_sum(root):
    result = float('-inf')
    def dfs(node):
        nonlocal result
        if not node:
            return 0
        left = max(dfs(node.left), 0)   # ignore negative paths
        right = max(dfs(node.right), 0)
        result = max(result, left + right + node.val)
        return max(left, right) + node.val
    dfs(root)
    return result
```

---

## Pitfalls & Edge Cases

| Pitfall | How to Handle |
|---------|---------------|
| Null/empty tree | Always check `root == null` |
| Single-node tree | Valid tree — height = 0 or 1 (clarify definition) |
| Skewed tree (all left/right) | Degenerates to linked list — O(n) height |
| BST with duplicate values | Clarify: left < root ≤ right? |
| Confusing depth vs height | Depth = root to node, Height = node to leaf |
| Returning new tree vs modifying in-place | Clarify with interviewer |
| Integer overflow in BST validation | Use `long` or `float('inf')` bounds |

---

## Practice Problems

| # | Problem | Difficulty | Pattern | Link |
|---|---------|-----------|---------|------|
| 1 | Maximum Depth of Binary Tree | Easy | Recursion | [LC 104](https://leetcode.com/problems/maximum-depth-of-binary-tree/) |
| 2 | Invert Binary Tree | Easy | DFS | [LC 226](https://leetcode.com/problems/invert-binary-tree/) |
| 3 | Same Tree | Easy | DFS | [LC 100](https://leetcode.com/problems/same-tree/) |
| 4 | Symmetric Tree | Easy | Recursion | [LC 101](https://leetcode.com/problems/symmetric-tree/) |
| 5 | Subtree of Another Tree | Easy | DFS | [LC 572](https://leetcode.com/problems/subtree-of-another-tree/) |
| 6 | Validate BST | Medium | BST + Bounds | [LC 98](https://leetcode.com/problems/validate-binary-search-tree/) |
| 7 | Level Order Traversal | Medium | BFS | [LC 102](https://leetcode.com/problems/binary-tree-level-order-traversal/) |
| 8 | Zigzag Level Order | Medium | BFS | [LC 103](https://leetcode.com/problems/binary-tree-zigzag-level-order-traversal/) |
| 9 | Right Side View | Medium | BFS/DFS | [LC 199](https://leetcode.com/problems/binary-tree-right-side-view/) |
| 10 | Lowest Common Ancestor | Medium | Recursion | [LC 236](https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-tree/) |
| 11 | Kth Smallest in BST | Medium | BST + Inorder | [LC 230](https://leetcode.com/problems/kth-smallest-element-in-a-bst/) |
| 12 | Binary Tree from Preorder & Inorder | Medium | Construction | [LC 105](https://leetcode.com/problems/construct-binary-tree-from-preorder-and-inorder-traversal/) |
| 13 | Diameter of Binary Tree | Easy | DFS | [LC 543](https://leetcode.com/problems/diameter-of-binary-tree/) |
| 14 | Path Sum III | Medium | Prefix Sum + DFS | [LC 437](https://leetcode.com/problems/path-sum-iii/) |
| 15 | Serialize/Deserialize Tree | Hard | DFS/BFS | [LC 297](https://leetcode.com/problems/serialize-and-deserialize-binary-tree/) |
| 16 | Binary Tree Maximum Path Sum | Hard | DFS | [LC 124](https://leetcode.com/problems/binary-tree-maximum-path-sum/) |
| 17 | Vertical Order Traversal | Hard | BFS + HashMap | [LC 987](https://leetcode.com/problems/vertical-order-traversal-of-a-binary-tree/) |

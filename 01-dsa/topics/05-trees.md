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

**C++**
```cpp
vector<int> morrisInorder(TreeNode* root) {
    vector<int> result;
    TreeNode* curr = root;
    while (curr) {
        if (!curr->left) {
            result.push_back(curr->val);
            curr = curr->right;
        } else {
            TreeNode* pred = curr->left;
            while (pred->right && pred->right != curr)
                pred = pred->right;
            if (!pred->right) {
                pred->right = curr;
                curr = curr->left;
            } else {
                pred->right = nullptr;
                result.push_back(curr->val);
                curr = curr->right;
            }
        }
    }
    return result;
}
```

**Java**
```java
public List<Integer> morrisInorder(TreeNode root) {
    List<Integer> result = new ArrayList<>();
    TreeNode curr = root;
    while (curr != null) {
        if (curr.left == null) {
            result.add(curr.val);
            curr = curr.right;
        } else {
            TreeNode pred = curr.left;
            while (pred.right != null && pred.right != curr)
                pred = pred.right;
            if (pred.right == null) {
                pred.right = curr;
                curr = curr.left;
            } else {
                pred.right = null;
                result.add(curr.val);
                curr = curr.right;
            }
        }
    }
    return result;
}
```

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

**C++**
```cpp
vector<vector<int>> zigzagLevelOrder(TreeNode* root) {
    vector<vector<int>> result;
    if (!root) return result;
    queue<TreeNode*> q;
    q.push(root);
    bool leftToRight = true;
    while (!q.empty()) {
        int size = q.size();
        vector<int> level(size);
        for (int i = 0; i < size; i++) {
            TreeNode* node = q.front(); q.pop();
            int idx = leftToRight ? i : size - 1 - i;
            level[idx] = node->val;
            if (node->left) q.push(node->left);
            if (node->right) q.push(node->right);
        }
        result.push_back(level);
        leftToRight = !leftToRight;
    }
    return result;
}
```

**Java**
```java
public List<List<Integer>> zigzagLevelOrder(TreeNode root) {
    List<List<Integer>> result = new ArrayList<>();
    if (root == null) return result;
    Queue<TreeNode> queue = new LinkedList<>();
    queue.offer(root);
    boolean leftToRight = true;
    while (!queue.isEmpty()) {
        int size = queue.size();
        Integer[] level = new Integer[size];
        for (int i = 0; i < size; i++) {
            TreeNode node = queue.poll();
            int idx = leftToRight ? i : size - 1 - i;
            level[idx] = node.val;
            if (node.left != null) queue.offer(node.left);
            if (node.right != null) queue.offer(node.right);
        }
        result.add(Arrays.asList(level));
        leftToRight = !leftToRight;
    }
    return result;
}
```

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

**C++**
```cpp
vector<int> rightSideView(TreeNode* root) {
    vector<int> result;
    if (!root) return result;
    queue<TreeNode*> q;
    q.push(root);
    while (!q.empty()) {
        int size = q.size();
        for (int i = 0; i < size; i++) {
            TreeNode* node = q.front(); q.pop();
            if (i == size - 1) result.push_back(node->val);
            if (node->left) q.push(node->left);
            if (node->right) q.push(node->right);
        }
    }
    return result;
}
```

**Java**
```java
public List<Integer> rightSideView(TreeNode root) {
    List<Integer> result = new ArrayList<>();
    if (root == null) return result;
    Queue<TreeNode> queue = new LinkedList<>();
    queue.offer(root);
    while (!queue.isEmpty()) {
        int size = queue.size();
        for (int i = 0; i < size; i++) {
            TreeNode node = queue.poll();
            if (i == size - 1) result.add(node.val);
            if (node.left != null) queue.offer(node.left);
            if (node.right != null) queue.offer(node.right);
        }
    }
    return result;
}
```

**Python**
```python
def right_side_view(root: TreeNode | None) -> list[int]:
    if not root:
        return []
    result, queue = [], deque([root])
    while queue:
        size = len(queue)
        for i in range(size):
            node = queue.popleft()
            if i == size - 1:
                result.append(node.val)
            if node.left:  queue.append(node.left)
            if node.right: queue.append(node.right)
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

**C++**
```cpp
bool hasPathSum(TreeNode* root, int target) {
    if (!root) return false;
    if (!root->left && !root->right) return root->val == target;
    return hasPathSum(root->left, target - root->val) ||
           hasPathSum(root->right, target - root->val);
}
```

**Java**
```java
public boolean hasPathSum(TreeNode root, int target) {
    if (root == null) return false;
    if (root.left == null && root.right == null) return root.val == target;
    return hasPathSum(root.left, target - root.val) ||
           hasPathSum(root.right, target - root.val);
}
```

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

**C++**
```cpp
int pathSumIII(TreeNode* root, int target) {
    int count = 0;
    unordered_map<long, int> prefix;
    prefix[0] = 1;
    function<void(TreeNode*, long)> dfs = [&](TreeNode* node, long curr) {
        if (!node) return;
        curr += node->val;
        count += prefix[curr - target];
        prefix[curr]++;
        dfs(node->left, curr);
        dfs(node->right, curr);
        prefix[curr]--;
    };
    dfs(root, 0);
    return count;
}
```

**Java**
```java
int count = 0;
public int pathSum(TreeNode root, int targetSum) {
    Map<Long, Integer> prefix = new HashMap<>();
    prefix.put(0L, 1);
    dfs(root, 0L, targetSum, prefix);
    return count;
}
private void dfs(TreeNode node, long curr, int target, Map<Long, Integer> prefix) {
    if (node == null) return;
    curr += node.val;
    count += prefix.getOrDefault(curr - target, 0);
    prefix.merge(curr, 1, Integer::sum);
    dfs(node.left, curr, target, prefix);
    dfs(node.right, curr, target, prefix);
    prefix.merge(curr, -1, Integer::sum);
}
```

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

**C++**
```cpp
bool isSymmetric(TreeNode* root) {
    return mirror(root, root);
}
bool mirror(TreeNode* a, TreeNode* b) {
    if (!a && !b) return true;
    if (!a || !b) return false;
    return a->val == b->val &&
           mirror(a->left, b->right) &&
           mirror(a->right, b->left);
}
```

**Java**
```java
public boolean isSymmetric(TreeNode root) {
    return mirror(root, root);
}
private boolean mirror(TreeNode a, TreeNode b) {
    if (a == null && b == null) return true;
    if (a == null || b == null) return false;
    return a.val == b.val &&
           mirror(a.left, b.right) &&
           mirror(a.right, b.left);
}
```

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

**C++**
```cpp
bool isSubtree(TreeNode* root, TreeNode* sub) {
    if (!root) return false;
    if (sameTree(root, sub)) return true;
    return isSubtree(root->left, sub) || isSubtree(root->right, sub);
}
bool sameTree(TreeNode* a, TreeNode* b) {
    if (!a && !b) return true;
    if (!a || !b) return false;
    return a->val == b->val && sameTree(a->left, b->left) && sameTree(a->right, b->right);
}
```

**Java**
```java
public boolean isSubtree(TreeNode root, TreeNode sub) {
    if (root == null) return false;
    if (sameTree(root, sub)) return true;
    return isSubtree(root.left, sub) || isSubtree(root.right, sub);
}
private boolean sameTree(TreeNode a, TreeNode b) {
    if (a == null && b == null) return true;
    if (a == null || b == null) return false;
    return a.val == b.val && sameTree(a.left, b.left) && sameTree(a.right, b.right);
}
```

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

**C++**
```cpp
TreeNode* lca(TreeNode* root, TreeNode* p, TreeNode* q) {
    if (!root || root == p || root == q) return root;
    TreeNode* left = lca(root->left, p, q);
    TreeNode* right = lca(root->right, p, q);
    if (left && right) return root;
    return left ? left : right;
}
```

**Java**
```java
public TreeNode lca(TreeNode root, TreeNode p, TreeNode q) {
    if (root == null || root == p || root == q) return root;
    TreeNode left = lca(root.left, p, q);
    TreeNode right = lca(root.right, p, q);
    if (left != null && right != null) return root;
    return left != null ? left : right;
}
```

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

**C++**
```cpp
int kthSmallest(TreeNode* root, int k) {
    stack<TreeNode*> stk;
    TreeNode* curr = root;
    while (curr || !stk.empty()) {
        while (curr) {
            stk.push(curr);
            curr = curr->left;
        }
        curr = stk.top(); stk.pop();
        if (--k == 0) return curr->val;
        curr = curr->right;
    }
    return -1;
}
```

**Java**
```java
public int kthSmallest(TreeNode root, int k) {
    Deque<TreeNode> stack = new ArrayDeque<>();
    TreeNode curr = root;
    while (curr != null || !stack.isEmpty()) {
        while (curr != null) {
            stack.push(curr);
            curr = curr.left;
        }
        curr = stack.pop();
        if (--k == 0) return curr.val;
        curr = curr.right;
    }
    return -1;
}
```

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

**C++**
```cpp
TreeNode* insertBST(TreeNode* root, int val) {
    if (!root) return new TreeNode(val);
    if (val < root->val) root->left = insertBST(root->left, val);
    else root->right = insertBST(root->right, val);
    return root;
}

TreeNode* deleteBST(TreeNode* root, int key) {
    if (!root) return nullptr;
    if (key < root->val) root->left = deleteBST(root->left, key);
    else if (key > root->val) root->right = deleteBST(root->right, key);
    else {
        if (!root->left) return root->right;
        if (!root->right) return root->left;
        TreeNode* succ = root->right;
        while (succ->left) succ = succ->left;
        root->val = succ->val;
        root->right = deleteBST(root->right, succ->val);
    }
    return root;
}
```

**Java**
```java
public TreeNode insertBST(TreeNode root, int val) {
    if (root == null) return new TreeNode(val);
    if (val < root.val) root.left = insertBST(root.left, val);
    else root.right = insertBST(root.right, val);
    return root;
}

public TreeNode deleteBST(TreeNode root, int key) {
    if (root == null) return null;
    if (key < root.val) root.left = deleteBST(root.left, key);
    else if (key > root.val) root.right = deleteBST(root.right, key);
    else {
        if (root.left == null) return root.right;
        if (root.right == null) return root.left;
        TreeNode succ = root.right;
        while (succ.left != null) succ = succ.left;
        root.val = succ.val;
        root.right = deleteBST(root.right, succ.val);
    }
    return root;
}
```

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

**C++**
```cpp
TreeNode* buildTree(vector<int>& preorder, vector<int>& inorder) {
    unordered_map<int, int> inMap;
    for (int i = 0; i < inorder.size(); i++) inMap[inorder[i]] = i;
    int preIdx = 0;
    function<TreeNode*(int, int)> build = [&](int lo, int hi) -> TreeNode* {
        if (lo > hi) return nullptr;
        TreeNode* root = new TreeNode(preorder[preIdx++]);
        int mid = inMap[root->val];
        root->left = build(lo, mid - 1);
        root->right = build(mid + 1, hi);
        return root;
    };
    return build(0, inorder.size() - 1);
}
```

**Java**
```java
int preIdx = 0;
Map<Integer, Integer> inMap = new HashMap<>();

public TreeNode buildTree(int[] preorder, int[] inorder) {
    for (int i = 0; i < inorder.length; i++) inMap.put(inorder[i], i);
    return build(preorder, 0, inorder.length - 1);
}
private TreeNode build(int[] preorder, int lo, int hi) {
    if (lo > hi) return null;
    TreeNode root = new TreeNode(preorder[preIdx++]);
    int mid = inMap.get(root.val);
    root.left = build(preorder, lo, mid - 1);
    root.right = build(preorder, mid + 1, hi);
    return root;
}
```

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

**C++**
```cpp
string serialize(TreeNode* root) {
    if (!root) return "#";
    return to_string(root->val) + "," +
           serialize(root->left) + "," +
           serialize(root->right);
}

TreeNode* deserialize(const string& data) {
    istringstream ss(data);
    return deserializeHelper(ss);
}
TreeNode* deserializeHelper(istringstream& ss) {
    string val;
    getline(ss, val, ',');
    if (val == "#") return nullptr;
    TreeNode* node = new TreeNode(stoi(val));
    node->left = deserializeHelper(ss);
    node->right = deserializeHelper(ss);
    return node;
}
```

**Java**
```java
public String serialize(TreeNode root) {
    if (root == null) return "#";
    return root.val + "," + serialize(root.left) + "," + serialize(root.right);
}

public TreeNode deserialize(String data) {
    Queue<String> queue = new LinkedList<>(Arrays.asList(data.split(",")));
    return deserializeHelper(queue);
}
private TreeNode deserializeHelper(Queue<String> queue) {
    String val = queue.poll();
    if ("#".equals(val)) return null;
    TreeNode node = new TreeNode(Integer.parseInt(val));
    node.left = deserializeHelper(queue);
    node.right = deserializeHelper(queue);
    return node;
}
```

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

**C++**
```cpp
TreeNode* sortedArrayToBST(vector<int>& nums) {
    return build(nums, 0, nums.size() - 1);
}
TreeNode* build(vector<int>& nums, int lo, int hi) {
    if (lo > hi) return nullptr;
    int mid = lo + (hi - lo) / 2;
    TreeNode* root = new TreeNode(nums[mid]);
    root->left = build(nums, lo, mid - 1);
    root->right = build(nums, mid + 1, hi);
    return root;
}
```

**Java**
```java
public TreeNode sortedArrayToBST(int[] nums) {
    return build(nums, 0, nums.length - 1);
}
private TreeNode build(int[] nums, int lo, int hi) {
    if (lo > hi) return null;
    int mid = lo + (hi - lo) / 2;
    TreeNode root = new TreeNode(nums[mid]);
    root.left = build(nums, lo, mid - 1);
    root.right = build(nums, mid + 1, hi);
    return root;
}
```

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

**C++**
```cpp
vector<vector<int>> verticalOrder(TreeNode* root) {
    if (!root) return {};
    map<int, vector<int>> colMap;
    queue<pair<TreeNode*, int>> q;
    q.push({root, 0});
    while (!q.empty()) {
        auto [node, col] = q.front(); q.pop();
        colMap[col].push_back(node->val);
        if (node->left) q.push({node->left, col - 1});
        if (node->right) q.push({node->right, col + 1});
    }
    vector<vector<int>> result;
    for (auto& [col, vals] : colMap)
        result.push_back(vals);
    return result;
}
```

**Java**
```java
public List<List<Integer>> verticalOrder(TreeNode root) {
    if (root == null) return new ArrayList<>();
    TreeMap<Integer, List<Integer>> colMap = new TreeMap<>();
    Queue<Object[]> queue = new LinkedList<>();
    queue.offer(new Object[]{root, 0});
    while (!queue.isEmpty()) {
        Object[] pair = queue.poll();
        TreeNode node = (TreeNode) pair[0];
        int col = (int) pair[1];
        colMap.computeIfAbsent(col, k -> new ArrayList<>()).add(node.val);
        if (node.left != null) queue.offer(new Object[]{node.left, col - 1});
        if (node.right != null) queue.offer(new Object[]{node.right, col + 1});
    }
    return new ArrayList<>(colMap.values());
}
```

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

**C++**
```cpp
int maxPathSum(TreeNode* root) {
    int result = INT_MIN;
    function<int(TreeNode*)> dfs = [&](TreeNode* node) -> int {
        if (!node) return 0;
        int left = max(dfs(node->left), 0);
        int right = max(dfs(node->right), 0);
        result = max(result, left + right + node->val);
        return max(left, right) + node->val;
    };
    dfs(root);
    return result;
}
```

**Java**
```java
int result = Integer.MIN_VALUE;
public int maxPathSum(TreeNode root) {
    dfs(root);
    return result;
}
private int dfs(TreeNode node) {
    if (node == null) return 0;
    int left = Math.max(dfs(node.left), 0);
    int right = Math.max(dfs(node.right), 0);
    result = Math.max(result, left + right + node.val);
    return Math.max(left, right) + node.val;
}
```

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

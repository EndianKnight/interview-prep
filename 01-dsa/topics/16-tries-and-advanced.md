# Tries & Advanced Data Structures

Specialized structures for prefix-based search, range queries, and advanced interview problems.

---

## Trie (Prefix Tree)

### Core Concept
- Tree where each node represents a character
- Shared prefixes share the same path
- O(L) insert, search, prefix check (L = word length)
- Use cases: autocomplete, spell check, word search, IP routing

### Implementation

**C++**
```cpp
class Trie {
    struct TrieNode {
        TrieNode* children[26] = {};
        bool isEnd = false;
    };
    TrieNode* root;
public:
    Trie() : root(new TrieNode()) {}

    void insert(const string& word) {
        TrieNode* node = root;
        for (char c : word) {
            int idx = c - 'a';
            if (!node->children[idx])
                node->children[idx] = new TrieNode();
            node = node->children[idx];
        }
        node->isEnd = true;
    }

    bool search(const string& word) {
        TrieNode* node = find(word);
        return node && node->isEnd;
    }

    bool startsWith(const string& prefix) {
        return find(prefix) != nullptr;
    }

private:
    TrieNode* find(const string& word) {
        TrieNode* node = root;
        for (char c : word) {
            int idx = c - 'a';
            if (!node->children[idx]) return nullptr;
            node = node->children[idx];
        }
        return node;
    }
};
```

**Java**
```java
class Trie {
    private Trie[] children = new Trie[26];
    private boolean isEnd = false;

    public void insert(String word) {
        Trie node = this;
        for (char c : word.toCharArray()) {
            int idx = c - 'a';
            if (node.children[idx] == null)
                node.children[idx] = new Trie();
            node = node.children[idx];
        }
        node.isEnd = true;
    }

    public boolean search(String word) {
        Trie node = find(word);
        return node != null && node.isEnd;
    }

    public boolean startsWith(String prefix) {
        return find(prefix) != null;
    }

    private Trie find(String word) {
        Trie node = this;
        for (char c : word.toCharArray()) {
            int idx = c - 'a';
            if (node.children[idx] == null) return null;
            node = node.children[idx];
        }
        return node;
    }
}
```

**Python**
```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word: str) -> None:
        node = self.root
        for c in word:
            if c not in node.children:
                node.children[c] = TrieNode()
            node = node.children[c]
        node.is_end = True

    def search(self, word: str) -> bool:
        node = self._find(word)
        return node is not None and node.is_end

    def starts_with(self, prefix: str) -> bool:
        return self._find(prefix) is not None

    def _find(self, word: str) -> TrieNode | None:
        node = self.root
        for c in word:
            if c not in node.children:
                return None
            node = node.children[c]
        return node
```

---

## Segment Tree (Range Queries)

### Use Cases
- Range sum / min / max queries with point updates
- O(log n) query and update, O(n) build

**C++**
```cpp
class SegmentTree {
    vector<int> tree;
    int n;
public:
    SegmentTree(vector<int>& nums) : n(nums.size()), tree(2 * nums.size()) {
        for (int i = 0; i < n; i++) tree[n + i] = nums[i];
        for (int i = n - 1; i > 0; i--) tree[i] = tree[2*i] + tree[2*i+1];
    }

    void update(int idx, int val) {
        idx += n;
        tree[idx] = val;
        while (idx > 1) { idx /= 2; tree[idx] = tree[2*idx] + tree[2*idx+1]; }
    }

    int query(int left, int right) { // [left, right)
        int res = 0;
        for (left += n, right += n; left < right; left >>= 1, right >>= 1) {
            if (left & 1) res += tree[left++];
            if (right & 1) res += tree[--right];
        }
        return res;
    }
};
```

**Java**
```java
class SegmentTree {
    int[] tree;
    int n;

    SegmentTree(int[] nums) {
        n = nums.length;
        tree = new int[2 * n];
        for (int i = 0; i < n; i++) tree[n + i] = nums[i];
        for (int i = n - 1; i > 0; i--) tree[i] = tree[2*i] + tree[2*i+1];
    }

    void update(int idx, int val) {
        idx += n;
        tree[idx] = val;
        while (idx > 1) { idx /= 2; tree[idx] = tree[2*idx] + tree[2*idx+1]; }
    }

    int query(int left, int right) { // [left, right)
        int res = 0;
        for (left += n, right += n; left < right; left >>= 1, right >>= 1) {
            if ((left & 1) == 1) res += tree[left++];
            if ((right & 1) == 1) res += tree[--right];
        }
        return res;
    }
}
```

**Python**
```python
class SegmentTree:
    def __init__(self, nums: list[int]):
        self.n = len(nums)
        self.tree = [0] * (2 * self.n)
        # Build
        for i in range(self.n):
            self.tree[self.n + i] = nums[i]
        for i in range(self.n - 1, 0, -1):
            self.tree[i] = self.tree[2 * i] + self.tree[2 * i + 1]

    def update(self, idx: int, val: int):
        idx += self.n
        self.tree[idx] = val
        while idx > 1:
            idx //= 2
            self.tree[idx] = self.tree[2 * idx] + self.tree[2 * idx + 1]

    def query(self, left: int, right: int) -> int:
        """Sum of [left, right)"""
        res = 0
        left += self.n
        right += self.n
        while left < right:
            if left & 1:
                res += self.tree[left]
                left += 1
            if right & 1:
                right -= 1
                res += self.tree[right]
            left >>= 1
            right >>= 1
        return res
```

---

## Practice Problems

| # | Problem | Difficulty | Pattern | Link |
|---|---------|-----------|---------|------|
| 1 | Implement Trie | Medium | Trie | [LeetCode 208](https://leetcode.com/problems/implement-trie-prefix-tree/) |
| 2 | Design Add and Search Words | Medium | Trie + DFS | [LeetCode 211](https://leetcode.com/problems/design-add-and-search-words-data-structure/) |
| 3 | Word Search II | Hard | Trie + Backtracking | [LeetCode 212](https://leetcode.com/problems/word-search-ii/) |
| 4 | Range Sum Query - Mutable | Medium | Segment Tree | [LeetCode 307](https://leetcode.com/problems/range-sum-query-mutable/) |

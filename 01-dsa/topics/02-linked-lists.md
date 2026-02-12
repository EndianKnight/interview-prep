# Linked Lists

A fundamental data structure for interviews — tests your ability to manipulate pointers and handle edge cases.

---

## Core Concepts

### Singly Linked List
- Each node stores a value and a pointer to the next node
- Head pointer tracks the start; last node points to `null`
- No random access — traversal is O(n)

### Doubly Linked List
- Each node has `prev` and `next` pointers
- Enables O(1) deletion when you have a reference to the node
- Used in LRU cache, browser history, undo operations

### Key Properties
| Property | Array | Linked List |
|----------|-------|-------------|
| Access | O(1) | O(n) |
| Insert at head | O(n) | O(1) |
| Insert at tail | O(1)* | O(1) with tail ptr |
| Delete (given node) | O(n) | O(1) |
| Memory | Contiguous | Scattered |
| Cache performance | Excellent | Poor |

---

## Pattern 1: Fast & Slow Pointers (Floyd's Algorithm)

**When to use:** Cycle detection, finding middle node, finding the start of a cycle.

### Technique
Use two pointers: `slow` moves 1 step, `fast` moves 2 steps. If they meet, there's a cycle. Slow will be at the middle when fast reaches the end.

### Example: Detect Cycle

**C++**
```cpp
struct ListNode {
    int val;
    ListNode* next;
    ListNode(int x) : val(x), next(nullptr) {}
};

bool hasCycle(ListNode* head) {
    ListNode* slow = head;
    ListNode* fast = head;

    while (fast && fast->next) {
        slow = slow->next;
        fast = fast->next->next;
        if (slow == fast) return true;
    }
    return false;
}
```

**Java**
```java
public class ListNode {
    int val;
    ListNode next;
    ListNode(int x) { val = x; next = null; }
}

public boolean hasCycle(ListNode head) {
    ListNode slow = head, fast = head;

    while (fast != null && fast.next != null) {
        slow = slow.next;
        fast = fast.next.next;
        if (slow == fast) return true;
    }
    return false;
}
```

**Python**
```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def has_cycle(head: ListNode | None) -> bool:
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            return True
    return False
```

### Example: Find Middle Node

**C++**
```cpp
ListNode* middleNode(ListNode* head) {
    ListNode* slow = head;
    ListNode* fast = head;
    while (fast && fast->next) {
        slow = slow->next;
        fast = fast->next->next;
    }
    return slow;
}
```

**Java**
```java
public ListNode middleNode(ListNode head) {
    ListNode slow = head, fast = head;
    while (fast != null && fast.next != null) {
        slow = slow.next;
        fast = fast.next.next;
    }
    return slow;
}
```

**Python**
```python
def find_middle(head: ListNode | None) -> ListNode | None:
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
    return slow  # for even length, returns second middle
```

---

## Pattern 2: Reversal

**When to use:** Reverse a linked list, reverse a portion, palindrome check, reorder list.

### Technique
Use three pointers: `prev`, `current`, `next`. At each step, reverse the `current.next` pointer.

### Example: Reverse Linked List

**C++**
```cpp
ListNode* reverseList(ListNode* head) {
    ListNode* prev = nullptr;
    ListNode* curr = head;

    while (curr) {
        ListNode* next = curr->next;
        curr->next = prev;
        prev = curr;
        curr = next;
    }
    return prev;
}
```

**Java**
```java
public ListNode reverseList(ListNode head) {
    ListNode prev = null, curr = head;

    while (curr != null) {
        ListNode next = curr.next;
        curr.next = prev;
        prev = curr;
        curr = next;
    }
    return prev;
}
```

**Python**
```python
def reverse_list(head: ListNode | None) -> ListNode | None:
    prev, curr = None, head
    while curr:
        curr.next, prev, curr = prev, curr, curr.next
    return prev
```

### Recursive Reversal

**C++**
```cpp
ListNode* reverseListRecursive(ListNode* head) {
    if (!head || !head->next) return head;
    ListNode* newHead = reverseListRecursive(head->next);
    head->next->next = head;
    head->next = nullptr;
    return newHead;
}
```

**Java**
```java
public ListNode reverseListRecursive(ListNode head) {
    if (head == null || head.next == null) return head;
    ListNode newHead = reverseListRecursive(head.next);
    head.next.next = head;
    head.next = null;
    return newHead;
}
```

**Python**
```python
def reverse_list_recursive(head: ListNode | None) -> ListNode | None:
    if not head or not head.next:
        return head
    new_head = reverse_list_recursive(head.next)
    head.next.next = head
    head.next = None
    return new_head
```

---

## Pattern 3: Dummy Head

**When to use:** When the head of the list might change — merging lists, removing nodes, partitioning.

### Technique
Create a dummy node before the head. Build the result list off the dummy, then return `dummy.next`.

### Example: Merge Two Sorted Lists

**C++**
```cpp
ListNode* mergeTwoLists(ListNode* l1, ListNode* l2) {
    ListNode dummy(0);
    ListNode* tail = &dummy;

    while (l1 && l2) {
        if (l1->val <= l2->val) {
            tail->next = l1;
            l1 = l1->next;
        } else {
            tail->next = l2;
            l2 = l2->next;
        }
        tail = tail->next;
    }
    tail->next = l1 ? l1 : l2;
    return dummy.next;
}
```

**Java**
```java
public ListNode mergeTwoLists(ListNode l1, ListNode l2) {
    ListNode dummy = new ListNode(0);
    ListNode tail = dummy;

    while (l1 != null && l2 != null) {
        if (l1.val <= l2.val) {
            tail.next = l1;
            l1 = l1.next;
        } else {
            tail.next = l2;
            l2 = l2.next;
        }
        tail = tail.next;
    }
    tail.next = (l1 != null) ? l1 : l2;
    return dummy.next;
}
```

**Python**
```python
def merge_two_lists(l1: ListNode | None, l2: ListNode | None) -> ListNode | None:
    dummy = tail = ListNode(0)

    while l1 and l2:
        if l1.val <= l2.val:
            tail.next = l1
            l1 = l1.next
        else:
            tail.next = l2
            l2 = l2.next
        tail = tail.next

    tail.next = l1 or l2
    return dummy.next
```

---

## Pattern 4: Two-Pointer Gap

**When to use:** Remove Nth node from end, finding kth element from end.

### Technique
Advance the first pointer by N steps, then move both pointers until the first reaches the end. The second pointer will be at the target position.

### Example: Remove Nth Node From End

**C++**
```cpp
ListNode* removeNthFromEnd(ListNode* head, int n) {
    ListNode dummy(0, head);
    ListNode* first = &dummy;
    ListNode* second = &dummy;

    for (int i = 0; i <= n; i++) first = first->next;
    while (first) {
        first = first->next;
        second = second->next;
    }
    ListNode* toDelete = second->next;
    second->next = second->next->next;
    delete toDelete;
    return dummy.next;
}
```

**Java**
```java
public ListNode removeNthFromEnd(ListNode head, int n) {
    ListNode dummy = new ListNode(0, head);
    ListNode first = dummy, second = dummy;

    for (int i = 0; i <= n; i++) first = first.next;
    while (first != null) {
        first = first.next;
        second = second.next;
    }
    second.next = second.next.next;
    return dummy.next;
}
```

**Python**
```python
def remove_nth_from_end(head: ListNode | None, n: int) -> ListNode | None:
    dummy = ListNode(0, head)
    first = second = dummy

    # Advance first by n+1 steps
    for _ in range(n + 1):
        first = first.next

    # Move both until first reaches end
    while first:
        first = first.next
        second = second.next

    second.next = second.next.next
    return dummy.next
```

---

## Pitfalls & Edge Cases

| Pitfall | How to Handle |
|---------|--------------|
| Empty list (`head = null`) | Always check before dereferencing |
| Single node | Test reversal, deletion, cycle detection |
| Modifying head | Use dummy node pattern |
| Memory leaks (C++) | `delete` removed nodes; consider smart pointers |
| Losing reference | Save `next` pointer before modifying `curr.next` |
| Even vs odd length | Fast/slow pointer — check `fast.next` too |

---

## Practice Problems

| # | Problem | Difficulty | Pattern | Link |
|---|---------|-----------|---------|------|
| 1 | Reverse Linked List | Easy | Reversal | [LeetCode 206](https://leetcode.com/problems/reverse-linked-list/) |
| 2 | Merge Two Sorted Lists | Easy | Dummy Head | [LeetCode 21](https://leetcode.com/problems/merge-two-sorted-lists/) |
| 3 | Linked List Cycle | Easy | Fast/Slow | [LeetCode 141](https://leetcode.com/problems/linked-list-cycle/) |
| 4 | Middle of Linked List | Easy | Fast/Slow | [LeetCode 876](https://leetcode.com/problems/middle-of-the-linked-list/) |
| 5 | Remove Nth from End | Medium | Two-Pointer Gap | [LeetCode 19](https://leetcode.com/problems/remove-nth-node-from-end-of-list/) |
| 6 | Reorder List | Medium | Split + Reverse + Merge | [LeetCode 143](https://leetcode.com/problems/reorder-list/) |
| 7 | Add Two Numbers | Medium | Dummy Head + Carry | [LeetCode 2](https://leetcode.com/problems/add-two-numbers/) |
| 8 | LRU Cache | Medium | Doubly LL + Hash Map | [LeetCode 146](https://leetcode.com/problems/lru-cache/) |
| 9 | Reverse Nodes in k-Group | Hard | Reversal | [LeetCode 25](https://leetcode.com/problems/reverse-nodes-in-k-group/) |
| 10 | Merge k Sorted Lists | Hard | Dummy + Heap | [LeetCode 23](https://leetcode.com/problems/merge-k-sorted-lists/) |

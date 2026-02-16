# Generics & Type Safety

Generics enable writing reusable, type-safe code that catches errors at compile time instead of blowing up at runtime.

---

## Why Generics

Without generics, container code relies on the top-level base type (`Object` in Java, `void*` in C) and requires manual casting, which is error-prone and defers type errors to runtime.

### The Problem: Runtime ClassCastException

**Java (without generics)**
```java
List names = new ArrayList();       // raw type — holds Object
names.add("Alice");
names.add(42);                      // compiles fine — no type check
String s = (String) names.get(1);   // ClassCastException at runtime!
```

### The Solution: Compile-Time Safety

**Java (with generics)**
```java
List<String> names = new ArrayList<>();
names.add("Alice");
names.add(42);                      // COMPILE ERROR — int is not String
String s = names.get(0);            // no cast needed
```

### Benefits Summary

| Benefit | Without Generics | With Generics |
|---------|-----------------|---------------|
| Type checking | Runtime | Compile time |
| Casting | Manual, everywhere | Automatic |
| Code reuse | Copy-paste per type | Single parameterized definition |
| Readability | `(String) list.get(i)` | `list.get(i)` |
| Documentation | Comment says "list of strings" | Signature says `List<String>` |

---

## Java Generics

### Generic Classes

```java
public class Box<T> {
    private T value;

    public Box(T value) { this.value = value; }
    public T getValue() { return value; }
    public void setValue(T value) { this.value = value; }
}

// Usage
Box<String> stringBox = new Box<>("hello");
Box<Integer> intBox = new Box<>(42);
String s = stringBox.getValue();  // no cast
```

Multiple type parameters:

```java
public class Pair<K, V> {
    private final K key;
    private final V value;

    public Pair(K key, V value) {
        this.key = key;
        this.value = value;
    }

    public K getKey() { return key; }
    public V getValue() { return value; }
}

Pair<String, Integer> entry = new Pair<>("age", 30);
```

### Generic Methods

```java
public class Utils {
    // The <T> before the return type declares the type parameter
    public static <T> T firstNonNull(T first, T second) {
        return first != null ? first : second;
    }

    public static <T extends Comparable<T>> T max(T a, T b) {
        return a.compareTo(b) >= 0 ? a : b;
    }
}

// Type is inferred from arguments
String name = Utils.firstNonNull(null, "default");
int bigger = Utils.max(3, 7);  // infers Integer
```

### Bounded Types

Upper bounds constrain the type parameter to a supertype or its subtypes:

```java
// T must implement Comparable
public static <T extends Comparable<T>> void sort(List<T> list) {
    Collections.sort(list);
}

// Multiple bounds — T must extend Number AND implement Comparable
public static <T extends Number & Comparable<T>> T clamp(T val, T min, T max) {
    if (val.compareTo(min) < 0) return min;
    if (val.compareTo(max) > 0) return max;
    return val;
}
```

### Wildcards

| Wildcard | Meaning | Use When |
|----------|---------|----------|
| `?` | Unknown type | You don't care about the type at all |
| `? extends T` | Upper bound — T or any subtype | Reading (producing) values |
| `? super T` | Lower bound — T or any supertype | Writing (consuming) values |

```java
// Unbounded — can only read as Object
public static void printAll(List<?> list) {
    for (Object item : list) {
        System.out.println(item);
    }
}

// Upper bound — read items as Number
public static double sum(List<? extends Number> list) {
    double total = 0;
    for (Number n : list) total += n.doubleValue();
    return total;
}
// Works with List<Integer>, List<Double>, etc.

// Lower bound — write Integer into the list
public static void addNumbers(List<? super Integer> list) {
    list.add(1);
    list.add(2);
}
// Works with List<Integer>, List<Number>, List<Object>
```

### PECS — Producer Extends, Consumer Super

The golden rule for deciding which wildcard to use:

- If the structure **produces** values you read from it, use `? extends T`.
- If the structure **consumes** values you write into it, use `? super T`.
- If you both read and write, don't use wildcards (use an exact type).

```java
// Classic example: copying from source (producer) to dest (consumer)
public static <T> void copy(List<? extends T> source, List<? super T> dest) {
    for (T item : source) {
        dest.add(item);
    }
}

List<Integer> ints = List.of(1, 2, 3);
List<Number> nums = new ArrayList<>();
copy(ints, nums);  // Integer extends Number, Number super Integer — works
```

### Type Erasure

Java generics are a **compile-time** mechanism. The JVM has no knowledge of type parameters at runtime.

```java
// What you write:
List<String> names = new ArrayList<>();
names.add("Alice");
String s = names.get(0);

// What the compiler generates (after erasure):
List names = new ArrayList();
names.add("Alice");
String s = (String) names.get(0);  // compiler inserts cast
```

**Consequences of type erasure:**

| Limitation | Example | Why |
|-----------|---------|-----|
| Cannot use `instanceof` with generics | `obj instanceof List<String>` — compile error | Type is erased |
| Cannot create generic arrays | `new T[10]` — compile error | Array type must be reifiable |
| Cannot call `new T()` | No way to instantiate type param | Constructor info erased |
| Overloading on type params fails | `void f(List<String>)` vs `void f(List<Integer>)` | Same erasure: `f(List)` |

### Diamond Operator and Raw Types

```java
// Diamond operator — compiler infers type from left side (Java 7+)
Map<String, List<Integer>> map = new HashMap<>();  // instead of new HashMap<String, List<Integer>>()

// Raw types — AVOID. Disables all generic type checking.
List rawList = new ArrayList();  // warning: raw type
rawList.add(42);
rawList.add("oops");             // no compile error — unsafe
```

---

## C++ Templates

C++ templates generate actual specialized code at compile time for each type used — no erasure, no boxing.

### Function Templates

```cpp
template <typename T>
T maxOf(T a, T b) {
    return (a > b) ? a : b;
}

// Usage — type deduced from arguments
int x = maxOf(3, 7);            // instantiates maxOf<int>
double y = maxOf(3.14, 2.72);   // instantiates maxOf<double>
std::string z = maxOf(std::string("abc"), std::string("xyz"));
```

### Class Templates

```cpp
template <typename T>
class Box {
    T value;
public:
    explicit Box(T v) : value(std::move(v)) {}
    T getValue() const { return value; }
    void setValue(T v) { value = std::move(v); }
};

// Usage
Box<int> intBox(42);
Box<std::string> strBox("hello");
```

Multiple type parameters with defaults:

```cpp
template <typename K, typename V, typename Comparator = std::less<K>>
class SortedMap {
    // ...
};

SortedMap<std::string, int> m;  // uses default std::less<std::string>
```

### Template Specialization

Provide a custom implementation for a specific type:

```cpp
// Primary template
template <typename T>
class Serializer {
public:
    static std::string serialize(const T& obj) {
        return std::to_string(obj);
    }
};

// Full specialization for std::string
template <>
class Serializer<std::string> {
public:
    static std::string serialize(const std::string& obj) {
        return "\"" + obj + "\"";  // wrap in quotes
    }
};

// Partial specialization for pointers
template <typename T>
class Serializer<T*> {
public:
    static std::string serialize(const T* obj) {
        return obj ? Serializer<T>::serialize(*obj) : "null";
    }
};
```

### SFINAE (Substitution Failure Is Not An Error)

When a template substitution fails, the compiler silently removes that overload rather than emitting an error:

```cpp
#include <type_traits>

// Only enabled for arithmetic types
template <typename T>
typename std::enable_if<std::is_arithmetic<T>::value, T>::type
add(T a, T b) {
    return a + b;
}

// Only enabled for string-like types
template <typename T>
typename std::enable_if<std::is_same<T, std::string>::value, T>::type
add(T a, T b) {
    return a + b;  // string concatenation
}

add(3, 4);                                         // calls arithmetic version
add(std::string("hi"), std::string(" there"));     // calls string version
```

### Concepts (C++20)

Concepts replace SFINAE with clean, readable constraints:

```cpp
#include <concepts>

// Define a concept
template <typename T>
concept Addable = requires(T a, T b) {
    { a + b } -> std::convertible_to<T>;
};

template <typename T>
concept Printable = requires(T a, std::ostream& os) {
    { os << a } -> std::same_as<std::ostream&>;
};

// Use as constraint
template <Addable T>
T add(T a, T b) {
    return a + b;
}

// Combine concepts
template <typename T>
    requires Addable<T> && Printable<T>
void addAndPrint(T a, T b) {
    std::cout << add(a, b) << std::endl;
}
```

### Templates vs Java Generics

| Aspect | C++ Templates | Java Generics |
|--------|--------------|---------------|
| Mechanism | Code generation (monomorphization) | Type erasure |
| Runtime type info | Full type info preserved | Erased to `Object` |
| Primitive types | `vector<int>` works directly | `List<int>` illegal — must use `List<Integer>` |
| Compilation | Slower (generates code per type) | Faster (single erased version) |
| Binary size | Larger (code duplication) | Smaller |
| Non-type params | `template <int N>` supported | Not supported |
| Error messages | Historically terrible, better with concepts | Clearer |
| Turing complete | Yes (template metaprogramming) | No |

---

## Python Typing

Python is dynamically typed, so generics are primarily for **static analysis** (mypy, pyright) and documentation. They have no effect at runtime by default.

### TypeVar and Generic

```python
from typing import TypeVar, Generic, Optional

T = TypeVar('T')

class Box(Generic[T]):
    def __init__(self, value: T) -> None:
        self.value = value

    def get(self) -> T:
        return self.value

    def set(self, value: T) -> None:
        self.value = value

# Usage — type checker understands these
str_box: Box[str] = Box("hello")
int_box: Box[int] = Box(42)
reveal_type(str_box.get())  # mypy says: str
```

### Bounded TypeVar

```python
from typing import TypeVar

# T must be int or float
Numeric = TypeVar('Numeric', int, float)

def add(a: Numeric, b: Numeric) -> Numeric:
    return a + b

# T must be a subtype of Comparable
from typing import Protocol

class Comparable(Protocol):
    def __lt__(self, other: 'Comparable') -> bool: ...

CT = TypeVar('CT', bound=Comparable)

def max_of(a: CT, b: CT) -> CT:
    return a if not a < b else b
```

### Generic Functions

```python
from typing import TypeVar, Sequence, Optional

T = TypeVar('T')

def first(items: Sequence[T]) -> Optional[T]:
    return items[0] if items else None

def first_non_none(*args: Optional[T]) -> Optional[T]:
    for arg in args:
        if arg is not None:
            return arg
    return None

# Type checker infers T from usage
name: Optional[str] = first(["Alice", "Bob"])   # T = str
num: Optional[int] = first([1, 2, 3])           # T = int
```

### Protocol — Structural Subtyping

Protocols allow type checking based on what an object **can do** (duck typing) rather than what it **inherits from**:

```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class Drawable(Protocol):
    def draw(self) -> None: ...

class Circle:
    def draw(self) -> None:
        print("Drawing circle")

class Square:
    def draw(self) -> None:
        print("Drawing square")

# Circle and Square satisfy Drawable without inheriting from it
def render(shape: Drawable) -> None:
    shape.draw()

render(Circle())  # OK — Circle has draw()
render(Square())  # OK — Square has draw()
render("hello")   # mypy error — str has no draw()

# With @runtime_checkable, isinstance works too
assert isinstance(Circle(), Drawable)  # True at runtime
```

### Type Hints for Containers (Python 3.9+)

```python
# Python 3.9+ — use built-in types directly
names: list[str] = ["Alice", "Bob"]
scores: dict[str, int] = {"Alice": 95}
point: tuple[float, float] = (1.0, 2.0)
maybe: int | None = None                  # Python 3.10+ union syntax

# Before 3.9 — import from typing
from typing import List, Dict, Tuple, Optional
names: List[str] = ["Alice", "Bob"]
maybe: Optional[int] = None               # equivalent to Union[int, None]
```

### Runtime vs Static Checking

| Aspect | Runtime (Python default) | Static (mypy / pyright) |
|--------|------------------------|------------------------|
| When errors caught | Execution time | Before running |
| Performance impact | None (hints are ignored) | Separate analysis step |
| Enforcement | No enforcement at runtime | Editor + CI enforcement |
| Generic types | Erased — `Box[str]` is just `Box` at runtime | Fully checked |

```bash
# Run static type checking
mypy my_module.py --strict
pyright my_module.py
```

---

## Variance

Variance determines how subtyping between container types relates to subtyping between their element types.

Assume: `Dog extends Animal`.

| Variance | Meaning | Container Relationship | Example |
|----------|---------|----------------------|---------|
| **Covariant** | Preserves subtype direction | `Container<Dog>` is subtype of `Container<Animal>` | `? extends T`, `out T` |
| **Contravariant** | Reverses subtype direction | `Container<Animal>` is subtype of `Container<Dog>` | `? super T`, `in T` |
| **Invariant** | No relationship | `Container<Dog>` has no relation to `Container<Animal>` | Java `List<T>` |

### Why Invariance Is the Safe Default

If `List<Dog>` were a subtype of `List<Animal>`, you could do this:

```java
List<Dog> dogs = new ArrayList<>();
dogs.add(new Dog());

List<Animal> animals = dogs;   // hypothetically allowed
animals.add(new Cat());        // Cat is an Animal — compiles

Dog d = dogs.get(1);           // BOOM — it's actually a Cat!
```

This is why Java generics are **invariant** by default.

### Java Arrays Are Covariant (Broken!)

Java arrays **are** covariant, which was a design mistake:

```java
Dog[] dogs = new Dog[3];
Animal[] animals = dogs;       // compiles — arrays are covariant
animals[0] = new Cat();        // compiles — but throws ArrayStoreException at runtime!
```

Generics fixed this by making `List<Dog>` **not** assignable to `List<Animal>`.

### Use-Site Variance with Wildcards (Java)

Java achieves variance through wildcards at the **use site**:

```java
// Covariant — safe to read, cannot write
List<? extends Animal> animals = new ArrayList<Dog>();  // OK
Animal a = animals.get(0);    // OK — read as Animal
animals.add(new Dog());       // COMPILE ERROR — can't write

// Contravariant — safe to write, reads return Object
List<? super Dog> dogs = new ArrayList<Animal>();  // OK
dogs.add(new Dog());          // OK — can write Dog
Object o = dogs.get(0);       // reads only as Object
```

### Variance in C++

C++ templates are **invariant** — `vector<Dog>` and `vector<Animal>` are completely unrelated types. Variance is handled through template parameters and concepts:

```cpp
// Simulate covariance via templated function
template <typename T>
    requires std::derived_from<T, Animal>
void processAnimals(const std::vector<T>& animals) {
    for (const auto& a : animals) {
        a.speak();  // OK — T is an Animal
    }
}

std::vector<Dog> dogs;
processAnimals(dogs);  // works — Dog derives from Animal
```

### Variance in Python

Python's `typing` module supports explicit variance declarations:

```python
from typing import TypeVar, Generic

# Covariant — only used in output (return) positions
T_co = TypeVar('T_co', covariant=True)

class ImmutableBox(Generic[T_co]):
    def __init__(self, value: T_co) -> None:
        self._value = value

    def get(self) -> T_co:
        return self._value
    # No setter — covariant types cannot appear in input positions

# Contravariant — only used in input (parameter) positions
T_contra = TypeVar('T_contra', contravariant=True)

class Processor(Generic[T_contra]):
    def process(self, item: T_contra) -> None:
        print(item)
```

### Variance Summary by Language

| Language | Default Variance | Covariance | Contravariance |
|----------|-----------------|------------|----------------|
| Java (generics) | Invariant | `? extends T` | `? super T` |
| Java (arrays) | Covariant (!) | Built-in (unsafe) | N/A |
| C++ | Invariant | Templates + concepts | Templates + concepts |
| Python | Invariant | `TypeVar(covariant=True)` | `TypeVar(contravariant=True)` |
| Kotlin | Invariant | `out T` (declaration-site) | `in T` (declaration-site) |
| C# | Invariant | `out T` | `in T` |

---

## Common Interview Questions

**1. What is type erasure in Java, and what are its limitations?**

Java generics are implemented via type erasure: the compiler removes all type parameter information and inserts casts. At runtime, `List<String>` and `List<Integer>` are both just `List`. This means you cannot use `instanceof` with generic types, cannot create generic arrays (`new T[]`), cannot instantiate type parameters (`new T()`), and cannot overload methods that differ only in their generic type arguments.

**2. Explain PECS. When do you use `extends` vs `super`?**

PECS stands for Producer Extends, Consumer Super. If a parameterized type **produces** values you read from it, use `? extends T` (covariant, safe to read). If it **consumes** values you write into it, use `? super T` (contravariant, safe to write). For example, `Collections.copy(List<? super T> dest, List<? extends T> src)` reads from the source (producer, extends) and writes to the destination (consumer, super).

**3. How do C++ templates differ from Java generics?**

C++ templates use monomorphization: the compiler generates separate, specialized code for each type used (e.g., `vector<int>` and `vector<double>` are distinct classes). Java generics use type erasure: a single class is compiled and type parameters are erased to `Object`. Consequences: C++ templates work with primitives, preserve runtime type info, support non-type template parameters, and enable compile-time metaprogramming, but produce larger binaries and slower compilation. Java generics require boxing for primitives and lose type info at runtime.

**4. Why are Java arrays covariant but generics invariant?**

Arrays were designed before generics existed (Java 1.0) and were made covariant so that methods like `Arrays.sort(Object[])` could accept any array type. This is unsafe: storing a `Cat` in a `Dog[]` (viewed as `Animal[]`) compiles but throws `ArrayStoreException` at runtime. Generics (Java 5) learned from this mistake and are invariant by default, catching such errors at compile time. Wildcards (`? extends`, `? super`) provide safe, controlled variance.

**5. What is a Python Protocol, and how does it differ from an abstract base class?**

A Protocol defines a structural type (duck typing with types): any class with matching methods satisfies the protocol without explicitly inheriting from it. An ABC requires explicit inheritance (`class Foo(ABC)`). Protocols enable retroactive conformance — you can declare that third-party classes satisfy your interface without modifying them. With `@runtime_checkable`, you can even use `isinstance()` checks against Protocols.

**6. What are C++20 Concepts, and what problem do they solve?**

Concepts are named Boolean predicates on template parameters that constrain which types can be used. They replace SFINAE (`enable_if`) with readable, composable constraints. Before concepts, invalid template instantiations produced notoriously unreadable error messages because the compiler would try substitution, fail deep in implementation code, and dump pages of template backtrace. Concepts catch constraint violations early and produce clear error messages like "type T does not satisfy Addable."

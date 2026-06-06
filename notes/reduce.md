# barrett_reduce()
Reduce a number back into the valid ML-KEM coefficient range around q = 3329.

---
# montgomery_reduce()
Efficiently performs modular reduction after multiplication

---

# Example

Suppose:
```
a = 5000
q = 3329
```
We want:
```
5000 mod 3329 = 1671
```
So:
```
barrett_reduce(5000) # Output: 1671
```

# Why?

During polynomial arithmetic, coefficients become larger than q.

Example:

2000 + 2500 = 4500

But ML-KEM requires coefficients to stay modulo: q = 3329

so we have to reduce them.

---

# Why Two Reduction Functions?
| Function              | Used After             |
| --------------------- | ---------------------- |
| `barrett_reduce()`    | Addition / subtraction |
| `montgomery_reduce()` | Multiplication         |


For **Addition/Subtraction** - we use `barrett_reduce()`
For **Multiplication** - we use `montgomery_reduce()`

# Important
[x] In `montgomery_reduce(int32_t a)` - why we use `int32`
Suppose two coefficients are:
```
a = 3000
b = 3000

Multiplication: 
3000 × 3000 = 9,000,000
```

Now let's compare with integer limits:
| Type      | Range                              |
| --------- | ---------------------------------- |
| `int16_t` | -32,768 to 32,767                  |
| `int32_t` | about -2.1 billion to +2.1 billion |

Clearly: `9,000,000` cannot fit inside `int16_t`. It would overflow.


Therefore `int32_t a` is used to safely hold multiplication results before reduction and after reduction, the value becomes small again and can fit into `int16_t` which is why the function returns `int16_t`



> Most ML-KEM operations work on 16-bit coefficients, but multiplication creates larger intermediate values that must be reduced back into the valid range.

In GPU
`torch.Tensor` function should reduce many coefficients at once, not just one integer unlike in C which reduce one coefficient at a time.

C int16_t a        → one coefficient
PyTorch Tensor a   → many coefficients

But internally we still treat the values like integers and return:
```
.to(torch.int16)
```

to match the C output type.
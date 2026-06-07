For PyTorch
| PQClean C                | Our PyTorch version             |
| ------------------------ | ------------------------------- |
| `int16_t r[256]`         | tensor shape `(256,)`           |
| `zetas[128]`             | tensor shape `(128,)`           |
| `ntt(r)`                 | transforms one polynomial       |
| `invntt(r)`              | transforms back                 |
| `basemul(r, a, b, zeta)` | multiplies 2-coefficient blocks |



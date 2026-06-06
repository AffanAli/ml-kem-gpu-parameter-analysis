# ML-KEM-512 / Kyber512 Parameters Explained (`params.md`)

## Table of Contents

1. [Introduction](#introduction)
2. [Main Parameter Table](#main-parameter-table)
3. [Polynomial vs Polynomial Vector](#polynomial-vs-polynomial-vector)
4. [Quick Size Relationships](#quick-size-relationships)
5. [Why Does One Polynomial Occupy 384 Bytes?](#why-does-one-polynomial-occupy-384-bytes)
6. [Where Does `KYBER_POLYVECCOMPRESSEDBYTES = 2 × 320` Come From?](#where-does-kyber_polyveccompressedbytes--2--320-come-from)
7. [Why is `KYBER_POLYCOMPRESSEDBYTES = 128`?](#why-is-kyber_polycompressedbytes--128)
8. [Understanding Compression in ML-KEM](#understanding-compression-in-ml-kem)

---

# Introduction

The file `params.h` contains the core configuration constants used by ML-KEM (formerly CRYSTALS-Kyber). These constants define:

- Polynomial dimensions
- Security level
- Key sizes
- Ciphertext sizes
- Compression parameters
- Noise distribution parameters

Most of the sizes used throughout the implementation can be derived from a few fundamental parameters:

```c
KYBER_N = 256
KYBER_Q = 3329
KYBER_K = 2
```

Understanding these parameters makes it much easier to understand key generation, encapsulation, decapsulation, serialization, and compression.

---

## 1. Main Parameter Table

| Constant | Value | Unit | One-line Explanation | Simple Example |
|---|---:|---|---|---|
| `KYBER_N` | 256 | coefficients | Number of coefficients in one polynomial. | One polynomial has 256 values: `[a0, a1, ..., a255]`. |
| `KYBER_Q` | 3329 | modulus value | Prime modulus used for modular arithmetic. | If a coefficient becomes `3500`, it is reduced as `3500 mod 3329 = 171`. |
| `KYBER_SYMBYTES` | 32 | bytes | Length of seeds and hash outputs. | A random seed is 32 bytes. |
| `KYBER_SSBYTES` | 32 | bytes | Length of the final shared secret. | Encapsulation produces a 32-byte shared secret. |
| `KYBER_K` | 2 | polynomials | Security/module dimension. ML-KEM-512 uses vectors of 2 polynomials. | Secret vector is `s = (s0, s1)`. |
| `KYBER_POLYBYTES` | 384 | bytes | Storage size of one serialized polynomial. | One polynomial with 256 coefficients is packed into 384 bytes. |
| `KYBER_POLYVECBYTES` | 768 | bytes | Storage size of one serialized polynomial vector. | `2 × 384 = 768` bytes. |
| `KYBER_ETA1` | 3 | distribution parameter | Noise distribution parameter for secret key generation. | Secret coefficients are sampled from small values using `eta = 3`. |
| `KYBER_ETA2` | 2 | distribution parameter | Noise distribution parameter for encryption noise. | Encryption noise uses `eta = 2`. |
| `KYBER_POLYCOMPRESSEDBYTES` | 128 | bytes | Size of one compressed polynomial in the ciphertext. | The `v` part of ciphertext is compressed to 128 bytes. |
| `KYBER_POLYVECCOMPRESSEDBYTES` | 640 | bytes | Size of one compressed polynomial vector in the ciphertext. | `2 × 320 = 640` bytes. |
| `KYBER_INDCPA_MSGBYTES` | 32 | bytes | Message size accepted by IND-CPA encryption. | A 32-byte message is encrypted internally. |
| `KYBER_INDCPA_PUBLICKEYBYTES` | 800 | bytes | Public key size for IND-CPA encryption. | Public key = 768-byte polynomial vector + 32-byte seed. |
| `KYBER_INDCPA_SECRETKEYBYTES` | 768 | bytes | Secret key size for IND-CPA encryption. | Secret key contains one polynomial vector: `2 × 384 = 768` bytes. |
| `KYBER_INDCPA_BYTES` | 768 | bytes | Ciphertext size for IND-CPA encryption. | Ciphertext = compressed polyvec + compressed polynomial = `640 + 128 = 768` bytes. |
| `KYBER_PUBLICKEYBYTES` | 800 | bytes | Final ML-KEM public key size. | The public key sent to another party is 800 bytes. |
| `KYBER_SECRETKEYBYTES` | 1632 | bytes | Final ML-KEM secret key size. | Secret key stores private values plus public key and hashes. |
| `KYBER_CIPHERTEXTBYTES` | 768 | bytes | Final ML-KEM ciphertext size. | Alice sends a 768-byte ciphertext to Bob during encapsulation. |

---

## Polynomial vs Polynomial Vector

### What is a Polynomial?

A polynomial is a mathematical expression such as:

```text
a(x) = a0 + a1x + a2x² + ... + a255x²⁵⁵
```

In ML-KEM:

```text
KYBER_N = 256
```

So one polynomial has exactly 256 coefficients.

In code, we can think of it like this:

```python
poly = [a0, a1, a2, ..., a255]
```

So a polynomial is like one list of 256 numbers.

Example with a very small toy polynomial:

```text
p(x) = 3 + 5x + 2x²
```

This can be represented as:

```python
p = [3, 5, 2]
```

Real ML-KEM uses 256 coefficients, not 3.

---

### What is a Polynomial Vector?

A polynomial vector is a vector whose elements are polynomials.

For ML-KEM-512:

```text
KYBER_K = 2
```

So a polynomial vector contains 2 polynomials.

Example:

```text
s = (s0, s1)
```

This means:

```text
s0 = one polynomial with 256 coefficients
s1 = one polynomial with 256 coefficients
```

A simple toy example:

```text
s0(x) = 1 - x + 2x³
s1(x) = x + x² - 2x³
```

In code:

```python
s = [
    [256 coefficients for s0],
    [256 coefficients for s1]
]
```

So for ML-KEM-512, the shape is:

```text
s.shape = (2, 256)
```

This is not an `(x, y)` coordinate. It is a full vector of polynomials.

---

# Quick Size Relationships

For ML-KEM-512:

```text
KYBER_N = 256
KYBER_Q = 3329
KYBER_K = 2
KYBER_SYMBYTES = 32
```

### Polynomial Size

```text
KYBER_POLYBYTES = 384 bytes
```

Reason:

```text
256 coefficients × 12 bits per coefficient = 3072 bits
3072 bits / 8 = 384 bytes
```

### Polynomial Vector Size

```text
KYBER_POLYVECBYTES = KYBER_K × KYBER_POLYBYTES
                    = 2 × 384
                    = 768 bytes
```

### IND-CPA Public Key Size

```text
KYBER_INDCPA_PUBLICKEYBYTES = KYBER_POLYVECBYTES + KYBER_SYMBYTES
                            = 768 + 32
                            = 800 bytes
```

The public key contains:

```text
1 polynomial vector t = 768 bytes
1 seed rho          = 32 bytes
Total               = 800 bytes
```

### IND-CPA Secret Key Size

```text
KYBER_INDCPA_SECRETKEYBYTES = KYBER_POLYVECBYTES
                            = 768 bytes
```

The IND-CPA secret key is essentially the secret polynomial vector `s`.

### Ciphertext Size

```text
KYBER_CIPHERTEXTBYTES = KYBER_POLYVECCOMPRESSEDBYTES + KYBER_POLYCOMPRESSEDBYTES
                      = 640 + 128
                      = 768 bytes
```

The ciphertext contains two parts:

```text
u = compressed polynomial vector = 640 bytes
v = compressed polynomial        = 128 bytes
Total                            = 768 bytes
```

### Final ML-KEM Secret Key Size

```text
KYBER_SECRETKEYBYTES = KYBER_INDCPA_SECRETKEYBYTES
                     + KYBER_INDCPA_PUBLICKEYBYTES
                     + 2 × KYBER_SYMBYTES

                     = 768 + 800 + 64
                     = 1632 bytes
```

The final secret key includes more than just the IND-CPA secret key. It also stores the public key and hash-related values used for secure decapsulation.

---

## Why Does One Polynomial Occupy 384 Bytes?

The value is not random. It comes from the packing format.

### Step 1: Number of coefficients

ML-KEM uses:

```text
KYBER_N = 256
```

So each polynomial has 256 coefficients.

### Step 2: Coefficient range

All coefficients are modulo:

```text
KYBER_Q = 3329
```

So each coefficient is in the range:

```text
0 to 3328
```

### Step 3: Bits needed for one coefficient

To store values up to 3328, we need 12 bits.

log<sub>2</sub>(3329) ≈ 11.7

So we need: 12 bits / per coefficient.


Therefore:

```text
one coefficient needs 12 bits
```

### Step 4: Total bits for one polynomial

```text
256 coefficients × 12 bits = 3072 bits
```

### Step 5: Convert bits to bytes

```text
3072 bits / 8 = 384 bytes
```

Therefore:

```text
KYBER_POLYBYTES = 384
```

### Why not 512 bytes?

If we stored every coefficient as a normal 16-bit integer:

```text
256 × 16 bits = 4096 bits
4096 / 8 = 512 bytes
```

But Kyber packs coefficients more efficiently using 12 bits per coefficient:

```text
256 × 12 bits = 3072 bits
3072 / 8 = 384 bytes
```

So `384 bytes` is the packed/serialized size, not the normal in-memory `int16` size.

---

# Where Does `KYBER_POLYVECCOMPRESSEDBYTES = 2 × 320` Come From?

For ML-KEM-512:

```text
KYBER_K = 2
```

So a polynomial vector contains 2 polynomials.

That explains the `2`.

Now we need to explain where `320` comes from.

### Step 1: One polynomial has 256 coefficients

```text
KYBER_N = 256
```

### Step 2: In compressed polynomial vector form, each coefficient is stored using 10 bits

For the `u` part of the ciphertext in ML-KEM-512, each coefficient is compressed to 10 bits.

So:

```text
1 compressed coefficient = 10 bits
```

### Step 3: Size of one compressed polynomial

```text
256 coefficients × 10 bits = 2560 bits
```

Convert to bytes:

```text
2560 bits / 8 = 320 bytes
```

Therefore:

```text
one compressed polynomial inside the polynomial vector = 320 bytes
```

### Step 4: Size of compressed polynomial vector

Since ML-KEM-512 has 2 polynomials in a vector:

```text
KYBER_POLYVECCOMPRESSEDBYTES = 2 × 320
                              = 640 bytes
```

Therefore:

```text
KYBER_POLYVECCOMPRESSEDBYTES = 640
```

---

# Why is `KYBER_POLYCOMPRESSEDBYTES = 128`?

This is for one compressed polynomial, usually the `v` part of the ciphertext.

Here, each coefficient is compressed to 4 bits.

```text
256 coefficients × 4 bits = 1024 bits
1024 bits / 8 = 128 bytes
```

So:

```text
KYBER_POLYCOMPRESSEDBYTES = 128
```

Important difference:

```text
compressed polynomial vector u: 10 bits per coefficient → 320 bytes per polynomial
compressed polynomial v:       4 bits per coefficient  → 128 bytes per polynomial
```

That is why the two compressed sizes are different.

---


# Understanding Compression in ML-KEM

One common question is:

> If a coefficient requires 12 bits, how can ML-KEM compress it to 10 bits or even 4 bits?

The answer is:

> Compression is intentionally lossy.

ML-KEM does not preserve the exact coefficient value. Instead, it stores an approximation.

---

## Exact Representation

A coefficient can be:

```text
0 ... 3328
```

Therefore:

```text
12 bits
```

are required for exact storage.

Example:

```text
Coefficient = 3000
```

Stored exactly using 12 bits.

---

## 10-bit Compression
10 bits can represent:

```text
0 ... 1023
```

Explanation

<img width="716" height="481" alt="Screenshot 2026-06-06 at 8 39 33 PM" src="https://github.com/user-attachments/assets/33c28b0a-0ca7-4370-ba3d-22464b7235d7" />


---

## 4-bit Compression
4 bits can represent:

```text
0 ... 15
```

Explanation

<img width="716" height="469" alt="Screenshot 2026-06-06 at 8 40 03 PM" src="https://github.com/user-attachments/assets/a5c884b7-47e6-40fe-8064-2dc85d9571da" />


---

## Why Does This Work?

ML-KEM is designed to tolerate small inaccuracies.

The underlying Module-LWE system already contains noise, and decryption only needs to determine the correct message region rather than recover every coefficient exactly.

This allows ML-KEM to dramatically reduce ciphertext size while maintaining correctness and security.

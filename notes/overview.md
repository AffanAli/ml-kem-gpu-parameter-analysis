# Understanding ML-KEM (Beginner Friendly)

## Why Do We Need ML-KEM?

Suppose Alice wants to send a secure message to Bob over the Internet.

```text
Alice ───── Internet ───── Bob
                ▲
                │
             Attacker
```

If Alice sends:

```text
Hello Bob
```

everyone can read it.

Therefore Alice and Bob need a **shared secret key** that only they know.

For example:

```text
Shared Secret = 32 random bytes
```

Example:

```text
7A 2F B1 8C ...
```

Once both parties have the same shared secret, they can use AES or another symmetric encryption algorithm to securely exchange messages.

---

## Traditional RSA Approach

Traditional public-key cryptography uses RSA.

Bob generates:

```text
Public Key
Private Key
```

Alice encrypts something using Bob's public key.

Bob decrypts it using his private key.

The problem is that a sufficiently powerful quantum computer could break RSA.

This is why post-quantum cryptography was developed.

---

## ML-KEM Approach

ML-KEM does not directly encrypt messages.

Instead it performs:

```text
Key Exchange
```

or more specifically:

```text
Key Encapsulation Mechanism (KEM)
```

Its goal is simple:

```text
Alice and Bob obtain the same 32-byte secret
```

without ever sending that secret across the network.

---

## High-Level Overview

```text
       Bob
        │
        │ Generate Keys
        ▼

 ┌───────────────────┐
 │ Public Key 800 B  │
 │ Secret Key 1632 B │
 └───────────────────┘

        │
        │ send public key
        ▼

       Alice

        │
        │ Encapsulation
        ▼

 ┌───────────────────┐
 │ Ciphertext 768 B  │
 │ Shared Key 32 B   │
 └───────────────────┘

        │
        │ send ciphertext
        ▼

       Bob

        │
        │ Decapsulation
        ▼

 ┌───────────────────┐
 │ Shared Key 32 B   │
 └───────────────────┘
```

At the end:

```text
Alice Shared Key = Bob Shared Key
```


# Understanding the Sizes

## Public Key = 800 Bytes

Think of the public key as:

```text
Bob's public lock
```

Bob publishes it so that anyone can securely establish a shared secret with him.

Calculation:

```text
= KYBER_K (KYBER_POLYBYTES) + KYBER_SSBYTES
= 2 ( 384) + 32
= 800
```

---

## Secret Key = 1632 Bytes

Think of the secret key as:

```text
Bob's private unlock mechanism
```

It never leaves Bob's device.

Calculation:

```text
KYBER_SECRETKEYBYTES = (
    KYBER_INDCPA_SECRETKEYBYTES + KYBER_INDCPA_PUBLICKEYBYTES + 2 * KYBER_SYMBYTES
)
= (2 * 384) + [(2 * 384) + 32] + (2 * 32)
= 768 + 800 + 64 
= 1632
```

---

## Ciphertext = 768 Bytes

This is often misunderstood. Many people assume: Ciphertext is Encrypted message However, in ML-KEM, Ciphertext is **Information required to reconstruct the shared secret**

Think of it as: A mathematical puzzle that only Bob can solve.

Calculation

```
KYBER_CIPHERTEXTBYTES = ( KYBER_POLYVECCOMPRESSEDBYTES + KYBER_POLYCOMPRESSEDBYTES )
 = [ (KYBER_K * 320) + 128 ] 
 = (2 * 320) + 128
 = 768
```

---

## Shared Secret = 32 Bytes

This is the actual goal of ML-KEM.

Example:

```text
2F A1 88 91 ...
```

Length: 32 bytes = 256 bits

Suitable for:

```text
AES-256
ChaCha20
HKDF
TLS
VPNs
```

---

# Why Is Ciphertext Larger Than The Shared Secret?

A common question is:

```text
Shared Secret = 32 bytes
Ciphertext    = 768 bytes

Why send 768 bytes to create only 32 bytes?
```

The answer is that ML-KEM is not transmitting the secret itself. Instead, it transmits enough mathematical information for Bob to derive the same secret.

---

## Real-World Analogy

Suppose I want you and me to agree on:

```text
Secret Number = 42
```

I could simply send:

```text
42
```

but everyone would see it.

Instead, I send:

```text
A difficult puzzle
```

that only you can solve.

The puzzle might be:

```text
10 pages long
```

yet the final answer is still:

```text
42
```

ML-KEM works similarly.

```text
768-byte ciphertext
=
Puzzle

32-byte shared secret
=
Answer
```

---

# The Simplest Mental Model

```text
Bob
│
├─ Public Key (800 B)
│
▼

Alice
│
├─ Creates:
│     Shared Secret (32 B)
│     Ciphertext (768 B)
│
▼

Bob
│
├─ Uses Secret Key
│
▼

Same Shared Secret (32 B)
```

The entire purpose of ML-KEM can be summarized as:

> Alice and Bob obtain the same 32-byte secret without ever transmitting that secret over the network.

1. Which function will Bob use to generate public and private key?
In the ML-KEM reference implementation, Bob generates the public and secret key using:
```
crypto_kem_keypair(pk, sk);
```

where
```
uint8_t pk[KYBER_PUBLICKEYBYTES];
uint8_t sk[KYBER_SECRETKEYBYTES];

crypto_kem_keypair(pk, sk);
```

After execution:
```
pk = Public Key (800 bytes)
sk = Secret Key (1632 bytes)
```

2. How does Alice use Bob's public key to create the Shared Secret and Ciphertext?
```
crypto_kem_enc(ct, ss, pk);

pk = Bob's public key (800 bytes)
ct = generated ciphertext (768 bytes)
ss = generated shared secret (32 bytes)

[X] VERY IMPORTANT POINT
Alice does not create ct and ss before calling crypto_kem_enc().

means:
    Use pk as input
    And Generate:
        ct
        ss

Store results in:
    ct buffer
    ss buffer
```

Alice then:
```
    Keeps ss locally.
    Sends ct to Bob.
```
She does not send the shared secret.

3. How bob is going to retrieve the Secret Key
uses:
```
crypto_kem_dec(ss, ct, sk);

ct = Ciphertext received from Alice (768 bytes)
sk = Bob's Secret Key (1632 bytes)
ss = Recovered Shared Secret (32 bytes)
```

The important point is: **Bob does not decrypt the shared secret from the ciphertext.**
Instead:
```
    Ciphertext + Bob's Secret Key
            ↓
    Reconstruct
            ↓
    Same Shared Secret
```


Overall flow:
```
1. Bob executes:

   crypto_kem_keypair(pk, sk);

   pk = 800 bytes
   sk = 1632 bytes


2. Bob sends pk to Alice

   Network Traffic: 800 bytes


3. Alice executes:

   crypto_kem_enc(ct, ss, pk);

   ct = 768 bytes
   ss = 32 bytes


4. Alice sends ct to Bob

   Network Traffic: 768 bytes


5. Bob executes:

   crypto_kem_dec(ss, ct, sk);

   ss = 32 bytes


Result:

Alice's ss == Bob's ss
```
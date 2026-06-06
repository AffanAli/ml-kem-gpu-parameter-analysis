"""
ML-KEM-512 Parameters
"""

# Polynomial degree
KYBER_N = 256

# Prime modulus used for modular arithmetic
KYBER_Q = 3329

# Size of seeds, hashes and shared secrets (bytes)
KYBER_SYMBYTES = 32
KYBER_SSBYTES = 32

# Serialized polynomial size (bytes)
KYBER_POLYBYTES = 384

# ML-KEM-512 uses k=2 polynomial vectors
KYBER_K = 2

# Serialized polyvec size (bytes)
KYBER_POLYVECBYTES = KYBER_K * KYBER_POLYBYTES

# Noise parameter used during secret generation
KYBER_ETA1 = 3

# Noise parameter used during encryption
KYBER_ETA2 = 2

# Compressed polynomial size (bytes)
KYBER_POLYCOMPRESSEDBYTES = 128

# Compressed polyvec size (bytes)
KYBER_POLYVECCOMPRESSEDBYTES = KYBER_K * 320

# Message size for IND-CPA encryption
KYBER_INDCPA_MSGBYTES = KYBER_SYMBYTES

# Public key size (bytes)
KYBER_INDCPA_PUBLICKEYBYTES = (
    KYBER_POLYVECBYTES + KYBER_SYMBYTES
)

# Secret key size (bytes)
KYBER_INDCPA_SECRETKEYBYTES = (
    KYBER_POLYVECBYTES
)

# Ciphertext size (bytes)
KYBER_INDCPA_BYTES = (
    KYBER_POLYVECCOMPRESSEDBYTES
    + KYBER_POLYCOMPRESSEDBYTES
)

# ML-KEM public key size
KYBER_PUBLICKEYBYTES = (
    KYBER_INDCPA_PUBLICKEYBYTES
)

# ML-KEM secret key size
KYBER_SECRETKEYBYTES = (
    KYBER_INDCPA_SECRETKEYBYTES
    + KYBER_INDCPA_PUBLICKEYBYTES
    + 2 * KYBER_SYMBYTES
)

# ML-KEM ciphertext size
KYBER_CIPHERTEXTBYTES = (
    KYBER_INDCPA_BYTES
)
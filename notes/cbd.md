The generated coefficients follow the expected centered binomial distribution. Coefficients near zero occur most frequently, while coefficients near the distribution boundaries occur least frequently.

So this activity is not a performance benchmark. It is: Statistical correctness validation of the ML-KEM noise generation process.

Notebook: `notebook/coding_testing/cdb_benchmarking`

We are sampling around 100 polynomials. In fact
    MPS distribution = CUDA distribution = CPU distribution
should be identical.
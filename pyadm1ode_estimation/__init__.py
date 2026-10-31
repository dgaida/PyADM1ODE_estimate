"""PyADM1ODE state estimation."""

import os as _os

# Cap the BLAS thread pool BEFORE numpy is imported anywhere in this package.
#
# This is a correctness guard, not a performance tweak. OpenBLAS sizes its internal buffer
# pool from the compiled-in maximum core count at library init, and on this machine that
# costs **3997 MB of commit charge per process** for nothing but `import numpy` — measured,
# with a resident set of only 31 MB, so it is reserved and never touched. Setting
# OPENBLAS_NUM_THREADS=1 brings the same import down to **16 MB**.
#
# Why it matters: the tuning runs 30 worker processes. Uncapped, that is ~124 GB of commit
# charge, which exhausted a 255 GB commit limit and froze the machine hard, with
# "OpenBLAS error: Memory allocation still failed after 10 retries" in the log.
#
# It has to live here, in the top-level package, because a guard inside a submodule runs
# only after the parent packages have already pulled in numpy. `setdefault` leaves an
# explicit choice by the caller untouched. Every worker is single-threaded by design: the
# parallelism comes from running many episodes at once, so multi-threaded BLAS on top would
# only oversubscribe the CPU as well.
for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    _os.environ.setdefault(_var, "1")

# hypre device smoke test

`device_smoke` is a diagnostic driver for `GinkgoTpl::hypre`
(`gko::ext::hypre::solver::Pcg`), exercising the device path (a CUDA Ginkgo
executor together with a GPU-enabled hypre), which `hypre/test` cannot:
those tests run only on `gko::ReferenceExecutor`.

It is **not** a ctest test: it needs a GPU, and `hypre/tools/CMakeLists.txt`
builds it but never registers it with `ctest`. Run it by hand, or from a job
script.

## What it does

In one process, on one executor: builds the 3D 7-point Laplacian as a
distributed matrix over `MPI_COMM_WORLD`, solves it with hypre's PCG
preconditioned by BoomerAMG (PMIS coarsening, matching the benchmark
settings), verifies the solution two ways without a second hypre solver
(hypre's policy is process-global, so a host-side reference solve is not an
option here), and prints one `key: value` block per repetition. Exit status
is non-zero if the solve did not converge, the independently recomputed
residual is too large, or the two verification solutions disagree too much.

## Running it on Daint (GH200)

Build with `GKO_TPL_BUILD_TESTS=ON` (the default) against a CUDA-enabled
Ginkgo and a GPU-enabled hypre; this also builds `hypre_device_smoke` in
`build/hypre/tools/`.

Run it **twice, once per executor, in separate job steps**, and diff the two
outputs. Since hypre's policy is process-global, do not pass both
`--executor` values in one job step or one process.

```sh
# Reference (host) run, for the baseline to diff against.
srun -n1 --gpus-per-task=0 \
    ./build/hypre/tools/hypre_device_smoke \
    --executor reference --grid 40 --reps 3 \
    > reference.out

# Device run: the one this tool exists to exercise.
srun -n1 --gpus-per-task=1 \
    ./build/hypre/tools/hypre_device_smoke \
    --executor cuda --grid 40 --reps 3 \
    > cuda.out

diff reference.out cuda.out
```

Add `-n3` (and drop `--gpus-per-task`, or set it to match ranks per node) to
also check the distributed path, e.g. `srun -n3 ...`.

Flags:

| Flag | Default | Meaning |
|---|---|---|
| `--executor` | `reference` | `reference`, `omp`, or `cuda`: which Ginkgo executor runs the whole generate-and-solve pipeline |
| `--grid` | `40` | Grid points per side of the 3D 7-point Laplacian; rows = `grid`^3 |
| `--reps` | `1` | Number of independent generate-and-solve repetitions |
| `--relax-type` | `l1-Jacobi` | BoomerAMG smoother passed to `with_relax_type`, e.g. `"l1-Jacobi"` or `"Jacobi"` |

The benchmark preset this driver otherwise mirrors uses `"Jacobi"`
(relaxation type 0), which has no hypre device implementation and crashes a
`--executor cuda` run inside hypre's solve; `l1-Jacobi` is the closest
device-capable match, so it is the default here. Pass `--relax-type Jacobi`
explicitly to reproduce the benchmark preset exactly, on a host executor
only. See `gko_tpl/hypre/detail/options.hpp`'s `relax_type_runs_on_device`
for which types are device-capable.

## Reading the output

Each repetition prints a block of `key: value` lines (times in milliseconds,
fixed precision; norms and residuals in scientific notation), followed by a
blank line:

```
executor: cuda
ranks: 1
grid: 40
rows: 64000
relax_type: l1-Jacobi
rep: 1
iterations: 12
converged: 1
hypre_residual_norm: 3.1234567890e-09
true_relative_residual: 3.1300000000e-09
max_diff_vs_ginkgo_cg: 8.4200000000e-06
import_time_ms: 42.113
setup_time_ms: 118.902
solve_time_ms: 9.447
hierarchy_levels: 6
hierarchy_rows: 64000 19345 4821 980 187 22
rep_status: ok
```

- `true_relative_residual` is computed independently of hypre, with Ginkgo's
  own operations on the same executor (`||b - A*x||_2 / ||b||_2`); compare
  it against `hypre_residual_norm` as a sanity check.
- `max_diff_vs_ginkgo_cg` is the largest elementwise difference between
  hypre's solution and an independent `gko::solver::Cg` solve of the same
  system (no hypre involved) — the correctness signal in place of a
  host-side hypre reference solve.
- `rep_status` reflects the same three checks that decide the process exit
  status.

**The two runs are meant to be diffed.** With the reference and cuda
executors given the same `--grid`/`--reps`/`--relax-type`, `iterations`,
`hypre_residual_norm`, `true_relative_residual`, and `max_diff_vs_ginkgo_cg`
should line up closely between the two output files; a `diff` clean apart
from `executor:` and the timing fields is the expected, healthy result.

**Hierarchies may differ across reps on the device — not a failure.**
BoomerAMG's PMIS coarsening can draw a different hierarchy at every setup on
the device (see `Pcg::get_hierarchy()`'s documentation), so
`hierarchy_levels`/`hierarchy_rows` can legitimately vary between reps
within the same `cuda.out`; it does not affect `rep_status`, which tracks
only convergence and the two solution checks.

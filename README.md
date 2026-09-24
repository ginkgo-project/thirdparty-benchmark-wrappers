# Ginkgo third-party benchmark wrappers

Wrappers that run third-party solvers as Ginkgo `LinOp`s, so that Ginkgo
benchmarks and examples can compare Ginkgo against other libraries on the same
matrices, vectors and executors.

Each wrapped library is a separate, header-only component with its own CMake
target, `GinkgoTpl::<component>`, which looks only for its own library. The
components contain no compiled Ginkgo code, so they can be used from inside
Ginkgo's own build tree without linking a second Ginkgo into the executable.

| Component | Target | Wraps |
|---|---|---|
| `petsc` | `GinkgoTpl::petsc` | PETSc's KSP solvers and preconditioners, including hypre BoomerAMG through PETSc |
| `hypre` | `GinkgoTpl::hypre` | hypre's ParCSR PCG preconditioned by BoomerAMG |

`spgemm/` holds older SpGEMM wrappers (nsparse, AC-SpGEMM, spECK, Kokkos
Kernels). They only build inside Ginkgo's own build and are not part of the
CMake package.

## Requirements

- CMake 3.21 or newer and a C++17 compiler.
- MPI.
- For `petsc`: PETSc with `double` scalars, found through pkg-config. Set
  `PETSC_DIR` (and `PETSC_ARCH` for an in-place PETSc build), or add PETSc's
  `lib/pkgconfig` to `PKG_CONFIG_PATH`. Solving on the GPU needs a
  CUDA-enabled PETSc.
- For `hypre`: hypre 2.32.0 or newer. Set `HYPRE_ROOT` to its installation, or
  to a PETSc prefix built with `--download-hypre` (PETSc installs hypre into
  its own prefix, so `HYPRE_ROOT=$PETSC_DIR` works).

The PETSc component is tested with PETSc 3.23 on the CPU. The CUDA path is
not tested yet.

## Building and testing

```sh
cmake -S . -B build -DGinkgo_DIR=<Ginkgo build dir or <prefix>/lib/cmake/Ginkgo>
cmake --build build -j
OMP_NUM_THREADS=1 ctest --test-dir build --output-on-failure
cmake --install build --prefix <prefix>
```

| Option | Default | Meaning |
|---|---|---|
| `GKO_TPL_BUILD_PETSC` | `ON` | Build the `petsc` component; a missing PETSc is an error, or skips the component when used as a subproject |
| `GKO_TPL_BUILD_HYPRE` | `ON` | Build the `hypre` component; a missing hypre is an error, or skips the component when used as a subproject |
| `GKO_TPL_BUILD_TESTS` | `ON` when top-level | Build the tests |
| `GKO_TPL_INSTALL` | `ON` when top-level | Generate install rules and the package config |

The tests use an installed GoogleTest if there is one and download it
otherwise; for an offline build, set `FETCHCONTENT_SOURCE_DIR_GOOGLETEST` to a
GoogleTest source tree. The distributed tests run on 3 MPI ranks; on a machine
with fewer cores, pass launcher flags through `MPIEXEC_PREFLAGS`, e.g.
`-DMPIEXEC_PREFLAGS=--oversubscribe` for Open MPI.

## Using the wrappers

From an installed package:

```cmake
find_package(GinkgoTplWrappers REQUIRED COMPONENTS petsc hypre)
target_link_libraries(app PRIVATE GinkgoTpl::petsc GinkgoTpl::hypre)
```

The installed package remembers the PETSc and the hypre it was built against
and uses them when `PETSC_DIR` and `HYPRE_ROOT` are not set. An executable
that links both `GinkgoTpl::petsc` and `GinkgoTpl::hypre` must use a single
hypre, or it ends up with two copies of hypre's symbols; pointing `HYPRE_ROOT`
at the PETSc prefix does that, since PETSc installs hypre into its own
prefix. This is verified to work: the combined executable builds, runs, and
both solvers converge to the same solution.

From the same build tree, e.g. in a project that already builds or finds
Ginkgo, with FetchContent or `add_subdirectory`:

```cmake
include(FetchContent)
FetchContent_Declare(
    ginkgo_tpl_wrappers
    GIT_REPOSITORY https://github.com/ginkgo-project/thirdparty-benchmark-wrappers.git
    GIT_TAG <commit>
    GIT_SUBMODULES "" # the spgemm/ submodules are not part of the package
)
FetchContent_MakeAvailable(ginkgo_tpl_wrappers)
if(TARGET GinkgoTpl::petsc)
    target_link_libraries(app PRIVATE GinkgoTpl::petsc)
endif()
if(TARGET GinkgoTpl::hypre)
    target_link_libraries(app PRIVATE GinkgoTpl::hypre)
endif()
```

If a `Ginkgo::ginkgo` target already exists, it is used instead of searching
for another Ginkgo. Tests and install rules are off by default in this case,
and a component whose library is not found is skipped instead of failing the
configuration, so check that its target exists. `GIT_SUBMODULES ""` needs the
consuming project to require CMake 3.16 or newer; otherwise FetchContent
clones every submodule, including Kokkos and Kokkos Kernels.

### PETSc KSP solver

`gko::ext::petsc::solver::Ksp` imports the system matrix into PETSc when it
is generated, then solves with PETSc on the memory of Ginkgo's vectors.
PETSc is configured through its usual options:

```cpp
#include <iostream>

#include <ginkgo/ginkgo.hpp>
#include <gko_tpl/petsc.hpp>

int main(int argc, char* argv[])
{
    const gko::experimental::mpi::environment mpi_env(argc, argv);
    // Initializes PETSc after MPI; every Ksp must be destroyed before it.
    const gko::ext::petsc::environment petsc_env;

    using ksp = gko::ext::petsc::solver::Ksp<double, gko::int32>;
    auto exec = gko::ReferenceExecutor::create();
    // A matrix::Csr, or an experimental::distributed::Matrix, see below.
    std::shared_ptr<gko::LinOp> A = /* ... */;

    auto solver = ksp::build()
                      .with_options("-ksp_type cg -ksp_rtol 1e-8 "
                                    "-pc_type hypre -pc_hypre_type boomeramg")
                      .on(exec)
                      ->generate(A);
    solver->apply(b, x);

    std::cout << solver->get_num_iterations() << " iterations, "
              << (solver->has_converged() ? "converged" : "not converged")
              << '\n';
}
```

- **Matrices:** a `matrix::Csr` is solved on `PETSC_COMM_SELF` with
  `matrix::Dense` vectors. An `experimental::distributed::Matrix` is solved on
  its communicator with `experimental::distributed::Vector`s, and must be read
  with a row partition whose parts are contiguous and ordered by rank. Only a
  single right-hand side is supported.
- **Executors:** on a `CudaExecutor` with a CUDA-enabled PETSc, the matrix is a
  `MATAIJCUSPARSE` and PETSc works on the device vectors in place. Otherwise
  the matrix is a host `MATAIJ`, and device vectors are copied to and from
  the host.
- **Options** apply to this solver only, since each instance reads them under
  its own prefix. Global options such as `-log_view` have no effect here; set
  them through the `PETSC_OPTIONS` environment variable instead.
- **As a preconditioner:** with `-ksp_type preonly`, each apply is one
  application of the PETSc preconditioner, so the `Ksp` can precondition a
  Ginkgo solver.
- **Timing:** `get_import_time()` covers copying the matrix into PETSc and
  creating the KSP, `get_setup_time()` covers `KSPSetUp`, e.g. the BoomerAMG
  setup. Both phases end with a device synchronization and an MPI barrier.

The solver can also be configured from JSON under the type name
`ext::petsc::solver::Ksp`, after adding its configuration map to the
registry:

```cpp
const gko::config::registry reg(ksp::get_config_map());
```

```json
{
  "type": "ext::petsc::solver::Ksp",
  "options": "-ksp_type preonly -pc_type hypre -pc_hypre_type boomeramg"
}
```

When comparing against hypre through PETSc, note that PETSc picks different
BoomerAMG defaults for device and host matrices: PMIS coarsening, extended+i
interpolation and l1-scaled Jacobi on the device, Falgout coarsening and
classical interpolation on the host. Pass the BoomerAMG options explicitly so
the configuration is the same on both.

### hypre PCG with BoomerAMG

`gko::ext::hypre::solver::Pcg` runs hypre's own ParCSR PCG, preconditioned by
BoomerAMG, directly on Ginkgo's matrix and vector memory — no PETSc in
between:

```cpp
#include <cmath>
#include <iostream>

#include <ginkgo/ginkgo.hpp>
#include <gko_tpl/hypre.hpp>

int main(int argc, char* argv[])
{
    const gko::experimental::mpi::environment mpi_env(argc, argv);
    // Initializes hypre after MPI; every Pcg must be destroyed before it.
    const gko::ext::hypre::environment hypre_env;

    auto exec = gko::ReferenceExecutor::create();
    using pcg = gko::ext::hypre::solver::Pcg<double, gko::int32>;
    // A matrix::Csr, or an experimental::distributed::Matrix, see below.
    std::shared_ptr<gko::LinOp> A = /* ... */;

    auto solver = pcg::build()
                      .with_tolerance(1e-8)
                      .with_max_iters(200)
                      .with_coarsen_type("PMIS")
                      .with_relax_type("Jacobi")
                      .on(exec)
                      ->generate(A);
    solver->apply(b, x);

    std::cout << solver->get_num_iterations() << " iterations, "
              << (solver->has_converged() ? "converged" : "not converged")
              << '\n';
}
```

- **Matrices:** the same as `Ksp` — a `matrix::Csr` on `MPI_COMM_SELF` with
  `matrix::Dense` vectors, or an `experimental::distributed::Matrix` read with
  a row partition whose parts are contiguous and ordered by rank, solved with
  `experimental::distributed::Vector`s. Only a single right-hand side is
  supported.
- **The matrix is aliased, not copied.** This is the reason the component
  exists: hypre's `hypre_ParCSRMatrix` points directly at the Ginkgo matrix's
  `diag`/`offd` arrays (and at the index map's remote global indices for the
  non-local block), so import time is hypre's own conversion cost, not a
  translation through PETSc, and nothing is staged through the host on any
  executor. `apply()` hands `b` and `x` to hypre over Ginkgo's own arrays the
  same way. Because the alias borrows the matrix's memory, the solver keeps a
  `shared_ptr` to the system matrix alive for as long as it needs it — unlike
  `Ksp`, which copies the matrix into PETSc's own triplets and lets the
  Ginkgo matrix go.
- **hypre's memory location and execution policy are process-global**
  (`HYPRE_SetMemoryLocation`/`HYPRE_SetExecutionPolicy`), not per solver.
  Generating a solver on a host executor and another on a device executor in
  the same process throws instead of silently running with the wrong policy.
- **Parameters:**

  | Parameter | Type | Note |
  |---|---|---|
  | `tolerance` | `double`, default `1e-8` | relative residual tolerance |
  | `max_iters` | `int`, default `100` | |
  | `two_norm` | `bool`, default `true` | stop on the unpreconditioned relative 2-norm, the quantity Ginkgo's `ResidualNorm` uses; hypre's own default is the preconditioned norm, which is not comparable |
  | `coarsen_type` | `std::string` | BoomerAMG coarsening, e.g. `"PMIS"`, `"HMIS"`, `"Falgout"`; empty (the default) keeps hypre's own default |
  | `interp_type` | `std::string` | BoomerAMG interpolation, e.g. `"ext+i"`, `"classical"`; empty keeps hypre's default |
  | `relax_type` | `std::string` | BoomerAMG smoother, e.g. `"Jacobi"`, `"l1-Jacobi"`, `"hybrid-GS"`; empty keeps hypre's default |
  | `relax_weight` | `double` | negative (the default) keeps hypre's default |
  | `num_sweeps` | `int` | zero (the default) keeps hypre's default |
  | `max_levels` | `int` | hypre counts the finest level, Ginkgo does not; zero keeps hypre's default |
  | `max_coarse_size` | `int` | hypre's equivalent of Ginkgo's `min_coarse_rows`; zero keeps hypre's default |
  | `strong_threshold` | `double` | negative keeps hypre's default |
  | `print_statistics` | `bool`, default `false` | BoomerAMG's setup statistics |
  | `amg_setup`, `krylov_setup` | `std::function<void(HYPRE_Solver)>` | called with the raw `HYPRE_Solver` after the typed setters, before setup; not expressible in JSON |

  Unknown `coarsen_type`/`interp_type`/`relax_type` names throw and name the
  accepted values.

  `amg_setup` and `krylov_setup` run on every rank, between collective calls
  in generation. They must therefore do the same thing on every rank and must
  not throw: an exception escaping one of them unwinds only the rank that
  threw and leaves the others waiting in the next collective, so the job
  hangs rather than failing.
- **Diagnostics:** `get_num_iterations()`, `get_residual_norm()`,
  `has_converged()`, `get_import_time()`, `get_setup_time()` — the same
  surface as `Ksp` — plus `get_hierarchy()`, which returns the number of
  levels and the rows per level BoomerAMG's setup drew. It is populated on
  every supported hypre, 2.32 included: the underlying
  `HYPRE_BoomerAMGGetGridHierarchy` is public API in both 2.32 and 3.2, and
  the two report the same hierarchy for the same setup.
- **Two hypre copies in one executable:** an executable that links both
  `GinkgoTpl::petsc` and `GinkgoTpl::hypre` must use a single hypre, or it
  ends up with two copies of hypre's symbols in the one binary. Point
  `HYPRE_ROOT` at the PETSc prefix so both components resolve to the same
  hypre; this is verified to work, including both solvers converging to the
  same solution in one process.
- **CUDA:** the component is written for a `CudaExecutor` and compiles there,
  but has not been run — no GPU-built hypre exists in the environments this
  component has been tested in, so the CUDA path is unverified, like `Ksp`'s.
  Generating on a Ginkgo device executor against a hypre built *without* GPU
  support is rejected at generation: such a hypre maps `HYPRE_MEMORY_DEVICE`
  to `HYPRE_MEMORY_HOST` internally and would read the device pointers as
  host memory without reporting anything.

The solver can also be configured from JSON under the type name
`ext::hypre::solver::Pcg`, after adding its configuration map to the
registry, the same way as `Ksp` above. `amg_setup` and `krylov_setup` are
`std::function`s and cannot be expressed in JSON, so they are not
configurable this way.

## Development

The repository uses the same [pre-commit](https://pre-commit.com) hooks as
Ginkgo. Install them once per clone:

```sh
pre-commit install
```

Commits are then formatted with clang-format (C, C++ and CUDA) and gersemi
(CMake), and every C, C++ and CUDA file gets an SPDX license header. Pull
requests are checked by the same hooks in CI.

## License

BSD-3-Clause, see [LICENSE](LICENSE).

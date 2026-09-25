// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

// Diagnostic driver for GinkgoTpl::hypre (gko::ext::hypre::solver::Pcg).
//
// hypre/test only ever runs on gko::ReferenceExecutor, so the component's
// device path (a CUDA-enabled Ginkgo executor together with a GPU-enabled
// hypre) has never actually been executed. This is the first thing that
// does: it generates and solves once, on one executor, and prints a
// canonical, greppable summary.
//
// hypre's memory location and execution policy are process-global, and
// gko_tpl/hypre/detail/policy.hpp deliberately throws if one process tries
// to use both a host and a device executor, so this driver never mixes
// executors within a run. It always does the whole generate-and-solve
// pipeline on exactly one executor, chosen with --executor.
//
// Intended use: run it twice, once per executor, and diff the two outputs,
// e.g. on CSCS Daint (GH200), see tools/README.md for the exact srun
// commands:
//
//   device_smoke --executor reference --grid 40 > reference.out
//   device_smoke --executor cuda      --grid 40 > cuda.out
//   diff reference.out cuda.out
//
// Repeating with --reps > 1 makes the device path's own nondeterminism
// visible: BoomerAMG's PMIS coarsening may draw a different hierarchy at
// every setup on the device (see Pcg::get_hierarchy()'s documentation), so
// two device reps of the *same run* can legitimately disagree with each
// other, not just with the reference executor's rep. That is an expected
// observation this driver is meant to surface, not a bug in it.
//
// Output format: one "key: value" line per field, so a diff between two
// runs produces isolated hunks instead of reflowing whole blocks. Wall-clock
// and hypre-internal times are printed in milliseconds with fixed
// precision; norms and residuals in scientific notation. Each repetition
// prints its own block, terminated by a blank line.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>

#include <mpi.h>

#include <execinfo.h>
#include <unistd.h>

#include <ginkgo/ginkgo.hpp>

#include <gko_tpl/hypre/environment.hpp>
#include <gko_tpl/hypre/pcg.hpp>

#include "common/test/laplacian.hpp"


namespace {


using value_type = double;
using local_index_type = gko::int32;
using global_index_type = gko::int64;
using pcg_type = gko::ext::hypre::solver::Pcg<value_type, local_index_type,
                                              global_index_type>;
using dense = gko::matrix::Dense<value_type>;
using dist_mtx =
    gko::experimental::distributed::Matrix<value_type, local_index_type,
                                           global_index_type>;
using dist_vec = gko::experimental::distributed::Vector<value_type>;
using partition_type =
    gko::experimental::distributed::Partition<local_index_type,
                                              global_index_type>;
using md_type = gko::matrix_data<value_type, global_index_type>;


struct options {
    std::string executor_name = "reference";
    global_index_type grid = 40;
    int reps = 1;
    // Defaults to l1-Jacobi, not the "Jacobi" (hypre relax_type 0) the
    // benchmark preset this driver otherwise mirrors uses: hypre has no
    // device implementation for plain weighted Jacobi (see
    // gko_tpl/hypre/detail/options.hpp's relax_type_runs_on_device), so a
    // device run with it crashes inside hypre's solve instead of reporting
    // an error; see tools/README.md.
    std::string relax_type = "l1-Jacobi";
};


bool parse_options(int argc, char** argv, options& opts, std::string& error)
{
    for (int i = 1; i < argc; i++) {
        const std::string arg = argv[i];
        auto next_value = [&]() -> std::optional<std::string> {
            if (i + 1 >= argc) {
                return std::nullopt;
            }
            return std::string(argv[++i]);
        };
        try {
            if (arg == "--executor") {
                auto v = next_value();
                if (!v) {
                    error = "--executor needs a value";
                    return false;
                }
                opts.executor_name = *v;
            } else if (arg == "--grid") {
                auto v = next_value();
                if (!v) {
                    error = "--grid needs a value";
                    return false;
                }
                opts.grid = static_cast<global_index_type>(std::stoll(*v));
            } else if (arg == "--reps") {
                auto v = next_value();
                if (!v) {
                    error = "--reps needs a value";
                    return false;
                }
                opts.reps = std::stoi(*v);
            } else if (arg == "--relax-type") {
                auto v = next_value();
                if (!v) {
                    error = "--relax-type needs a value";
                    return false;
                }
                opts.relax_type = *v;
            } else if (arg == "-h" || arg == "--help") {
                error.clear();
                return false;
            } else {
                error = "unknown argument '" + arg + "'";
                return false;
            }
        } catch (const std::exception&) {
            error = "invalid value for " + arg;
            return false;
        }
    }
    if (opts.executor_name != "reference" && opts.executor_name != "omp" &&
        opts.executor_name != "cuda") {
        error = "--executor must be reference, omp, or cuda";
        return false;
    }
    if (opts.grid <= 0) {
        error = "--grid must be positive";
        return false;
    }
    if (opts.reps <= 0) {
        error = "--reps must be positive";
        return false;
    }
    return true;
}


void print_usage()
{
    std::cerr
        << "usage: device_smoke [--executor reference|omp|cuda] [--grid n] "
           "[--reps n] [--relax-type name]\n"
        << "  --executor  which Ginkgo executor to run the whole "
           "generate-and-solve pipeline on (default: reference)\n"
        << "  --grid      grid points per side of the 3D 7-point Laplacian; "
           "rows = grid^3 (default: 40)\n"
        << "  --reps      number of independent generate-and-solve "
           "repetitions (default: 1)\n"
        << "  --relax-type  BoomerAMG smoother passed to with_relax_type, "
           "e.g. \"l1-Jacobi\" or \"Jacobi\" (default: l1-Jacobi; see "
           "tools/README.md for why this differs from the Jacobi the "
           "benchmark preset otherwise mirrors)\n";
}


// gko::CudaExecutor::create is declared unconditionally (it is part of
// Ginkgo's core interface), so this always compiles. Checked against an
// actual CUDA-less Ginkgo build: the *create* call below does NOT itself
// throw when Ginkgo was built without CUDA: it happily returns an
// executor object, and the throw only happens lazily, on the first real
// device operation (its implementation is GKO_NOT_COMPILED(cuda), see
// <ginkgo/core/base/exception.hpp>/exception_helpers.hpp, but that macro
// only fires from the actual device-op entry points, e.g. raw_alloc, not
// from create() itself). So a trivial allocation is forced right here,
// immediately after create(), specifically so that this function's own
// try/catch is what turns a missing backend into a clear message naming
// the likely cause, instead of a raw_alloc failure surfacing later, deep
// inside partition or matrix construction with a less obvious backtrace.
std::shared_ptr<gko::Executor> create_executor(const std::string& name)
{
    try {
        std::shared_ptr<gko::Executor> exec;
        if (name == "reference") {
            exec = gko::ReferenceExecutor::create();
        } else if (name == "omp") {
            exec = gko::OmpExecutor::create();
        } else if (name == "cuda") {
            exec = gko::CudaExecutor::create(0, gko::OmpExecutor::create());
        } else {
            std::cerr << "error: unknown --executor '" << name
                      << "'; expected reference, omp, or cuda\n";
            return nullptr;
        }
        // Forces the lazy NotCompiled check described above.
        gko::array<double> probe{exec, 1};
        return exec;
    } catch (const gko::NotCompiled& e) {
        std::cerr
            << "error: creating the '" << name
            << "' executor failed: " << e.what() << "\n"
            << "likely cause: this Ginkgo build does not have that backend "
               "compiled in. For --executor cuda this almost always means "
               "Ginkgo was configured with GINKGO_BUILD_CUDA=OFF (or a "
               "CUDA-disabled build is the one -DGinkgo_DIR pointed at); "
               "rebuild Ginkgo with CUDA enabled and re-point this build at "
               "it.\n";
        return nullptr;
    } catch (const gko::Error& e) {
        std::cerr << "error: creating the '" << name
                  << "' executor failed: " << e.what() << "\n";
        return nullptr;
    }
}


// Reads back a 1x1 gko::matrix::Dense scalar that may live on a device
// executor: Dense::at() is only meaningful for host-accessible memory, so
// the scalar is cloned to the host first. On a host executor this clone is
// cheap (its own master).
double read_scalar(std::shared_ptr<const gko::Executor> exec, const dense* d)
{
    auto host = gko::clone(exec->get_master(), d);
    return host->at(0, 0);
}


// max_i |a_i - b_i| over every row this rank owns, reduced across `comm`.
// Brings both local vectors to the host: there is no reduction operation
// for an elementwise max on a Ginkgo LinOp, so this is done by hand, as a
// diagnostic tool rather than hot solver code.
double max_abs_difference(MPI_Comm comm,
                          std::shared_ptr<const gko::Executor> exec,
                          const dist_vec* a, const dist_vec* b)
{
    const auto host = exec->get_master();
    const auto host_a = gko::clone(host, a->get_local_vector());
    const auto host_b = gko::clone(host, b->get_local_vector());
    double local_max = 0.0;
    const auto n = host_a->get_size()[0];
    for (gko::size_type i = 0; i < n; i++) {
        local_max =
            std::max(local_max, std::abs(host_a->at(i, 0) - host_b->at(i, 0)));
    }
    double global_max = local_max;
    MPI_Allreduce(MPI_IN_PLACE, &global_max, 1, MPI_DOUBLE, MPI_MAX, comm);
    return global_max;
}


}  // namespace


// Prints a native backtrace when the process faults. The device path of this
// component runs only on machines this code is not developed on, where a
// debugger is awkward (no TTY under the batch launcher) and core files are
// often disabled, so a segfault would otherwise arrive as an exit status and
// nothing else. Built with -rdynamic so the frames carry names.
extern "C" void crash_handler(int signal_number)
{
    void* frames[64];
    const int count = backtrace(frames, 64);
    const char* message = "\n=== device_smoke caught a fatal signal ===\n";
    ssize_t ignored = write(STDERR_FILENO, message, std::strlen(message));
    (void)ignored;
    backtrace_symbols_fd(frames, count, STDERR_FILENO);
    // Re-raise with the default handler so the exit status still reports the
    // signal and a core is written where that is enabled.
    std::signal(signal_number, SIG_DFL);
    std::raise(signal_number);
}


void install_crash_handler()
{
    std::signal(SIGSEGV, crash_handler);
    std::signal(SIGBUS, crash_handler);
    std::signal(SIGFPE, crash_handler);
    std::signal(SIGABRT, crash_handler);
}


int main(int argc, char** argv)
{
    gko::experimental::mpi::environment mpi_env(argc, argv);

    options opts;
    std::string parse_error;
    if (!parse_options(argc, argv, opts, parse_error)) {
        if (!parse_error.empty()) {
            std::cerr << "error: " << parse_error << "\n";
        }
        print_usage();
        return 2;
    }

    install_crash_handler();

    auto exec = create_executor(opts.executor_name);
    if (!exec) {
        return 2;
    }

    gko::experimental::mpi::communicator comm(MPI_COMM_WORLD);
    const auto rank = comm.rank();

    // Progress markers, flushed, on stderr: stdout stays the diffable
    // key: value block. A crash then shows which phase it died in, which is
    // the only thing a remote debugger has when a run produces no output.
    const auto mark = [rank](const char* phase) {
        std::cerr << "phase[" << rank << "]: " << phase << std::endl;
    };
    mark("executor_created");

    try {
        // hypre must be initialized before any Pcg is generated, and after
        // the executor: set_process_policy runs at generation time, not
        // here, so this order does not by itself commit to a policy.
        gko::ext::hypre::environment hypre_env;
        mark("hypre_initialized");

        const auto grid = opts.grid;
        const auto num_rows = static_cast<gko::size_type>(grid) *
                              static_cast<gko::size_type>(grid) *
                              static_cast<gko::size_type>(grid);
        auto part = gko::share(partition_type::build_from_global_size_uniform(
            exec, comm.size(), static_cast<global_index_type>(num_rows)));

        // b_i = sin(0.1 * i), the same right-hand side hypre/test uses.
        md_type b_data(gko::dim<2>{num_rows, 1});
        for (gko::size_type i = 0; i < num_rows; i++) {
            b_data.nonzeros.emplace_back(
                static_cast<global_index_type>(i), 0,
                std::sin(0.1 * static_cast<double>(i)));
        }

        int overall_status = 0;

        for (int rep = 1; rep <= opts.reps; rep++) {
            auto mtx = gko::share(dist_mtx::create(exec, comm));
            mtx->read_distributed(laplacian_3d<global_index_type>(grid), part);
            auto b = dist_vec::create(exec, comm);
            b->read_distributed(b_data, part);
            auto x0 = dist_vec::create(exec, comm);
            // read_distributed zero-fills entries missing from the data.
            x0->read_distributed(md_type(gko::dim<2>{num_rows, 1}), part);
            mark("matrix_and_vectors_built");

            auto solver = pcg_type::build()
                              .with_coarsen_type("PMIS")
                              .with_strong_threshold(0.25)
                              .with_relax_type(opts.relax_type)
                              .with_relax_weight(0.9)
                              .with_num_sweeps(2)
                              .with_tolerance(1e-8)
                              .with_max_iters(200)
                              .on(exec)
                              ->generate(mtx);
            mark("generated");

            auto x_hypre = gko::clone(x0);
            // hypre/include/gko_tpl/hypre/pcg.hpp documents that "a solve
            // that does not converge does not throw, check
            // has_converged()": apply_impl masks HYPRE_ERROR_CONV out of
            // hypre's global error flag before deciding whether to throw, so
            // a non-converging solve returns normally with has_converged()
            // false, get_num_iterations()/get_residual_norm() reporting the
            // attempt that was made, and x holding where it got to, all
            // read below. Any other hypre failure still throws, and is left
            // to propagate out of main()'s own try/catch, which reports it
            // clearly and exits non-zero; that is genuine robustness this
            // driver should not paper over with a rep-local catch.
            const auto solve_start = std::chrono::steady_clock::now();
            solver->apply(b, x_hypre);
            // Ginkgo's device executors queue work asynchronously; nothing
            // else in this loop iteration touches `exec` before the timer
            // is read, so this synchronize is what makes solve_time_ms
            // trustworthy on a device executor. hypre's own solve already
            // blocks until done (its convergence check is host-side every
            // iteration), so this is about Ginkgo-side work, not hypre's.
            exec->synchronize();
            const auto solve_end = std::chrono::steady_clock::now();
            mark("solved");
            const auto solve_time_ms =
                std::chrono::duration<double, std::milli>(solve_end -
                                                          solve_start)
                    .count();

            // True relative residual ||b - A*x||_2 / ||b||_2, computed with
            // Ginkgo's own operations on `exec`, independent of hypre's
            // internally reported residual norm.
            auto one = gko::initialize<dense>({1.0}, exec);
            auto neg_one = gko::initialize<dense>({-1.0}, exec);
            auto residual = gko::clone(b);
            mtx->apply(neg_one, x_hypre, one, residual);
            auto residual_norm = dense::create(exec, gko::dim<2>{1, 1});
            auto b_norm = dense::create(exec, gko::dim<2>{1, 1});
            residual->compute_norm2(residual_norm);
            b->compute_norm2(b_norm);
            const auto true_relative_residual =
                read_scalar(exec, residual_norm.get()) /
                read_scalar(exec, b_norm.get());

            // A second, independent solve with Ginkgo's own CG, with no
            // hypre involved at all: a correctness signal that does not
            // need a host hypre solve in this process, which the
            // process-global policy above rules out.
            auto x_ginkgo = gko::clone(x0);
            auto ginkgo_solver =
                gko::solver::Cg<value_type>::build()
                    .with_criteria(
                        gko::stop::Iteration::build().with_max_iters(5000u),
                        gko::stop::ResidualNorm<value_type>::build()
                            .with_reduction_factor(1e-10))
                    .on(exec)
                    ->generate(mtx);
            ginkgo_solver->apply(b, x_ginkgo);

            const auto max_diff = max_abs_difference(
                comm.get(), exec, x_hypre.get(), x_ginkgo.get());

            const auto& hierarchy = solver->get_hierarchy();

            // Pass/fail thresholds:
            // - true_relative_residual_tolerance: hypre's own tolerance is
            //   1e-8 on the unpreconditioned two-norm (two_norm is left at
            //   its default, true), the same quantity computed above, so an
            //   independently recomputed residual should land close to
            //   hypre's own report. 100x slack (1e-6) covers hypre stopping
            //   the iteration one step after crossing the threshold and
            //   floating-point-order differences between hypre's and
            //   Ginkgo's reductions, which matter most on the device path
            //   across many ranks.
            // - max_diff_tolerance: the independent Ginkgo CG solve above
            //   targets a reduction factor of 1e-10, two orders tighter
            //   than hypre's 1e-8, so it is the more accurate of the two;
            //   1e-4 leaves room for the remaining gap between a
            //   1e-8- and a 1e-10-accurate solution of this system, in the
            //   same spirit as (and looser than) the 1e-12-tolerance-solves
            //   / 1e-8-comparison gap in hypre/test/pcg.cpp's
            //   MatchesGinkgoCg test.
            constexpr double true_relative_residual_tolerance = 1e-6;
            constexpr double max_diff_tolerance = 1e-4;

            const bool rep_ok =
                solver->has_converged() &&
                true_relative_residual <= true_relative_residual_tolerance &&
                max_diff <= max_diff_tolerance;
            if (!rep_ok) {
                overall_status = 1;
            }

            if (rank == 0) {
                mark("verified");
                std::cout << "executor: " << opts.executor_name << "\n";
                std::cout << "ranks: " << comm.size() << "\n";
                std::cout << "grid: " << grid << "\n";
                std::cout << "rows: " << num_rows << "\n";
                std::cout << "relax_type: " << opts.relax_type << "\n";
                std::cout << "rep: " << rep << "\n";
                std::cout << "iterations: " << solver->get_num_iterations()
                          << "\n";
                std::cout << "converged: " << (solver->has_converged() ? 1 : 0)
                          << "\n";
                std::cout << std::scientific << std::setprecision(10);
                std::cout << "hypre_residual_norm: "
                          << solver->get_residual_norm() << "\n";
                std::cout << "true_relative_residual: "
                          << true_relative_residual << "\n";
                std::cout << "max_diff_vs_ginkgo_cg: " << max_diff << "\n";
                std::cout << std::fixed << std::setprecision(3);
                std::cout << "import_time_ms: "
                          << std::chrono::duration<double, std::milli>(
                                 solver->get_import_time())
                                 .count()
                          << "\n";
                std::cout << "setup_time_ms: "
                          << std::chrono::duration<double, std::milli>(
                                 solver->get_setup_time())
                                 .count()
                          << "\n";
                std::cout << "solve_time_ms: " << solve_time_ms << "\n";
                if (hierarchy.has_value()) {
                    std::cout << "hierarchy_levels: " << hierarchy->num_levels
                              << "\n";
                    std::cout << "hierarchy_rows:";
                    for (const auto r : hierarchy->rows_per_level) {
                        std::cout << " " << r;
                    }
                    std::cout << "\n";
                } else {
                    std::cout << "hierarchy_levels: 0\n";
                    std::cout << "hierarchy_rows:\n";
                }
                std::cout << "rep_status: " << (rep_ok ? "ok" : "FAIL") << "\n";
                std::cout << std::endl;
            }
        }

        return overall_status;
    } catch (const std::exception& e) {
        std::cerr << "error: device_smoke failed on rank " << rank << ": "
                  << e.what() << "\n";
        return 1;
    }
}

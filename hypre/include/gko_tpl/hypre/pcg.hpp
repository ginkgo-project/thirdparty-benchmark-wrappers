// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_TPL_HYPRE_PCG_HPP_
#define GKO_TPL_HYPRE_PCG_HPP_


#include <algorithm>
#include <chrono>
#include <functional>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <mpi.h>

#include <HYPRE_krylov.h>
#include <HYPRE_parcsr_ls.h>
#include <HYPRE_utilities.h>

#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/config/config.hpp>
#include <ginkgo/core/config/registry.hpp>
#include <ginkgo/core/distributed/vector.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>

#include <gko_tpl/hypre/detail/error.hpp>
#include <gko_tpl/hypre/detail/import.hpp>
#include <gko_tpl/hypre/detail/options.hpp>
#include <gko_tpl/hypre/detail/policy.hpp>
#include <gko_tpl/hypre/detail/timing.hpp>
#include <gko_tpl/hypre/detail/vectors.hpp>


namespace gko {
namespace ext {
namespace hypre {
namespace solver {


/**
 * hypre's ParCSR conjugate gradient, preconditioned by BoomerAMG.
 *
 * Generation has two phases, which are timed separately for benchmarks:
 * - import: the Ginkgo matrix is wrapped as a hypre ParCSR matrix and the
 *   solvers are created and configured, see get_import_time();
 * - setup: BoomerAMG's setup, see get_setup_time().
 *
 * Unlike the PETSc wrapper, the matrix is aliased from Ginkgo's arrays,
 * which the solver keeps alive; b and x are handed to hypre in place on
 * every apply. The exception is the diagonal block's columns and values,
 * copied and permuted into hypre's required diagonal-first order (see
 * detail/import.hpp), in the executor's own memory space; nothing is
 * staged through the host.
 *
 * Supported system matrices:
 * - matrix::Csr<ValueType, LocalIndexType>, solved on MPI_COMM_SELF and
 *   applied to matrix::Dense vectors;
 * - experimental::distributed::Matrix read with a row partition whose parts
 *   are contiguous and ordered by rank, solved on the matrix's communicator
 *   and applied to experimental::distributed::Vector vectors.
 *
 * hypre's memory location and execution policy are process-global: all
 * solvers in a process must share one, and generating one that needs the
 * other throws. A GPU-enabled hypre defaults to HYPRE_MEMORY_DEVICE (set by
 * HYPRE_Initialize()); every hypre object this component creates establishes
 * the policy for its own memory location first, so this is handled
 * automatically regardless of what else ran earlier in the process.
 *
 * Only a single right-hand side is supported. The input x is the initial
 * guess. A solve that does not converge does not throw, check
 * has_converged(); get_num_iterations() and get_residual_norm() then report
 * the attempt that was made, and x holds where it got to. Every other hypre
 * failure still throws.
 *
 * hypre must be initialized before generation, e.g. with
 * gko::ext::hypre::environment, and every solver must be destroyed before
 * hypre is finalized.
 *
 * @tparam ValueType  the value type, which must be hypre's scalar type double
 * @tparam LocalIndexType  the (local) index type of the system matrix
 * @tparam GlobalIndexType  the global index type of a distributed system
 *                          matrix
 */
template <typename ValueType = double, typename LocalIndexType = int32,
          typename GlobalIndexType = int64>
class Pcg : public LinOp {
    static_assert(std::is_same<ValueType, double>::value,
                  "gko::ext::hypre::solver::Pcg only supports double, the "
                  "scalar type of the supported hypre builds");

public:
    using value_type = ValueType;
    using local_index_type = LocalIndexType;
    using global_index_type = GlobalIndexType;

    class Factory;

    struct parameters_type : enable_parameters_type<parameters_type, Factory> {
        /** Relative residual tolerance of the CG iteration. */
        double GKO_FACTORY_PARAMETER_SCALAR(tolerance, 1e-8);
        /** Maximum number of CG iterations. */
        int GKO_FACTORY_PARAMETER_SCALAR(max_iters, 100);
        /**
         * Stop on the unpreconditioned relative 2-norm, the quantity Ginkgo's
         * ResidualNorm criterion uses. hypre's own default is the
         * preconditioned norm, which is not comparable.
         */
        bool GKO_FACTORY_PARAMETER_SCALAR(two_norm, true);
        /** BoomerAMG coarsening, e.g. "PMIS"; empty keeps hypre's default. */
        std::string GKO_FACTORY_PARAMETER_SCALAR(coarsen_type, std::string{});
        /** BoomerAMG interpolation, e.g. "ext+i". */
        std::string GKO_FACTORY_PARAMETER_SCALAR(interp_type, std::string{});
        /**
         * BoomerAMG smoother, e.g. "l1-Jacobi".
         *
         * "Jacobi" (relaxation type 0) has no hypre device implementation;
         * generating on a device executor with it throws here instead of
         * crashing later inside hypre's solve, see
         * detail::relax_type_runs_on_device. Use "l1-Jacobi" on a device.
         */
        std::string GKO_FACTORY_PARAMETER_SCALAR(relax_type, std::string{});
        /** Smoother weight; negative keeps hypre's default. */
        double GKO_FACTORY_PARAMETER_SCALAR(relax_weight, -1.0);
        /** Smoother sweeps per level; zero keeps hypre's default. */
        int GKO_FACTORY_PARAMETER_SCALAR(num_sweeps, 0);
        /** Maximum number of levels, counting the finest; zero keeps hypre's.
         */
        int GKO_FACTORY_PARAMETER_SCALAR(max_levels, 0);
        /** Largest coarse problem solved directly; zero keeps hypre's. */
        int GKO_FACTORY_PARAMETER_SCALAR(max_coarse_size, 0);
        /** Strength threshold; negative keeps hypre's default. */
        double GKO_FACTORY_PARAMETER_SCALAR(strong_threshold, -1.0);
        /** Print BoomerAMG's setup statistics, including the hierarchy. */
        bool GKO_FACTORY_PARAMETER_SCALAR(print_statistics, false);
        /**
         * Called with the BoomerAMG solver after the typed setters.
         *
         * Generation is collective on the system matrix's communicator, and
         * this callback runs between collective calls, on every rank. It must
         * therefore do the same thing on every rank and must not throw: an
         * exception escaping it unwinds only the rank that threw, leaving the
         * others waiting in the next collective, so the job hangs instead of
         * failing. Report a problem by returning normally and checking
         * afterwards, or abort the job.
         */
        std::function<void(HYPRE_Solver)> GKO_FACTORY_PARAMETER_SCALAR(
            amg_setup, nullptr);
        /**
         * Called with the PCG solver after the typed setters. The same
         * collective constraints as amg_setup apply: it runs between
         * collective calls, so an exception thrown from it hangs the job
         * rather than failing it.
         */
        std::function<void(HYPRE_Solver)> GKO_FACTORY_PARAMETER_SCALAR(
            krylov_setup, nullptr);
    };
    GKO_ENABLE_LIN_OP_FACTORY(Pcg, parameters, Factory);
    GKO_ENABLE_BUILD_METHOD(Factory);

    /** Iterations of the most recent apply, or 0 before the first. */
    size_type get_num_iterations() const;

    /** hypre's final relative residual norm of the most recent apply. */
    double get_residual_norm() const;

    /** Whether the most recent apply converged. */
    bool has_converged() const;

    /**
     * Duration of the import phase, see the class documentation. The phase
     * starts when this rank entered generation and ends with a
     * synchronization of hypre's device work, if any, and a barrier on the
     * solver's communicator, so it includes waiting for the slowest rank.
     */
    std::chrono::nanoseconds get_import_time() const;

    /**
     * Duration of BoomerAMG's setup. It starts where the import phase ends
     * and ends with the same kind of synchronization.
     */
    std::chrono::nanoseconds get_setup_time() const;

    /** The hierarchy BoomerAMG built, see get_hierarchy(). */
    struct hierarchy {
        /** Number of levels, counting the finest. */
        size_type num_levels;
        /** Global rows on each level, finest first. */
        std::vector<size_type> rows_per_level;
    };

    /**
     * The hierarchy of the most recent setup. Grid complexity is the sum of
     * rows_per_level divided by its first entry. Always populated on
     * supported hypre (2.32 and newer); empty only on a moved-from solver.
     * Device PMIS can draw a different hierarchy at every setup.
     */
    const std::optional<hierarchy>& get_hierarchy() const;

    /**
     * Parse parameters from a configuration property tree. Only keys
     * actually present in the configuration are applied; every parameter
     * absent from the configuration keeps this class's own default, which
     * for coarsen_type/interp_type/relax_type/relax_weight/num_sweeps/
     * max_levels/max_coarse_size/strong_threshold means "leave hypre's own
     * default alone", see the parameters_type documentation. amg_setup and
     * krylov_setup are std::functions and cannot be expressed in JSON, so
     * they are not configurable through this function.
     */
    static parameters_type parse(
        const config::pnode& config, const config::registry& context,
        const config::type_descriptor& td_for_child =
            config::make_type_descriptor<ValueType, LocalIndexType,
                                         GlobalIndexType>());

    /**
     * Returns a configuration_map for registering this type with a
     * config::registry, under the name "ext::hypre::solver::Pcg".
     */
    static config::configuration_map get_config_map();

    /** Creates a copy of the solver, sharing the hypre objects. */
    Pcg(const Pcg&);

    /** Moves from the given solver, leaving it empty. */
    Pcg(Pcg&&) noexcept;

    Pcg& operator=(const Pcg&);

    Pcg& operator=(Pcg&&) noexcept;

protected:
    Pcg(const Factory* factory, std::shared_ptr<const LinOp> system_matrix);

    void apply_impl(const LinOp* b, LinOp* x) const override;

    void apply_impl(const LinOp* alpha, const LinOp* b, const LinOp* beta,
                    LinOp* x) const override;

private:
    struct state;
    std::shared_ptr<state> state_;
};


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
struct Pcg<ValueType, LocalIndexType, GlobalIndexType>::state {
    detail::par_csr matrix;
    HYPRE_Solver amg = nullptr;
    HYPRE_Solver pcg = nullptr;
    // Whether the hypre objects live on the device; the phase boundaries
    // synchronize on this.
    bool device = false;
    size_type num_iterations = 0;
    double residual_norm = 0.0;
    bool converged = false;
    std::chrono::nanoseconds import_time{0};
    std::chrono::nanoseconds setup_time{0};
    std::optional<hierarchy> hierarchy_info;
    // The vectors HYPRE_ParCSRPCGSetup is handed; apply_impl re-points them
    // at each apply's arrays instead of building new ones, see set_values().
    // BoomerAMG's setup keeps the pointers it was given in its level-0 slots
    // until the first solve replaces them, so these arrays must outlive
    // generation; declared before the vectors placed over them, so the
    // vectors are destroyed first.
    array<double> setup_b_values;
    array<double> setup_x_values;
    std::unique_ptr<detail::placed_vector> setup_b;
    std::unique_ptr<detail::placed_vector> setup_x;

    ~state()
    {
        // Destruction never throws, and must not call into hypre or MPI once
        // either of them is finalized. The destroy functions accept null.
        int mpi_finalized = 0;
        MPI_Finalized(&mpi_finalized);
        if (mpi_finalized || !HYPRE_Initialized() || HYPRE_Finalized()) {
            return;
        }
        if (pcg) {
            HYPRE_ParCSRPCGDestroy(pcg);
        }
        if (amg) {
            HYPRE_BoomerAMGDestroy(amg);
        }
    }
};


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
Pcg<ValueType, LocalIndexType, GlobalIndexType>::Pcg(
    const Factory* factory, std::shared_ptr<const LinOp> system_matrix)
    : LinOp{factory->get_executor(), system_matrix->get_size()},
      parameters_{factory->get_parameters()},
      state_{std::make_shared<state>()}
{
    if (!HYPRE_Initialized()) {
        GKO_INVALID_STATE(
            "hypre is not initialized; create a gko::ext::hypre::environment "
            "before generating a gko::ext::hypre::solver::Pcg");
    }
    GKO_ASSERT_IS_SQUARE_MATRIX(system_matrix);

    const auto import_start = std::chrono::steady_clock::now();

    // Redundant with the identical call inside build_serial_par_csr /
    // build_distributed_par_csr / placed_vector, but kept here so a policy
    // conflict is reported uniformly, before any of that runs.
    detail::set_process_policy(
        detail::memory_location_of(this->get_executor()));

    if (dynamic_cast<const matrix::Csr<ValueType, LocalIndexType>*>(
            system_matrix.get())) {
        state_->matrix = detail::build_serial_par_csr<LocalIndexType>(
            std::move(system_matrix));
    } else if (auto distributed =
                   detail::build_any_distributed_par_csr<LocalIndexType>(
                       system_matrix)) {
        state_->matrix = std::move(*distributed);
    } else {
        GKO_NOT_SUPPORTED(*system_matrix);
    }
    state_->device = state_->matrix.memory_location == HYPRE_MEMORY_DEVICE;

    GKO_TPL_ASSERT_NO_HYPRE_ERRORS(HYPRE_BoomerAMGCreate(&state_->amg));
    const auto& params = this->get_parameters();
    if (!params.coarsen_type.empty()) {
        GKO_TPL_ASSERT_NO_HYPRE_ERRORS(HYPRE_BoomerAMGSetCoarsenType(
            state_->amg, detail::coarsen_type_id(params.coarsen_type)));
    }
    if (!params.interp_type.empty()) {
        GKO_TPL_ASSERT_NO_HYPRE_ERRORS(HYPRE_BoomerAMGSetInterpType(
            state_->amg, detail::interp_type_id(params.interp_type)));
    }
    if (!params.relax_type.empty()) {
        const auto relax_type = detail::relax_type_id(params.relax_type);
        // Rejected now rather than crashing later inside hypre's solve, see
        // relax_type_runs_on_device's documentation.
        if (state_->device && !detail::relax_type_runs_on_device(relax_type)) {
            GKO_INVALID_STATE(
                "relax_type '" + params.relax_type +
                "' has no hypre device implementation and would crash "
                "inside hypre's solve on a device executor; use relax_type "
                "\"l1-Jacobi\" instead");
        }
        GKO_TPL_ASSERT_NO_HYPRE_ERRORS(
            HYPRE_BoomerAMGSetRelaxType(state_->amg, relax_type));
    }
    if (params.relax_weight >= 0.0) {
        GKO_TPL_ASSERT_NO_HYPRE_ERRORS(
            HYPRE_BoomerAMGSetRelaxWt(state_->amg, params.relax_weight));
    }
    if (params.num_sweeps > 0) {
        GKO_TPL_ASSERT_NO_HYPRE_ERRORS(
            HYPRE_BoomerAMGSetNumSweeps(state_->amg, params.num_sweeps));
    }
    if (params.max_levels > 0) {
        GKO_TPL_ASSERT_NO_HYPRE_ERRORS(
            HYPRE_BoomerAMGSetMaxLevels(state_->amg, params.max_levels));
    }
    if (params.max_coarse_size > 0) {
        GKO_TPL_ASSERT_NO_HYPRE_ERRORS(HYPRE_BoomerAMGSetMaxCoarseSize(
            state_->amg, params.max_coarse_size));
    }
    if (params.strong_threshold >= 0.0) {
        GKO_TPL_ASSERT_NO_HYPRE_ERRORS(HYPRE_BoomerAMGSetStrongThreshold(
            state_->amg, params.strong_threshold));
    }
    GKO_TPL_ASSERT_NO_HYPRE_ERRORS(HYPRE_BoomerAMGSetPrintLevel(
        state_->amg, params.print_statistics ? 3 : 0));
    // As a preconditioner, BoomerAMG runs one cycle with no stopping
    // criterion of its own.
    GKO_TPL_ASSERT_NO_HYPRE_ERRORS(HYPRE_BoomerAMGSetTol(state_->amg, 0.0));
    GKO_TPL_ASSERT_NO_HYPRE_ERRORS(HYPRE_BoomerAMGSetMaxIter(state_->amg, 1));
    if (params.amg_setup) {
        params.amg_setup(state_->amg);
    }

    GKO_TPL_ASSERT_NO_HYPRE_ERRORS(
        HYPRE_ParCSRPCGCreate(state_->matrix.comm, &state_->pcg));
    GKO_TPL_ASSERT_NO_HYPRE_ERRORS(
        HYPRE_ParCSRPCGSetTol(state_->pcg, params.tolerance));
    GKO_TPL_ASSERT_NO_HYPRE_ERRORS(
        HYPRE_ParCSRPCGSetMaxIter(state_->pcg, params.max_iters));
    GKO_TPL_ASSERT_NO_HYPRE_ERRORS(
        HYPRE_ParCSRPCGSetTwoNorm(state_->pcg, params.two_norm ? 1 : 0));
    GKO_TPL_ASSERT_NO_HYPRE_ERRORS(HYPRE_ParCSRPCGSetPrecond(
        state_->pcg,
        reinterpret_cast<HYPRE_PtrToParSolverFcn>(HYPRE_BoomerAMGSolve),
        reinterpret_cast<HYPRE_PtrToParSolverFcn>(HYPRE_BoomerAMGSetup),
        state_->amg));
    if (params.krylov_setup) {
        params.krylov_setup(state_->pcg);
    }

    // Setup needs vectors of the right shape only; the real values are
    // placed per apply. A rank owning no rows still needs a non-null data
    // pointer, hence the minimum of one entry.
    const auto local_rows =
        static_cast<size_type>(state_->matrix.num_local_rows);
    const auto exec = this->get_executor();
    state_->setup_b_values =
        array<double>{exec, std::max<size_type>(1, local_rows)};
    state_->setup_x_values =
        array<double>{exec, std::max<size_type>(1, local_rows)};
    state_->setup_b_values.fill(0.0);
    state_->setup_x_values.fill(0.0);
    auto dummy_b = std::make_unique<detail::placed_vector>(
        state_->matrix.comm, state_->matrix.global_rows,
        state_->matrix.row_start, state_->matrix.num_local_rows,
        state_->setup_b_values.get_const_data(),
        state_->matrix.memory_location);
    auto dummy_x = std::make_unique<detail::placed_vector>(
        state_->matrix.comm, state_->matrix.global_rows,
        state_->matrix.row_start, state_->matrix.num_local_rows,
        state_->setup_x_values.get_const_data(),
        state_->matrix.memory_location);

    const auto setup_start =
        detail::end_of_phase(state_->matrix.comm, state_->device);
    GKO_TPL_ASSERT_NO_HYPRE_ERRORS(HYPRE_ParCSRPCGSetup(
        state_->pcg,
        reinterpret_cast<HYPRE_ParCSRMatrix>(state_->matrix.matrix),
        reinterpret_cast<HYPRE_ParVector>(dummy_b->get()),
        reinterpret_cast<HYPRE_ParVector>(dummy_x->get())));
    const auto setup_end =
        detail::end_of_phase(state_->matrix.comm, state_->device);
    state_->setup_b = std::move(dummy_b);
    state_->setup_x = std::move(dummy_x);
    state_->import_time = std::chrono::duration_cast<std::chrono::nanoseconds>(
        setup_start - import_start);
    state_->setup_time = std::chrono::duration_cast<std::chrono::nanoseconds>(
        setup_end - setup_start);

    // HYPRE_BoomerAMGGetGridHierarchy is public API since hypre 2.32.
#if HYPRE_RELEASE_NUMBER >= 23200
    {
        // last_level[i] is the last level row i reaches; level count is its
        // max plus one. Sized max(1, local_rows): hypre rejects a null
        // array, and an empty vector's data() may be null; only the first
        // local_rows entries are read. The reductions below are collective,
        // which is why this sits in the constructor rather than
        // get_hierarchy(); the hypre error is folded into the same
        // reduction so a rank-local throw cannot strand the others.
        std::vector<HYPRE_Int> last_level(std::max<size_type>(1, local_rows),
                                          0);
        const auto hierarchy_error = static_cast<int>(
            HYPRE_BoomerAMGGetGridHierarchy(state_->amg, last_level.data()));
        if (hierarchy_error != 0) {
            HYPRE_ClearAllErrors();
        }
        HYPRE_Int local_max = 0;
        for (size_type i = 0; i < local_rows; i++) {
            local_max = std::max(local_max, last_level[i]);
        }
        // [0] is any rank's hypre error, [1] the global number of levels.
        int reduced[2] = {hierarchy_error, static_cast<int>(local_max) + 1};
        GKO_ASSERT_NO_MPI_ERRORS(MPI_Allreduce(
            MPI_IN_PLACE, reduced, 2, MPI_INT, MPI_MAX, state_->matrix.comm));
        if (reduced[0] != 0) {
            GKO_INVALID_STATE(
                "hypre error code " + std::to_string(reduced[0]) + " (" +
                detail::describe_error(static_cast<HYPRE_Int>(reduced[0])) +
                ") returned by HYPRE_BoomerAMGGetGridHierarchy on at least "
                "one rank");
        }
        const int num_levels = reduced[1];
        std::vector<long long> rows(static_cast<size_type>(num_levels), 0);
        for (size_type i = 0; i < local_rows; i++) {
            for (int l = 0; l <= static_cast<int>(last_level[i]); l++) {
                rows[static_cast<size_type>(l)]++;
            }
        }
        GKO_ASSERT_NO_MPI_ERRORS(MPI_Allreduce(MPI_IN_PLACE, rows.data(),
                                               num_levels, MPI_LONG_LONG,
                                               MPI_SUM, state_->matrix.comm));
        hierarchy result{static_cast<size_type>(num_levels), {}};
        for (const auto count : rows) {
            result.rows_per_level.push_back(static_cast<size_type>(count));
        }
        state_->hierarchy_info = std::move(result);
    }
#endif
}


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
Pcg<ValueType, LocalIndexType, GlobalIndexType>::Pcg(const Pcg& other)
    : LinOp{other.get_executor()}
{
    *this = other;
}


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
Pcg<ValueType, LocalIndexType, GlobalIndexType>::Pcg(Pcg&& other) noexcept
    : LinOp{other.get_executor()}
{
    *this = std::move(other);
}


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
Pcg<ValueType, LocalIndexType, GlobalIndexType>&
Pcg<ValueType, LocalIndexType, GlobalIndexType>::operator=(const Pcg& other)
{
    if (this != &other) {
        LinOp::operator=(other);
        parameters_ = other.parameters_;
        state_ = other.state_;
    }
    return *this;
}


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
Pcg<ValueType, LocalIndexType, GlobalIndexType>&
Pcg<ValueType, LocalIndexType, GlobalIndexType>::operator=(Pcg&& other) noexcept
{
    if (this != &other) {
        LinOp::operator=(std::move(other));
        parameters_ = std::move(other.parameters_);
        state_ = std::move(other.state_);
    }
    return *this;
}


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
size_type Pcg<ValueType, LocalIndexType, GlobalIndexType>::get_num_iterations()
    const
{
    return state_ ? state_->num_iterations : 0;
}


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
double Pcg<ValueType, LocalIndexType, GlobalIndexType>::get_residual_norm()
    const
{
    return state_ ? state_->residual_norm : 0.0;
}


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
bool Pcg<ValueType, LocalIndexType, GlobalIndexType>::has_converged() const
{
    return state_ ? state_->converged : false;
}


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
std::chrono::nanoseconds
Pcg<ValueType, LocalIndexType, GlobalIndexType>::get_import_time() const
{
    return state_ ? state_->import_time : std::chrono::nanoseconds{0};
}


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
std::chrono::nanoseconds
Pcg<ValueType, LocalIndexType, GlobalIndexType>::get_setup_time() const
{
    return state_ ? state_->setup_time : std::chrono::nanoseconds{0};
}


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
const std::optional<
    typename Pcg<ValueType, LocalIndexType, GlobalIndexType>::hierarchy>&
Pcg<ValueType, LocalIndexType, GlobalIndexType>::get_hierarchy() const
{
    static const std::optional<hierarchy> none;
    return state_ ? state_->hierarchy_info : none;
}


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
void Pcg<ValueType, LocalIndexType, GlobalIndexType>::apply_impl(const LinOp* b,
                                                                 LinOp* x) const
{
    // The column count is the same on every rank, so rejecting it here
    // cannot make ranks diverge.
    if (b->get_size()[1] != 1) {
        throw DimensionMismatch(__FILE__, __LINE__, __func__, "b",
                                b->get_size()[0], b->get_size()[1], "x",
                                x->get_size()[0], x->get_size()[1],
                                "Pcg supports only a single right-hand side");
    }
    // hypre reads and writes b and x in place, wherever they live.
    auto solve = [this](const double* b_values, double* x_values) {
        // Re-points the vectors built at generation rather than constructing
        // new ones, which is not free on 2.33+ (HYPRE_IJVectorAssemble does an
        // unconditional Allreduce); needs no re-assembly, see
        // placed_vector::set_values.
        auto& hypre_b = *state_->setup_b;
        auto& hypre_x = *state_->setup_x;
        hypre_b.set_values(b_values);
        hypre_x.set_values(x_values);
        // HYPRE_ERROR_CONV is hypre's non-convergence flag, returned through
        // the same global bitmask as real failures, so it cannot go through
        // GKO_TPL_ASSERT_NO_HYPRE_ERRORS unfiltered: that would throw for
        // the non-convergence has_converged() documents as normal. The
        // convergence bit is cleared below; every other bit still throws.
        const HYPRE_Int solve_error = HYPRE_ParCSRPCGSolve(
            state_->pcg,
            reinterpret_cast<HYPRE_ParCSRMatrix>(state_->matrix.matrix),
            reinterpret_cast<HYPRE_ParVector>(hypre_b.get()),
            reinterpret_cast<HYPRE_ParVector>(hypre_x.get()));
        // The flag is global and sticky until cleared; clearing the
        // convergence bit here (HYPRE_ClearError touches only that bit)
        // keeps a non-converged solve from surfacing as a failure of a
        // later, unrelated hypre call.
        if (solve_error & HYPRE_ERROR_CONV) {
            HYPRE_ClearError(HYPRE_ERROR_CONV);
        }
        const HYPRE_Int fatal_error = solve_error & ~HYPRE_ERROR_CONV;
        if (fatal_error != 0) {
            const auto description = detail::describe_error(fatal_error);
            HYPRE_ClearAllErrors();
            GKO_INVALID_STATE(std::string("hypre error code ") +
                              std::to_string(static_cast<int>(fatal_error)) +
                              " (" + description +
                              ") returned by HYPRE_ParCSRPCGSolve");
        }
        HYPRE_Int iterations = 0;
        HYPRE_Real norm = 0.0;
        HYPRE_Int converged = 0;
        GKO_TPL_ASSERT_NO_HYPRE_ERRORS(
            HYPRE_ParCSRPCGGetNumIterations(state_->pcg, &iterations));
        GKO_TPL_ASSERT_NO_HYPRE_ERRORS(
            HYPRE_ParCSRPCGGetFinalRelativeResidualNorm(state_->pcg, &norm));
        GKO_TPL_ASSERT_NO_HYPRE_ERRORS(
            HYPRE_PCGGetConverged(state_->pcg, &converged));
        state_->num_iterations = static_cast<size_type>(iterations);
        state_->residual_norm = static_cast<double>(norm);
        state_->converged = converged != 0;
    };
    if (state_->matrix.distributed) {
        using dist_vec = experimental::distributed::Vector<ValueType>;
        auto dist_b = dynamic_cast<const dist_vec*>(b);
        auto dist_x = dynamic_cast<dist_vec*>(x);
        // Every rejection below is rank-local (a bad cast, stride, or size
        // can differ per rank), while placing the vectors and
        // HYPRE_ParCSRPCGSolve are collective. Folding rejections into one
        // bitmask and reducing before any throw keeps every rank reaching
        // the same collective, so a throw on one rank cannot strand the
        // others.
        constexpr int bad_cast_flag = 1;
        constexpr int stride_flag = 2;
        constexpr int size_flag = 4;
        int local_flags = (!dist_b || !dist_x) ? bad_cast_flag : 0;
        size_type local_b_rows = 0;
        size_type local_x_rows = 0;
        const auto owned_rows =
            static_cast<size_type>(state_->matrix.num_local_rows);
        // Only touch the vectors once the casts are known to have succeeded
        // on this rank.
        if (!(local_flags & bad_cast_flag)) {
            if (dist_b->get_local_vector()->get_stride() != 1 ||
                dist_x->get_local_vector()->get_stride() != 1) {
                local_flags |= stride_flag;
            }
            local_b_rows = dist_b->get_local_vector()->get_size()[0];
            local_x_rows = dist_x->get_local_vector()->get_size()[0];
            if (local_b_rows != owned_rows || local_x_rows != owned_rows) {
                local_flags |= size_flag;
            }
        }
        GKO_ASSERT_NO_MPI_ERRORS(MPI_Allreduce(MPI_IN_PLACE, &local_flags, 1,
                                               MPI_INT, MPI_BOR,
                                               state_->matrix.comm));
        if (local_flags & bad_cast_flag) {
            throw NotSupported(
                __FILE__, __LINE__, __func__,
                "b and x must both be "
                "experimental::distributed::Vector<ValueType> on every "
                "rank");
        }
        if (local_flags & stride_flag) {
            throw NotSupported(__FILE__, __LINE__, __func__,
                               "a distributed::Vector with a local stride "
                               "other than 1 on at least one rank");
        }
        if (local_flags & size_flag) {
            throw DimensionMismatch(
                __FILE__, __LINE__, __func__, "local b", local_b_rows, 1,
                "local x", local_x_rows, 1,
                "the local rows of b and x must match the " +
                    std::to_string(owned_rows) +
                    " rows this rank owns in the system matrix; this check "
                    "is collective, so the sizes shown may belong to a rank "
                    "whose local rows actually matched");
        }
        solve(dist_b->get_const_local_values(), dist_x->get_local_values());
        return;
    }
    auto dense_b = dynamic_cast<const matrix::Dense<ValueType>*>(b);
    auto dense_x = dynamic_cast<matrix::Dense<ValueType>*>(x);
    if (!dense_b) {
        GKO_NOT_SUPPORTED(*b);
    }
    if (!dense_x) {
        GKO_NOT_SUPPORTED(*x);
    }
    if (dense_b->get_stride() != 1 || dense_x->get_stride() != 1) {
        throw NotSupported(__FILE__, __LINE__, __func__,
                           "matrix::Dense with a stride other than 1");
    }
    solve(dense_b->get_const_values(), dense_x->get_values());
}


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
void Pcg<ValueType, LocalIndexType, GlobalIndexType>::apply_impl(
    const LinOp* alpha, const LinOp* b, const LinOp* beta, LinOp* x) const
{
    // x = alpha * solve(b) + beta * x, where the solve starts from x.
    auto x_solve = gko::clone(x);
    this->apply_impl(b, x_solve.get());
    if (auto dist_x =
            dynamic_cast<experimental::distributed::Vector<ValueType>*>(x)) {
        dist_x->scale(beta);
        dist_x->add_scaled(alpha, x_solve);
    } else if (auto dense_x = dynamic_cast<matrix::Dense<ValueType>*>(x)) {
        dense_x->scale(beta);
        dense_x->add_scaled(alpha, x_solve);
    } else {
        GKO_NOT_SUPPORTED(*x);
    }
}


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
typename Pcg<ValueType, LocalIndexType, GlobalIndexType>::parameters_type
Pcg<ValueType, LocalIndexType, GlobalIndexType>::parse(
    const config::pnode& config, const config::registry& context,
    const config::type_descriptor& td_for_child)
{
    auto params = Pcg::build();
    // config_check_decorator is only available in core, so we manually
    // check for unknown keys here.
    const std::set<std::string> allowed_keys = {"type",
                                                "value_type",
                                                "tolerance",
                                                "max_iters",
                                                "two_norm",
                                                "coarsen_type",
                                                "interp_type",
                                                "relax_type",
                                                "relax_weight",
                                                "num_sweeps",
                                                "max_levels",
                                                "max_coarse_size",
                                                "strong_threshold",
                                                "print_statistics"};
    if (config.get_tag() == config::pnode::tag_t::map) {
        for (const auto& [key, _] : config.get_map()) {
            GKO_THROW_IF_INVALID(allowed_keys.count(key),
                                 key + " is not an allowed key.");
        }
    }
    // Only a key present in the configuration calls its with_... setter;
    // materializing an absent key's sentinel would override hypre's own
    // default, which parameters_type promises not to do.
    if (const auto& obj = config.get("tolerance"); obj) {
        params.with_tolerance(obj.get_real());
    }
    if (const auto& obj = config.get("max_iters"); obj) {
        params.with_max_iters(static_cast<int>(obj.get_integer()));
    }
    if (const auto& obj = config.get("two_norm"); obj) {
        params.with_two_norm(obj.get_boolean());
    }
    if (const auto& obj = config.get("coarsen_type"); obj) {
        params.with_coarsen_type(obj.get_string());
    }
    if (const auto& obj = config.get("interp_type"); obj) {
        params.with_interp_type(obj.get_string());
    }
    if (const auto& obj = config.get("relax_type"); obj) {
        params.with_relax_type(obj.get_string());
    }
    if (const auto& obj = config.get("relax_weight"); obj) {
        params.with_relax_weight(obj.get_real());
    }
    if (const auto& obj = config.get("num_sweeps"); obj) {
        params.with_num_sweeps(static_cast<int>(obj.get_integer()));
    }
    if (const auto& obj = config.get("max_levels"); obj) {
        params.with_max_levels(static_cast<int>(obj.get_integer()));
    }
    if (const auto& obj = config.get("max_coarse_size"); obj) {
        params.with_max_coarse_size(static_cast<int>(obj.get_integer()));
    }
    if (const auto& obj = config.get("strong_threshold"); obj) {
        params.with_strong_threshold(obj.get_real());
    }
    if (const auto& obj = config.get("print_statistics"); obj) {
        params.with_print_statistics(obj.get_boolean());
    }
    return params;
}


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
config::configuration_map
Pcg<ValueType, LocalIndexType, GlobalIndexType>::get_config_map()
{
    return {{"ext::hypre::solver::Pcg",
             [](const config::pnode& config, const config::registry& context,
                config::type_descriptor td)
                 -> deferred_factory_parameter<LinOpFactory> {
                 return Pcg<ValueType, LocalIndexType, GlobalIndexType>::parse(
                     config, context, td);
             }}};
}


}  // namespace solver
}  // namespace hypre
}  // namespace ext
}  // namespace gko


#endif  // GKO_TPL_HYPRE_PCG_HPP_

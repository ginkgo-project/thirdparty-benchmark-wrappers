// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_TPL_PETSC_KSP_HPP_
#define GKO_TPL_PETSC_KSP_HPP_


#include <chrono>
#include <cstdint>
#include <cstring>
#include <memory>
#include <set>
#include <string>
#include <type_traits>
#include <utility>

#include <mpi.h>

#include <petscksp.h>

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

#include <gko_tpl/petsc/detail/error.hpp>
#include <gko_tpl/petsc/detail/import.hpp>
#include <gko_tpl/petsc/detail/options.hpp>
#include <gko_tpl/petsc/detail/timing.hpp>
#include <gko_tpl/petsc/detail/vectors.hpp>


namespace gko {
namespace ext {
namespace petsc {
namespace solver {


/**
 * A solver backed by a PETSc KSP, configured through a PETSc options string.
 *
 * With `-ksp_type cg -pc_type hypre` it is a complete PETSc CG solve
 * preconditioned by hypre BoomerAMG. With `-ksp_type preonly -pc_type hypre`
 * every apply is one preconditioner application, so the Ksp can serve as a
 * preconditioner inside Ginkgo solvers.
 *
 * Generation has two phases, which are timed separately for benchmarks:
 * - import: the system matrix is copied through the host into a PETSc AIJ
 *   matrix, and the KSP and its vectors are created and configured, see
 *   get_import_time();
 * - setup: KSPSetUp, e.g. the BoomerAMG setup including PETSc's conversion
 *   of the matrix into hypre's format, see get_setup_time().
 *
 * Supported system matrices:
 * - matrix::Csr<ValueType, IndexType>, solved on PETSC_COMM_SELF, applied to
 *   matrix::Dense vectors;
 * - experimental::distributed::Matrix<ValueType, IndexType, GlobalIndexType>
 *   read with a row partition whose parts are contiguous and ordered by rank,
 *   solved on the matrix's communicator, applied to
 *   experimental::distributed::Vector vectors.
 *
 * On a CudaExecutor with a CUDA-enabled PETSc build, the PETSc matrix is a
 * MATAIJCUSPARSE and b and x are passed to PETSc in place, on the device.
 * Otherwise the PETSc matrix is a host MATAIJ, and on a device executor b and
 * x are copied to and from the host.
 *
 * Only a single right-hand side is supported. For iterative KSP types the
 * input x is the initial guess; for `preonly` it is ignored. A solve that
 * does not converge does not throw, check has_converged().
 *
 * Options apply to this instance only, since every option name gets a unique
 * prefix. Global PETSc options such as `-log_view` therefore have no effect
 * here; pass them through PETSc's usual mechanisms, e.g. the PETSC_OPTIONS
 * environment variable.
 *
 * PETSc must be initialized before generation, e.g. with
 * gko::ext::petsc::environment, and every Ksp must be destroyed before PETSc
 * is finalized.
 *
 * @tparam ValueType  the value type, which must be PETSc's scalar type double
 * @tparam IndexType  the (local) index type of the system matrix
 */
template <typename ValueType = double, typename IndexType = int32>
class Ksp : public LinOp {
    GKO_ASSERT_SUPPORTED_INDEX_TYPE;
    static_assert(std::is_same<ValueType, double>::value,
                  "gko::ext::petsc::solver::Ksp only supports double, the "
                  "scalar type of the supported PETSc builds");

public:
    using value_type = ValueType;
    using index_type = IndexType;

    class Factory;

    struct parameters_type : enable_parameters_type<parameters_type, Factory> {
        /**
         * PETSc options for this solver, separated by whitespace, e.g.
         * "-ksp_type cg -ksp_rtol 1e-8 -pc_type hypre". Option values must
         * not contain spaces.
         */
        std::string GKO_FACTORY_PARAMETER_SCALAR(options, std::string{});
    };
    GKO_ENABLE_LIN_OP_FACTORY(Ksp, parameters, Factory);
    GKO_ENABLE_BUILD_METHOD(Factory);

    /**
     * Returns the number of iterations of the most recent apply, or 0 before
     * the first apply.
     */
    size_type get_num_iterations() const;

    /**
     * Returns the residual norm PETSc reported for the most recent apply (the
     * norm selected by `-ksp_norm_type`), or 0 before the first apply.
     */
    double get_residual_norm() const;

    /**
     * Returns PETSc's KSPConvergedReason of the most recent apply, or 0
     * before the first apply. Positive values mean converged, negative values
     * mean diverged.
     */
    int get_converged_reason() const;

    /** Returns true if the most recent apply converged. */
    bool has_converged() const;

    /**
     * Returns the duration of the import phase of generation, see the class
     * documentation. The phase starts when this rank entered generation and
     * ends with a synchronization of PETSc's device work, if any, and a
     * barrier on the solver's communicator, so it includes waiting for the
     * slowest rank. Work PETSc hands to other streams, e.g. hypre's, is not
     * waited for.
     */
    std::chrono::nanoseconds get_import_time() const;

    /**
     * Returns the duration of the setup phase of generation (KSPSetUp). It
     * starts where the import phase ends and ends with the same kind of
     * synchronization.
     */
    std::chrono::nanoseconds get_setup_time() const;

    /**
     * Parse parameters from a configuration property tree.
     */
    static parameters_type parse(
        const config::pnode& config, const config::registry& context,
        const config::type_descriptor& td_for_child =
            config::make_type_descriptor<ValueType, IndexType>());

    /**
     * Returns a configuration_map for registering this type with a
     * config::registry, under the name "ext::petsc::solver::Ksp".
     */
    static config::configuration_map get_config_map();

    /** Creates a copy of the solver, sharing the PETSc objects. */
    Ksp(const Ksp&);

    /** Moves from the given solver, leaving it empty. */
    Ksp(Ksp&&) noexcept;

    Ksp& operator=(const Ksp&);

    Ksp& operator=(Ksp&&) noexcept;

protected:
    Ksp(const Factory* factory, std::shared_ptr<const LinOp> system_matrix);

    void apply_impl(const LinOp* b, LinOp* x) const override;

    void apply_impl(const LinOp* alpha, const LinOp* b, const LinOp* beta,
                    LinOp* x) const override;

private:
    struct state;
    std::shared_ptr<state> state_;
};


template <typename ValueType, typename IndexType>
struct Ksp<ValueType, IndexType>::state {
    Mat mat = nullptr;
    KSP ksp = nullptr;
    Vec b = nullptr;
    Vec x = nullptr;
    MPI_Comm comm = MPI_COMM_NULL;
    bool distributed = false;
    // Whether mat/b/x are device types, so the solve places device pointers.
    bool device_vectors = false;
    PetscInt num_local_rows = 0;
    size_type num_iterations = 0;
    double residual_norm = 0.0;
    int converged_reason = 0;
    std::chrono::nanoseconds import_time{0};
    std::chrono::nanoseconds setup_time{0};

    ~state()
    {
        // Destruction never throws, and must not call into PETSc or MPI once
        // either of them is finalized. The destroy functions accept null.
        int mpi_finalized = 0;
        MPI_Finalized(&mpi_finalized);
        PetscBool petsc_initialized = PETSC_FALSE;
        PetscInitialized(&petsc_initialized);
        if (mpi_finalized || !petsc_initialized) {
            return;
        }
        VecDestroy(&x);
        VecDestroy(&b);
        KSPDestroy(&ksp);
        MatDestroy(&mat);
    }
};


template <typename ValueType, typename IndexType>
Ksp<ValueType, IndexType>::Ksp(const Factory* factory,
                               std::shared_ptr<const LinOp> system_matrix)
    : LinOp{factory->get_executor(), system_matrix->get_size()},
      parameters_{factory->get_parameters()},
      state_{std::make_shared<state>()}
{
    PetscBool petsc_initialized = PETSC_FALSE;
    GKO_TPL_ASSERT_NO_PETSC_ERRORS(PetscInitialized(&petsc_initialized));
    if (!petsc_initialized) {
        GKO_INVALID_STATE(
            "PETSc is not initialized; create a gko::ext::petsc::environment "
            "before generating a gko::ext::petsc::solver::Ksp");
    }
    GKO_ASSERT_IS_SQUARE_MATRIX(system_matrix);
    // The global size is identical on every rank, and every row and column
    // index of the system matrix is bounded by it, so validating it here,
    // before any collective call, guarantees the per-entry PetscInt range
    // checks in append_block (via to_petsc_int) can no longer throw on some
    // ranks but not others.
    detail::to_petsc_int(
        static_cast<std::int64_t>(system_matrix->get_size()[0]));

    const auto import_start = std::chrono::steady_clock::now();

    auto coo = [&]() -> detail::coo_data {
        if (dynamic_cast<const matrix::Csr<ValueType, IndexType>*>(
                system_matrix.get())) {
            return detail::build_serial_coo<IndexType>(system_matrix);
        }
        if (auto distributed_coo = detail::build_any_distributed_coo<IndexType>(
                system_matrix.get())) {
            return std::move(*distributed_coo);
        }
        GKO_NOT_SUPPORTED(*system_matrix);
    }();
    state_->distributed = coo.distributed;
    state_->num_local_rows = coo.num_local_rows;
    state_->comm = coo.comm;

    // PETSc copies the triplets, so the Ginkgo matrix need not outlive this
    // solver and the COO arrays are freed when generation ends.
    GKO_TPL_ASSERT_NO_PETSC_ERRORS(MatCreate(coo.comm, &state_->mat));
    GKO_TPL_ASSERT_NO_PETSC_ERRORS(
        MatSetSizes(state_->mat, coo.num_local_rows, coo.num_local_rows,
                    PETSC_DETERMINE, PETSC_DETERMINE));
    // PETSc derives hypre's option set from the matrix's memory type
    // (src/ksp/pc/impls/hypre/impl/ihypre.c): a device matrix gets the
    // GPU-supported defaults -- PMIS coarsening, ext+i interpolation,
    // l1-scaled Jacobi -- while a host matrix gets Falgout with classical
    // interpolation, neither of which has a device implementation. Handing
    // PETSc a host MATAIJ while Ginkgo runs on a GPU therefore benchmarks
    // host-side BoomerAMG, so follow the executor instead.
    auto mat_type = MATAIJ;
#if defined(PETSC_HAVE_CUDA)
    if (std::dynamic_pointer_cast<const CudaExecutor>(this->get_executor())) {
        mat_type = MATAIJCUSPARSE;
        // MatCreateVecs below then yields CUDA vectors.
        state_->device_vectors = true;
    }
#endif
    GKO_TPL_ASSERT_NO_PETSC_ERRORS(MatSetType(state_->mat, mat_type));
    GKO_TPL_ASSERT_NO_PETSC_ERRORS(MatSetPreallocationCOO(
        state_->mat, static_cast<PetscCount>(coo.rows.size()), coo.rows.data(),
        coo.cols.data()));
    GKO_TPL_ASSERT_NO_PETSC_ERRORS(
        MatSetValuesCOO(state_->mat, coo.values.data(), INSERT_VALUES));

    // Every instance reads its options under its own prefix, so that options
    // of different instances in PETSc's global database do not interfere.
    const auto prefix =
        "gko_ksp_" + std::to_string(detail::next_instance_id++) + "_";
    const auto prefixed_options =
        detail::prefix_options(this->get_parameters().options, prefix);
    GKO_TPL_ASSERT_NO_PETSC_ERRORS(KSPCreate(coo.comm, &state_->ksp));
    GKO_TPL_ASSERT_NO_PETSC_ERRORS(
        KSPSetOperators(state_->ksp, state_->mat, state_->mat));
    GKO_TPL_ASSERT_NO_PETSC_ERRORS(
        PetscOptionsInsertString(nullptr, prefixed_options.c_str()));
    GKO_TPL_ASSERT_NO_PETSC_ERRORS(
        KSPSetOptionsPrefix(state_->ksp, prefix.c_str()));
    GKO_TPL_ASSERT_NO_PETSC_ERRORS(KSPSetFromOptions(state_->ksp));
    // Like Ginkgo solvers, iterative KSPs start from the input x. PETSc
    // rejects a nonzero initial guess for preonly.
    KSPType ksp_type = nullptr;
    GKO_TPL_ASSERT_NO_PETSC_ERRORS(KSPGetType(state_->ksp, &ksp_type));
    if (std::strcmp(ksp_type, KSPPREONLY) != 0) {
        GKO_TPL_ASSERT_NO_PETSC_ERRORS(
            KSPSetInitialGuessNonzero(state_->ksp, PETSC_TRUE));
    }
    GKO_TPL_ASSERT_NO_PETSC_ERRORS(
        MatCreateVecs(state_->mat, &state_->x, &state_->b));

    const auto setup_start =
        detail::end_of_phase(coo.comm, state_->device_vectors);
    GKO_TPL_ASSERT_NO_PETSC_ERRORS(KSPSetUp(state_->ksp));
    const auto setup_end =
        detail::end_of_phase(coo.comm, state_->device_vectors);
    state_->import_time = std::chrono::duration_cast<std::chrono::nanoseconds>(
        setup_start - import_start);
    state_->setup_time = std::chrono::duration_cast<std::chrono::nanoseconds>(
        setup_end - setup_start);
}


template <typename ValueType, typename IndexType>
Ksp<ValueType, IndexType>::Ksp(const Ksp& other) : LinOp{other.get_executor()}
{
    *this = other;
}


template <typename ValueType, typename IndexType>
Ksp<ValueType, IndexType>::Ksp(Ksp&& other) noexcept
    : LinOp{other.get_executor()}
{
    *this = std::move(other);
}


template <typename ValueType, typename IndexType>
Ksp<ValueType, IndexType>& Ksp<ValueType, IndexType>::operator=(
    const Ksp& other)
{
    if (this != &other) {
        LinOp::operator=(other);
        parameters_ = other.parameters_;
        state_ = other.state_;
    }
    return *this;
}


template <typename ValueType, typename IndexType>
Ksp<ValueType, IndexType>& Ksp<ValueType, IndexType>::operator=(
    Ksp&& other) noexcept
{
    if (this != &other) {
        LinOp::operator=(std::move(other));
        parameters_ = std::move(other.parameters_);
        state_ = std::move(other.state_);
    }
    return *this;
}


template <typename ValueType, typename IndexType>
size_type Ksp<ValueType, IndexType>::get_num_iterations() const
{
    return state_ ? state_->num_iterations : 0;
}


template <typename ValueType, typename IndexType>
double Ksp<ValueType, IndexType>::get_residual_norm() const
{
    return state_ ? state_->residual_norm : 0.0;
}


template <typename ValueType, typename IndexType>
int Ksp<ValueType, IndexType>::get_converged_reason() const
{
    return state_ ? state_->converged_reason : 0;
}


template <typename ValueType, typename IndexType>
bool Ksp<ValueType, IndexType>::has_converged() const
{
    return this->get_converged_reason() > 0;
}


template <typename ValueType, typename IndexType>
std::chrono::nanoseconds Ksp<ValueType, IndexType>::get_import_time() const
{
    return state_ ? state_->import_time : std::chrono::nanoseconds{0};
}


template <typename ValueType, typename IndexType>
std::chrono::nanoseconds Ksp<ValueType, IndexType>::get_setup_time() const
{
    return state_ ? state_->setup_time : std::chrono::nanoseconds{0};
}


template <typename ValueType, typename IndexType>
void Ksp<ValueType, IndexType>::apply_impl(const LinOp* b, LinOp* x) const
{
    if (b->get_size()[1] != 1) {
        throw DimensionMismatch(__FILE__, __LINE__, __func__, "b",
                                b->get_size()[0], b->get_size()[1], "x",
                                x->get_size()[0], x->get_size()[1],
                                "Ksp supports only a single right-hand side");
    }
    // Solves with arrays holding this rank's rows of b and x, in whichever
    // memory space state_->b and state_->x were created in.
    auto solve = [this](const double* b_values, double* x_values) {
        const bool dev = state_->device_vectors;
        detail::placed_array placed_b{state_->b, b_values, dev};
        detail::placed_array placed_x{state_->x, x_values, dev};
        GKO_TPL_ASSERT_NO_PETSC_ERRORS(
            KSPSolve(state_->ksp, state_->b, state_->x));
        PetscInt num_iterations = 0;
        PetscReal residual_norm = 0;
        KSPConvergedReason reason = KSP_CONVERGED_ITERATING;
        GKO_TPL_ASSERT_NO_PETSC_ERRORS(
            KSPGetIterationNumber(state_->ksp, &num_iterations));
        GKO_TPL_ASSERT_NO_PETSC_ERRORS(
            KSPGetResidualNorm(state_->ksp, &residual_norm));
        GKO_TPL_ASSERT_NO_PETSC_ERRORS(
            KSPGetConvergedReason(state_->ksp, &reason));
        state_->num_iterations = static_cast<size_type>(num_iterations);
        state_->residual_norm = static_cast<double>(residual_norm);
        state_->converged_reason = static_cast<int>(reason);
    };
    const auto exec = this->get_executor();
    if (state_->distributed) {
        using dist_vec = experimental::distributed::Vector<ValueType>;
        auto dist_b = dynamic_cast<const dist_vec*>(b);
        auto dist_x = dynamic_cast<dist_vec*>(x);
        // Every rejection below depends on this rank's own operands (a bad
        // cast, a local stride, or a local size can each be wrong on only
        // some ranks), while KSPSolve at the end of this branch is
        // collective. Folding every rejection into one bitmask and reducing
        // it before any throw guarantees every rank that enters this branch
        // reaches the same collective, and that if any rank must throw, all
        // ranks throw the same exception together, instead of leaving some
        // ranks blocked in KSPSolve (or, if the throws stayed rank-local,
        // in this very reduction).
        constexpr int bad_cast_flag = 1;
        constexpr int stride_flag = 2;
        constexpr int size_flag = 4;
        int local_flags = (!dist_b || !dist_x) ? bad_cast_flag : 0;
        size_type local_b_rows = 0;
        size_type local_x_rows = 0;
        const auto owned_rows = static_cast<size_type>(state_->num_local_rows);
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
                                               MPI_INT, MPI_BOR, state_->comm));
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
        detail::solve_with_arrays(exec, state_->device_vectors, owned_rows,
                                  dist_b->get_const_local_values(),
                                  dist_x->get_local_values(), solve);
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
    detail::solve_with_arrays(
        exec, state_->device_vectors, dense_x->get_size()[0],
        dense_b->get_const_values(), dense_x->get_values(), solve);
}


template <typename ValueType, typename IndexType>
void Ksp<ValueType, IndexType>::apply_impl(const LinOp* alpha, const LinOp* b,
                                           const LinOp* beta, LinOp* x) const
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


template <typename ValueType, typename IndexType>
typename Ksp<ValueType, IndexType>::parameters_type
Ksp<ValueType, IndexType>::parse(const config::pnode& config,
                                 const config::registry& context,
                                 const config::type_descriptor& td_for_child)
{
    auto params = Ksp::build();
    // config_check_decorator is only available in core, so we manually
    // check for unknown keys here.
    const std::set<std::string> allowed_keys = {"type", "value_type",
                                                "options"};
    if (config.get_tag() == config::pnode::tag_t::map) {
        for (const auto& [key, _] : config.get_map()) {
            GKO_THROW_IF_INVALID(allowed_keys.count(key),
                                 key + " is not an allowed key.");
        }
    }
    if (const auto& obj = config.get("options"); obj) {
        params.with_options(obj.get_string());
    }
    return params;
}


template <typename ValueType, typename IndexType>
config::configuration_map Ksp<ValueType, IndexType>::get_config_map()
{
    return {{"ext::petsc::solver::Ksp",
             [](const config::pnode& config, const config::registry& context,
                config::type_descriptor td)
                 -> deferred_factory_parameter<LinOpFactory> {
                 return Ksp<ValueType, IndexType>::parse(config, context, td);
             }}};
}


}  // namespace solver
}  // namespace petsc
}  // namespace ext
}  // namespace gko


#endif  // GKO_TPL_PETSC_KSP_HPP_

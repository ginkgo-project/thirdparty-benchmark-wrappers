// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_TPL_HYPRE_DETAIL_IMPORT_HPP_
#define GKO_TPL_HYPRE_DETAIL_IMPORT_HPP_


#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>

#include <mpi.h>

#include <_hypre_parcsr_mv.h>
#include <HYPRE_utilities.h>

#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/distributed/index_map.hpp>
#include <ginkgo/core/distributed/matrix.hpp>
#include <ginkgo/core/distributed/partition.hpp>
#include <ginkgo/core/matrix/csr.hpp>

#include <gko_tpl/hypre/detail/error.hpp>


namespace gko {
namespace ext {
namespace hypre {
namespace detail {


// The supported hypre builds use real double scalars; Ginkgo's value arrays
// are handed to hypre without conversion.
static_assert(std::is_same<HYPRE_Real, double>::value,
              "gko::ext::hypre requires a hypre build with real double "
              "scalars");


inline HYPRE_BigInt to_hypre_big_int(std::int64_t value)
{
    if (value < std::numeric_limits<HYPRE_BigInt>::min() ||
        value > std::numeric_limits<HYPRE_BigInt>::max()) {
        GKO_INVALID_STATE("index " + std::to_string(value) +
                          " does not fit into HYPRE_BigInt; a hypre build "
                          "configured with --enable-bigint is required");
    }
    return static_cast<HYPRE_BigInt>(value);
}


inline HYPRE_MemoryLocation memory_location_of(
    std::shared_ptr<const Executor> exec)
{
    return exec == exec->get_master() ? HYPRE_MEMORY_HOST : HYPRE_MEMORY_DEVICE;
}


// hypre's view of a Ginkgo index array: the array itself when the index types
// match, otherwise a copy converted through the host into `storage`.
template <typename IndexType>
const HYPRE_Int* as_hypre_int(std::shared_ptr<const Executor> exec,
                              const IndexType* data, size_type size,
                              array<HYPRE_Int>& storage)
{
    if constexpr (std::is_same<IndexType, HYPRE_Int>::value) {
        return data;
    } else {
        const auto host = exec->get_master();
        array<IndexType> host_source{host, size};
        host->copy_from(exec, size, data, host_source.get_data());
        array<HYPRE_Int> host_converted{host, size};
        for (size_type i = 0; i < size; i++) {
            host_converted.get_data()[i] =
                static_cast<HYPRE_Int>(host_source.get_const_data()[i]);
        }
        storage = array<HYPRE_Int>{exec, std::move(host_converted)};
        return storage.get_const_data();
    }
}


// A hypre matrix over Ginkgo's memory, plus what must outlive it: the Ginkgo
// matrix its arrays point at, and any array converted for a differing index
// type.
struct par_csr {
    hypre_ParCSRMatrix* matrix = nullptr;
    MPI_Comm comm = MPI_COMM_NULL;
    bool distributed = false;
    HYPRE_Int num_local_rows = 0;
    HYPRE_BigInt row_start = 0;
    HYPRE_BigInt global_rows = 0;
    HYPRE_MemoryLocation memory_location = HYPRE_MEMORY_HOST;
    array<HYPRE_Int> diag_row_ptrs;
    array<HYPRE_Int> diag_col_idxs;
    array<HYPRE_Int> offd_row_ptrs;
    array<HYPRE_Int> offd_col_idxs;
    std::shared_ptr<const LinOp> system_matrix;

    par_csr() = default;
    par_csr(const par_csr&) = delete;
    par_csr& operator=(const par_csr&) = delete;
    par_csr(par_csr&& other) noexcept { *this = std::move(other); }

    par_csr& operator=(par_csr&& other) noexcept
    {
        if (this != &other) {
            // Every member is swapped rather than plain-assigned, so each
            // object keeps one coherent set: a matrix with the `distributed`
            // flag and borrowed arrays ~par_csr reads to decide what to
            // detach before freeing. Assigning `distributed` alone would let
            // it destroy a serial matrix as if distributed, or vice versa.
            std::swap(matrix, other.matrix);
            std::swap(comm, other.comm);
            std::swap(distributed, other.distributed);
            std::swap(num_local_rows, other.num_local_rows);
            std::swap(row_start, other.row_start);
            std::swap(global_rows, other.global_rows);
            std::swap(memory_location, other.memory_location);
            std::swap(diag_row_ptrs, other.diag_row_ptrs);
            std::swap(diag_col_idxs, other.diag_col_idxs);
            std::swap(offd_row_ptrs, other.offd_row_ptrs);
            std::swap(offd_col_idxs, other.offd_col_idxs);
            std::swap(system_matrix, other.system_matrix);
        }
        return *this;
    }

    ~par_csr()
    {
        // Destruction must not call into hypre once it is finalized, and
        // hypre_ParCSRMatrixDestroy accepts null.
        if (matrix && HYPRE_Initialized() && !HYPRE_Finalized()) {
            // hypre_CSRMatrixDestroy frees a block's row-pointer array
            // unconditionally, regardless of the data-owner flag (unlike its
            // column-index and value arrays), so `I` must be nulled by hand
            // to keep it from freeing Ginkgo's memory.
            auto* diag = hypre_ParCSRMatrixDiag(matrix);
            if (diag) {
                hypre_CSRMatrixI(diag) = nullptr;
                hypre_CSRMatrixJ(diag) = nullptr;
                hypre_CSRMatrixData(diag) = nullptr;
            }
            // Distributed: offd's I/J/Data are borrowed the same way and
            // need the same treatment. Serial: offd is hypre's own empty
            // block and is left alone so hypre frees it normally.
            // col_map_offd is never nulled: it is allocated with hypre's own
            // allocator and handed to the matrix, so it, like Rownnz, is
            // hypre's to free.
            if (distributed) {
                auto* offd = hypre_ParCSRMatrixOffd(matrix);
                if (offd) {
                    hypre_CSRMatrixI(offd) = nullptr;
                    hypre_CSRMatrixJ(offd) = nullptr;
                    hypre_CSRMatrixData(offd) = nullptr;
                }
            }
            hypre_ParCSRMatrixDestroy(matrix);
        }
    }
};


// Points a hypre CSR block at Ginkgo's arrays, clearing hypre's ownership so
// that destroying the matrix leaves Ginkgo's memory alone.
inline void set_block(hypre_CSRMatrix* block, const HYPRE_Int* row_ptrs,
                      const HYPRE_Int* col_idxs, const double* values,
                      HYPRE_Int num_nonzeros)
{
    hypre_CSRMatrixI(block) = const_cast<HYPRE_Int*>(row_ptrs);
    hypre_CSRMatrixJ(block) = const_cast<HYPRE_Int*>(col_idxs);
    hypre_CSRMatrixData(block) =
        const_cast<HYPRE_Complex*>(static_cast<const HYPRE_Complex*>(values));
    hypre_CSRMatrixNumNonzeros(block) = num_nonzeros;
    hypre_CSRMatrixSetDataOwner(block, 0);
}


// A serial Csr as a one-rank hypre matrix on MPI_COMM_SELF, with no
// off-diagonal block.
template <typename IndexType>
par_csr build_serial_par_csr(std::shared_ptr<const LinOp> system_matrix)
{
    using csr = matrix::Csr<double, IndexType>;
    const auto mtx = std::dynamic_pointer_cast<const csr>(system_matrix);
    if (!mtx) {
        GKO_NOT_SUPPORTED(*system_matrix);
    }
    const auto exec = mtx->get_executor();
    const auto num_rows = static_cast<std::int64_t>(mtx->get_size()[0]);
    const auto nnz = static_cast<std::int64_t>(mtx->get_num_stored_elements());

    par_csr result;
    result.comm = MPI_COMM_SELF;
    result.distributed = false;
    result.num_local_rows = static_cast<HYPRE_Int>(num_rows);
    result.row_start = 0;
    result.global_rows = to_hypre_big_int(num_rows);
    result.memory_location = memory_location_of(exec);
    result.system_matrix = system_matrix;

    HYPRE_BigInt starts[2] = {0, result.global_rows};
    result.matrix = hypre_ParCSRMatrixCreate(result.comm, result.global_rows,
                                             result.global_rows, starts, starts,
                                             0, static_cast<HYPRE_Int>(nnz), 0);
    const auto row_ptrs = as_hypre_int(exec, mtx->get_const_row_ptrs(),
                                       static_cast<size_type>(num_rows) + 1,
                                       result.diag_row_ptrs);
    const auto col_idxs =
        as_hypre_int(exec, mtx->get_const_col_idxs(),
                     static_cast<size_type>(nnz), result.diag_col_idxs);
    // BoomerAMG converges identically whether the diagonal block's columns
    // are in Ginkgo's ascending order or reordered diagonal-first, so the
    // block below is aliased directly rather than copied and reordered.
    set_block(hypre_ParCSRMatrixDiag(result.matrix), row_ptrs, col_idxs,
              mtx->get_const_values(), static_cast<HYPRE_Int>(nnz));
    GKO_TPL_ASSERT_NO_HYPRE_ERRORS(
        hypre_ParCSRMatrixInitialize_v2(result.matrix, result.memory_location));
    hypre_CSRMatrixSetRownnz(hypre_ParCSRMatrixDiag(result.matrix));
    GKO_TPL_ASSERT_NO_HYPRE_ERRORS(
        hypre_ParCSRMatrixSetNumNonzeros(result.matrix));
    if (!hypre_ParCSRMatrixCommPkg(result.matrix)) {
        GKO_TPL_ASSERT_NO_HYPRE_ERRORS(
            hypre_MatvecCommPkgCreate(result.matrix));
    }
    return result;
}


// Ginkgo's distributed matrix and hypre's ParCSR agree on the format: the
// local block is hypre's diag, the non-local block is its offd with columns
// numbered by the index map's remote indices, and col_map_offd is those
// indices' global numbers. The index map orders them by owning rank and
// global index, ascending overall for a partition ordered by rank, as hypre
// requires. col_map_offd is a host array even for a device matrix.
template <typename LocalIndexType, typename GlobalIndexType>
par_csr build_distributed_par_csr(
    const experimental::distributed::Matrix<double, LocalIndexType,
                                            GlobalIndexType>* system_matrix,
    std::shared_ptr<const LinOp> owner)
{
    using csr = matrix::Csr<double, LocalIndexType>;
    // The partition is the same on every rank, so throwing here, before any
    // collective call, cannot leave other ranks waiting.
    const auto partition = system_matrix->get_row_partition();
    if (!partition || !partition->has_ordered_parts()) {
        throw NotSupported(__FILE__, __LINE__, __func__,
                           "distributed::Matrix without a row partition of "
                           "contiguous parts ordered by rank");
    }
    const auto diag =
        std::dynamic_pointer_cast<const csr>(system_matrix->get_diag_matrix());
    const auto offd = std::dynamic_pointer_cast<const csr>(
        system_matrix->get_off_diag_matrix());
    if (!diag || !offd) {
        GKO_NOT_SUPPORTED(*system_matrix);
    }
    const auto exec = system_matrix->get_executor();
    const auto comm = system_matrix->get_communicator();
    const auto& imap = system_matrix->get_index_map();

    par_csr result;
    result.comm = comm.get();
    result.distributed = true;
    result.num_local_rows = static_cast<HYPRE_Int>(diag->get_size()[0]);
    result.global_rows = to_hypre_big_int(
        static_cast<std::int64_t>(system_matrix->get_size()[0]));
    result.memory_location = memory_location_of(exec);
    result.system_matrix = std::move(owner);

    // This rank's first global row is the number of rows the ranks before it
    // own, an exclusive scan of the local row counts.
    const auto num_local_rows = static_cast<std::int64_t>(diag->get_size()[0]);
    std::int64_t row_end = 0;
    GKO_ASSERT_NO_MPI_ERRORS(MPI_Scan(&num_local_rows, &row_end, 1, MPI_INT64_T,
                                      MPI_SUM, comm.get()));
    result.row_start = to_hypre_big_int(row_end - num_local_rows);
    HYPRE_BigInt starts[2] = {result.row_start, to_hypre_big_int(row_end)};

    const auto num_cols_offd =
        static_cast<HYPRE_Int>(imap.get_non_local_size());
    const auto diag_nnz =
        static_cast<HYPRE_Int>(diag->get_num_stored_elements());
    const auto offd_nnz =
        static_cast<HYPRE_Int>(offd->get_num_stored_elements());
    result.matrix = hypre_ParCSRMatrixCreate(result.comm, result.global_rows,
                                             result.global_rows, starts, starts,
                                             num_cols_offd, diag_nnz, offd_nnz);

    set_block(
        hypre_ParCSRMatrixDiag(result.matrix),
        as_hypre_int(exec, diag->get_const_row_ptrs(),
                     static_cast<size_type>(num_local_rows) + 1,
                     result.diag_row_ptrs),
        as_hypre_int(exec, diag->get_const_col_idxs(),
                     static_cast<size_type>(diag_nnz), result.diag_col_idxs),
        diag->get_const_values(), diag_nnz);
    set_block(
        hypre_ParCSRMatrixOffd(result.matrix),
        as_hypre_int(exec, offd->get_const_row_ptrs(),
                     static_cast<size_type>(num_local_rows) + 1,
                     result.offd_row_ptrs),
        as_hypre_int(exec, offd->get_const_col_idxs(),
                     static_cast<size_type>(offd_nnz), result.offd_col_idxs),
        offd->get_const_values(), offd_nnz);
    GKO_TPL_ASSERT_NO_HYPRE_ERRORS(
        hypre_ParCSRMatrixInitialize_v2(result.matrix, result.memory_location));
    hypre_CSRMatrixSetRownnz(hypre_ParCSRMatrixDiag(result.matrix));
    hypre_CSRMatrixSetRownnz(hypre_ParCSRMatrixOffd(result.matrix));

    // The off-diagonal columns' global numbers, in the non-local block's
    // order; they live on the executor and hypre wants them on the host.
    const auto& remote = imap.get_remote_global_idxs();
    const auto host = exec->get_master();
    array<GlobalIndexType> host_remote{host,
                                       static_cast<size_type>(num_cols_offd)};
    host->copy_from(remote.get_executor(),
                    static_cast<size_type>(num_cols_offd),
                    remote.get_const_flat_data(), host_remote.get_data());
    auto* col_map =
        hypre_CTAlloc(HYPRE_BigInt, num_cols_offd, HYPRE_MEMORY_HOST);
    for (HYPRE_Int i = 0; i < num_cols_offd; i++) {
        col_map[i] = to_hypre_big_int(
            static_cast<std::int64_t>(host_remote.get_const_data()[i]));
    }
    hypre_TFree(hypre_ParCSRMatrixColMapOffd(result.matrix), HYPRE_MEMORY_HOST);
    hypre_ParCSRMatrixColMapOffd(result.matrix) = col_map;

    GKO_TPL_ASSERT_NO_HYPRE_ERRORS(
        hypre_ParCSRMatrixSetNumNonzeros(result.matrix));
    if (!hypre_ParCSRMatrixCommPkg(result.matrix)) {
        GKO_TPL_ASSERT_NO_HYPRE_ERRORS(
            hypre_MatvecCommPkgCreate(result.matrix));
    }
    return result;
}


// The hypre matrix of a distributed system matrix with local index type
// IndexType, or nothing for any other operator. Distributed matrices are only
// instantiated with a global index type at least as wide as the local one, so
// a 32-bit global index type is only tried for 32-bit local indices.
template <typename IndexType>
std::optional<par_csr> build_any_distributed_par_csr(
    std::shared_ptr<const LinOp> system_matrix)
{
    using experimental::distributed::Matrix;
    if (auto matrix =
            std::dynamic_pointer_cast<const Matrix<double, IndexType, int64>>(
                system_matrix)) {
        return build_distributed_par_csr(matrix.get(), system_matrix);
    }
    if constexpr (std::is_same<IndexType, int32>::value) {
        if (auto matrix =
                std::dynamic_pointer_cast<const Matrix<double, int32, int32>>(
                    system_matrix)) {
            return build_distributed_par_csr(matrix.get(), system_matrix);
        }
    }
    return std::nullopt;
}


}  // namespace detail
}  // namespace hypre
}  // namespace ext
}  // namespace gko


#endif  // GKO_TPL_HYPRE_DETAIL_IMPORT_HPP_

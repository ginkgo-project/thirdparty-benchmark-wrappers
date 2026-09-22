// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_TPL_PETSC_DETAIL_IMPORT_HPP_
#define GKO_TPL_PETSC_DETAIL_IMPORT_HPP_


#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <vector>

#include <mpi.h>

#include <petscsys.h>

#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/distributed/index_map.hpp>
#include <ginkgo/core/distributed/matrix.hpp>
#include <ginkgo/core/distributed/partition.hpp>
#include <ginkgo/core/matrix/csr.hpp>


namespace gko {
namespace ext {
namespace petsc {
namespace detail {


// The supported PETSc builds use real double scalars; Ginkgo's value arrays
// are placed into PETSc vectors without conversion.
static_assert(std::is_same<PetscScalar, double>::value,
              "gko::ext::petsc requires a PETSc build with real double "
              "scalars");


inline PetscInt to_petsc_int(std::int64_t value)
{
    if (value < std::numeric_limits<PetscInt>::min() ||
        value > std::numeric_limits<PetscInt>::max()) {
        GKO_INVALID_STATE("index " + std::to_string(value) +
                          " does not fit into PetscInt; a PETSc build "
                          "configured with --with-64-bit-indices is required");
    }
    return static_cast<PetscInt>(value);
}


// A matrix as global (row, column, value) triplets, with the communicator it
// lives on and the number of rows this rank owns.
struct coo_data {
    MPI_Comm comm;
    bool distributed;
    PetscInt num_local_rows;
    std::vector<PetscInt> rows;
    std::vector<PetscInt> cols;
    std::vector<PetscScalar> values;
};


// Appends the entries of a host Csr block: row i of the block is global row
// row_offset + i, and entry k is in global column global_cols[k].
template <typename IndexType, typename ColIndexType>
void append_block(const matrix::Csr<double, IndexType>* block,
                  std::int64_t row_offset, const ColIndexType* global_cols,
                  coo_data& coo)
{
    const auto row_ptrs = block->get_const_row_ptrs();
    const auto values = block->get_const_values();
    const auto num_rows = static_cast<std::int64_t>(block->get_size()[0]);
    for (std::int64_t row = 0; row < num_rows; row++) {
        for (auto k = row_ptrs[row]; k < row_ptrs[row + 1]; k++) {
            coo.rows.push_back(to_petsc_int(row_offset + row));
            coo.cols.push_back(
                to_petsc_int(static_cast<std::int64_t>(global_cols[k])));
            coo.values.push_back(values[k]);
        }
    }
}


template <typename IndexType>
coo_data build_serial_coo(std::shared_ptr<const LinOp> system_matrix)
{
    using csr = matrix::Csr<double, IndexType>;
    const auto host = system_matrix->get_executor()->get_master();
    const auto host_csr = copy_and_convert_to<csr>(host, system_matrix);
    coo_data coo{
        PETSC_COMM_SELF,
        false,
        to_petsc_int(static_cast<std::int64_t>(host_csr->get_size()[0])),
        {},
        {},
        {}};
    const auto nnz = host_csr->get_num_stored_elements();
    coo.rows.reserve(nnz);
    coo.cols.reserve(nnz);
    coo.values.reserve(nnz);
    append_block(host_csr.get(), 0, host_csr->get_const_col_idxs(), coo);
    return coo;
}


template <typename LocalIndexType, typename GlobalIndexType>
coo_data build_distributed_coo(
    const experimental::distributed::Matrix<double, LocalIndexType,
                                            GlobalIndexType>* system_matrix)
{
    using csr = matrix::Csr<double, LocalIndexType>;
    using experimental::distributed::index_space;
    // PETSc's parallel AIJ matrices give each rank one contiguous block of
    // global rows, with the blocks ordered by rank. The partition, and so
    // this outcome, is the same on all ranks, so throwing here before any
    // collective call cannot leave other ranks waiting.
    const auto partition = system_matrix->get_row_partition();
    if (!partition || !partition->has_ordered_parts()) {
        throw NotSupported(__FILE__, __LINE__, __func__,
                           "distributed::Matrix without a row partition of "
                           "contiguous parts ordered by rank");
    }
    const auto comm = system_matrix->get_communicator();
    const auto host = system_matrix->get_executor()->get_master();
    const auto diag =
        copy_and_convert_to<csr>(host, system_matrix->get_diag_matrix());
    const auto off_diag =
        copy_and_convert_to<csr>(host, system_matrix->get_off_diag_matrix());
    // This rank's first global row is the number of rows owned by the ranks
    // before it.
    const auto num_local_rows = static_cast<std::int64_t>(diag->get_size()[0]);
    std::int64_t row_end = 0;
    GKO_ASSERT_NO_MPI_ERRORS(MPI_Scan(&num_local_rows, &row_end, 1, MPI_INT64_T,
                                      MPI_SUM, comm.get()));
    const auto row_start = row_end - num_local_rows;
    // Maps a block's local column indices to global ones, on the host.
    const auto& imap = system_matrix->get_index_map();
    auto to_global = [&](const csr* block, index_space space) {
        const auto cols = block->get_const_col_idxs();
        const array<LocalIndexType> host_cols(
            host, cols, cols + block->get_num_stored_elements());
        auto global_cols = imap.map_to_global(
            array<LocalIndexType>(imap.get_executor(), host_cols), space);
        global_cols.set_executor(host);
        return global_cols;
    };
    const auto diag_cols = to_global(diag.get(), index_space::local);
    const auto off_diag_cols =
        to_global(off_diag.get(), index_space::non_local);

    coo_data coo{comm.get(), true, to_petsc_int(num_local_rows), {}, {}, {}};
    const auto nnz =
        diag->get_num_stored_elements() + off_diag->get_num_stored_elements();
    coo.rows.reserve(nnz);
    coo.cols.reserve(nnz);
    coo.values.reserve(nnz);
    append_block(diag.get(), row_start, diag_cols.get_const_data(), coo);
    append_block(off_diag.get(), row_start, off_diag_cols.get_const_data(),
                 coo);
    return coo;
}


// Returns the COO data of a distributed system matrix with local index type
// IndexType, or nothing for any other operator. Distributed matrices are only
// instantiated with a global index type at least as wide as the local one, so
// a 32-bit global index type is only tried for 32-bit local indices.
template <typename IndexType>
std::optional<coo_data> build_any_distributed_coo(const LinOp* system_matrix)
{
    using experimental::distributed::Matrix;
    if (auto matrix = dynamic_cast<const Matrix<double, IndexType, int64>*>(
            system_matrix)) {
        return build_distributed_coo(matrix);
    }
    if constexpr (std::is_same<IndexType, int32>::value) {
        if (auto matrix = dynamic_cast<const Matrix<double, int32, int32>*>(
                system_matrix)) {
            return build_distributed_coo(matrix);
        }
    }
    return std::nullopt;
}


}  // namespace detail
}  // namespace petsc
}  // namespace ext
}  // namespace gko


#endif  // GKO_TPL_PETSC_DETAIL_IMPORT_HPP_

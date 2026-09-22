// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_TPL_PETSC_DETAIL_VECTORS_HPP_
#define GKO_TPL_PETSC_DETAIL_VECTORS_HPP_


#include <memory>

#include <petscvec.h>

#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/types.hpp>

#include <gko_tpl/petsc/detail/error.hpp>


namespace gko {
namespace ext {
namespace petsc {
namespace detail {


// Places a caller-owned array into a PETSc vector for the lifetime of this
// object and resets the vector on every exit path, including exceptions, so
// PETSc never keeps a pointer to Ginkgo's data.
class placed_array {
public:
    // `device` must match how the Vec was created: MatCreateVecs on a
    // device matrix yields CUDA vectors, and VecPlaceArray sets only their
    // host array. Placing a host array under a GPU solve leaves the offload
    // mask inconsistent, so the solver can read a stale device array and the
    // result read back is silently wrong.
    placed_array(Vec vec, const PetscScalar* values, bool device)
        : vec_{vec}, device_{device}
    {
        if (device_) {
            GKO_TPL_ASSERT_NO_PETSC_ERRORS(VecCUDAPlaceArray(vec_, values));
        } else {
            GKO_TPL_ASSERT_NO_PETSC_ERRORS(VecPlaceArray(vec_, values));
        }
    }

    ~placed_array()
    {
        if (device_) {
            VecCUDAResetArray(vec_);
        } else {
            VecResetArray(vec_);
        }
    }

    placed_array(const placed_array&) = delete;
    placed_array& operator=(const placed_array&) = delete;

private:
    Vec vec_;
    bool device_;
};


// Calls solve(b, x) with pointers to n values each, in the memory space
// PETSc's vectors live in. `device_vectors` is set only when the matrix was
// created as a device type, which happens only on a device executor, so the
// caller's arrays are already in the right place and pass through untouched.
// Otherwise the vectors are host ones: a host executor passes through too,
// and a device executor is staged through host copies, with x copied back.
template <typename SolveFunction>
void solve_with_arrays(std::shared_ptr<const Executor> exec,
                       bool device_vectors, size_type n, const double* b,
                       double* x, SolveFunction solve)
{
    const auto host = exec->get_master();
    if (device_vectors || exec == host) {
        solve(b, x);
        return;
    }
    array<double> host_b{host, n};
    array<double> host_x{host, n};
    host->copy_from(exec, n, b, host_b.get_data());
    host->copy_from(exec, n, x, host_x.get_data());
    solve(host_b.get_const_data(), host_x.get_data());
    exec->copy_from(host, n, host_x.get_const_data(), x);
}


}  // namespace detail
}  // namespace petsc
}  // namespace ext
}  // namespace gko


#endif  // GKO_TPL_PETSC_DETAIL_VECTORS_HPP_

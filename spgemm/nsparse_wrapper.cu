// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <cuda.h>

#include <HashSpGEMM_volta.hpp>
#include <helper_cuda.h>

#include "benchmark_wrappers.hpp"
#include "core/matrix/csr_builder.hpp"

namespace gko {


template <typename ValueType>
void NSparseCsr<ValueType>::apply_impl(const LinOp* b, LinOp* x) const
{
    auto exec = as<CudaExecutor>(this->get_executor());

    auto b_csr = as<csr>(b);
    auto x_csr = as<csr>(x);
    CSR<int32, ValueType> this_wrap{};
    CSR<int32, ValueType> b_wrap{};
    CSR<int32, ValueType> x_wrap{};
    this_wrap.nrow = this->get_size()[0];
    this_wrap.ncolumn = this->get_size()[1];
    this_wrap.nnz = this->csr_->get_num_stored_elements();
    this_wrap.d_rpt = const_cast<int32*>(this->csr_->get_const_row_ptrs());
    this_wrap.d_colids = const_cast<int32*>(this->csr_->get_const_col_idxs());
    this_wrap.d_values = const_cast<ValueType*>(this->csr_->get_const_values());
    this_wrap.device_malloc = true;

    b_wrap.nrow = b_csr->get_size()[0];
    b_wrap.ncolumn = b_csr->get_size()[1];
    b_wrap.nnz = b_csr->get_num_stored_elements();
    b_wrap.d_rpt = const_cast<int32*>(b_csr->get_const_row_ptrs());
    b_wrap.d_colids = const_cast<int32*>(b_csr->get_const_col_idxs());
    b_wrap.d_values = const_cast<ValueType*>(b_csr->get_const_values());
    b_wrap.device_malloc = true;

    x_wrap.nrow = x_csr->get_size()[0];
    x_wrap.ncolumn = x_csr->get_size()[1];
    x_wrap.nnz = 0;
    x_wrap.d_rpt = x_csr->get_row_ptrs();
    x_wrap.d_colids = nullptr;
    x_wrap.d_values = nullptr;
    x_wrap.device_malloc = true;

    BIN<int32, BIN_NUM> bin(this_wrap.nrow);

    bin.set_max_bin(this_wrap.d_rpt, this_wrap.d_colids, b_wrap.d_rpt,
                    this_wrap.nrow, TS_S_P, TS_S_T);

    this->get_executor()->run(
        "symbolic", [] {},
        [&] { hash_symbolic(this_wrap, b_wrap, x_wrap, bin); }, [] {});

    matrix::CsrBuilder<ValueType, int32> x_builder{x_csr};
    x_builder.get_col_idx_array().resize_and_reset(x_wrap.nnz);
    x_builder.get_value_array().resize_and_reset(x_wrap.nnz);

    x_wrap.d_colids = x_csr->get_col_idxs();
    x_wrap.d_values = x_csr->get_values();

    bin.set_min_bin(this_wrap.nrow, TS_N_P, TS_N_T);

    this->get_executor()->run(
        "numeric", [] {},
        [&] {
            hash_numeric<int32, ValueType, true>(this_wrap, b_wrap, x_wrap,
                                                 bin);
        },
        [] {});
}


template class NSparseCsr<float>;
template class NSparseCsr<double>;

}  // namespace gko

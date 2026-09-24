// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_TPL_COMMON_TEST_LAPLACIAN_HPP_
#define GKO_TPL_COMMON_TEST_LAPLACIAN_HPP_


#include <ginkgo/core/base/dim.hpp>
#include <ginkgo/core/base/matrix_data.hpp>
#include <ginkgo/core/base/types.hpp>


// The 7-point Laplacian on an n x n x n grid with Dirichlet boundaries,
// indexed as i = x + n * y + n * n * z.
template <typename IndexType>
gko::matrix_data<double, IndexType> laplacian_3d(IndexType n)
{
    const auto num_rows = static_cast<gko::size_type>(n) * n * n;
    gko::matrix_data<double, IndexType> data(gko::dim<2>{num_rows, num_rows});
    for (IndexType z = 0; z < n; z++) {
        for (IndexType y = 0; y < n; y++) {
            for (IndexType x = 0; x < n; x++) {
                const IndexType row = x + n * y + n * n * z;
                if (z > 0) {
                    data.nonzeros.emplace_back(row, row - n * n, -1.0);
                }
                if (y > 0) {
                    data.nonzeros.emplace_back(row, row - n, -1.0);
                }
                if (x > 0) {
                    data.nonzeros.emplace_back(row, row - 1, -1.0);
                }
                data.nonzeros.emplace_back(row, row, 6.0);
                if (x < n - 1) {
                    data.nonzeros.emplace_back(row, row + 1, -1.0);
                }
                if (y < n - 1) {
                    data.nonzeros.emplace_back(row, row + n, -1.0);
                }
                if (z < n - 1) {
                    data.nonzeros.emplace_back(row, row + n * n, -1.0);
                }
            }
        }
    }
    return data;
}


#endif  // GKO_TPL_COMMON_TEST_LAPLACIAN_HPP_

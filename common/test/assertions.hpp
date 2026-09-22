// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_TPL_COMMON_TEST_ASSERTIONS_HPP_
#define GKO_TPL_COMMON_TEST_ASSERTIONS_HPP_


#include <algorithm>
#include <cmath>
#include <limits>

#include <gtest/gtest.h>

#include <ginkgo/ginkgo.hpp>


namespace gko_tpl_test {


// ||a - b||_F / max(||a||_F, ||b||_F), computed on the host, the error that
// Ginkgo's (non-installed) GKO_ASSERT_MTX_NEAR bounds.
inline double relative_error(gko::ptr_param<const gko::matrix::Dense<double>> a,
                             gko::ptr_param<const gko::matrix::Dense<double>> b)
{
    EXPECT_EQ(a->get_size(), b->get_size());
    if (a->get_size() != b->get_size()) {
        return std::numeric_limits<double>::infinity();
    }
    const auto host = a->get_executor()->get_master();
    const auto host_a = gko::make_temporary_clone(host, a.get());
    const auto host_b = gko::make_temporary_clone(host, b.get());
    double diff = 0.0;
    double norm_a = 0.0;
    double norm_b = 0.0;
    for (gko::size_type row = 0; row < a->get_size()[0]; row++) {
        for (gko::size_type col = 0; col < a->get_size()[1]; col++) {
            const auto va = host_a->at(row, col);
            const auto vb = host_b->at(row, col);
            diff += (va - vb) * (va - vb);
            norm_a += va * va;
            norm_b += vb * vb;
        }
    }
    const auto scale = std::max(std::sqrt(norm_a), std::sqrt(norm_b));
    return scale > 0.0 ? std::sqrt(diff) / scale : std::sqrt(diff);
}


}  // namespace gko_tpl_test


#define GKO_TPL_ASSERT_MTX_NEAR(_a, _b, _tol) \
    ASSERT_LE(::gko_tpl_test::relative_error(_a, _b), _tol)


#endif  // GKO_TPL_COMMON_TEST_ASSERTIONS_HPP_

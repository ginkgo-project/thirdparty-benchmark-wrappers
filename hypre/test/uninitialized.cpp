// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <memory>

#include <gtest/gtest.h>

#include <ginkgo/ginkgo.hpp>

#include <gko_tpl/hypre/pcg.hpp>


// hypre is never initialized in this executable, so generating must say so
// instead of crashing inside hypre.
TEST(Uninitialized, GeneratingWithoutHypreThrows)
{
    auto ref = gko::ReferenceExecutor::create();
    auto mtx = gko::share(gko::matrix::Csr<double, gko::int32>::create(
        ref, gko::dim<2>{2, 2}, 2));
    using pcg_type = gko::ext::hypre::solver::Pcg<double, gko::int32>;

    ASSERT_THROW(pcg_type::build().on(ref)->generate(mtx),
                 gko::InvalidStateError);
}

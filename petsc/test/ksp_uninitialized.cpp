// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <memory>

#include <gtest/gtest.h>

#include <ginkgo/ginkgo.hpp>

#include <gko_tpl/petsc/ksp.hpp>

#include "common/test/laplacian.hpp"


TEST(KspWithoutPetscEnvironment, ThrowsOnGenerate)
{
    auto ref = gko::ReferenceExecutor::create();
    auto mtx = gko::share(gko::matrix::Csr<double, gko::int32>::create(ref));
    mtx->read(laplacian_3d<gko::int32>(3));
    auto factory = gko::ext::petsc::solver::Ksp<double, gko::int32>::build()
                       .with_options("-ksp_type cg")
                       .on(ref);

    ASSERT_THROW(factory->generate(mtx), gko::InvalidStateError);
}

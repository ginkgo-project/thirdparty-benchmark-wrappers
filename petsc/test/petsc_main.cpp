// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <gtest/gtest.h>

#include <ginkgo/core/base/mpi.hpp>

#include <gko_tpl/petsc/environment.hpp>


// Runs the tests with MPI and PETSc initialized. PETSc is finalized before
// MPI, after the tests have destroyed every solver they created.
int main(int argc, char** argv)
{
    gko::experimental::mpi::environment mpi_env(argc, argv);
    gko::ext::petsc::environment petsc_env;
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

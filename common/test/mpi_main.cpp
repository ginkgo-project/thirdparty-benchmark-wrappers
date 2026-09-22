// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <gtest/gtest.h>

#include <ginkgo/core/base/mpi.hpp>


// Runs the tests with MPI initialized, on every rank.
int main(int argc, char** argv)
{
    gko::experimental::mpi::environment mpi_env(argc, argv);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

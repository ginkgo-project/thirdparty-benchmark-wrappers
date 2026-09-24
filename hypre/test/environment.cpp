// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <gtest/gtest.h>

#include <HYPRE_utilities.h>

#include <gko_tpl/hypre/environment.hpp>


// This executable uses the MPI-only main, so hypre starts uninitialized.
TEST(Environment, InitializesAndFinalizesHypre)
{
    ASSERT_EQ(HYPRE_Initialized(), 0);
    {
        gko::ext::hypre::environment env;
        ASSERT_NE(HYPRE_Initialized(), 0);
    }
    ASSERT_EQ(HYPRE_Initialized(), 0);
}


TEST(Environment, NestsWithoutFinalizingTwice)
{
    gko::ext::hypre::environment outer;
    {
        // The inner guard finds hypre initialized and must leave it alone.
        gko::ext::hypre::environment inner;
    }
    ASSERT_NE(HYPRE_Initialized(), 0);
}

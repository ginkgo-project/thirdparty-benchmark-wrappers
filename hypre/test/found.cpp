// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <string>

#include <gtest/gtest.h>

#include <HYPRE_config.h>
#include <HYPRE_utilities.h>


TEST(Hypre, LinksAgainstAUsableBuild)
{
    ASSERT_FALSE(std::string{HYPRE_RELEASE_VERSION}.empty());
    // 2.32.0 is the oldest version this component is tested against.
    ASSERT_GE(HYPRE_RELEASE_NUMBER, 23200);
    ASSERT_EQ(HYPRE_Initialized(), 0);
}

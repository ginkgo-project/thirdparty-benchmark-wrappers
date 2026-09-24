// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <string>

#include <gtest/gtest.h>

#include <ginkgo/core/base/exception.hpp>

#include <gko_tpl/hypre/detail/options.hpp>


TEST(Options, MapsDocumentedNames)
{
    ASSERT_EQ(gko::ext::hypre::detail::coarsen_type_id("PMIS"), 8);
    ASSERT_EQ(gko::ext::hypre::detail::coarsen_type_id("HMIS"), 10);
    ASSERT_EQ(gko::ext::hypre::detail::coarsen_type_id("Falgout"), 6);
    ASSERT_EQ(gko::ext::hypre::detail::interp_type_id("ext+i"), 6);
    ASSERT_EQ(gko::ext::hypre::detail::interp_type_id("classical"), 0);
    ASSERT_EQ(gko::ext::hypre::detail::relax_type_id("Jacobi"), 0);
    ASSERT_EQ(gko::ext::hypre::detail::relax_type_id("l1-Jacobi"), 18);
}


TEST(Options, RejectsUnknownNamesAndListsTheAcceptedOnes)
{
    try {
        gko::ext::hypre::detail::coarsen_type_id("PIMS");
        FAIL() << "an unknown coarsening name must throw";
    } catch (const gko::InvalidStateError& error) {
        const std::string message{error.what()};
        ASSERT_NE(message.find("PIMS"), std::string::npos);
        ASSERT_NE(message.find("PMIS"), std::string::npos);
    }
    ASSERT_THROW(gko::ext::hypre::detail::interp_type_id("nope"),
                 gko::InvalidStateError);
    ASSERT_THROW(gko::ext::hypre::detail::relax_type_id("nope"),
                 gko::InvalidStateError);
}

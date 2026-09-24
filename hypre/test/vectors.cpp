// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <vector>

#include <gtest/gtest.h>

#include <_hypre_parcsr_mv.h>

#include <ginkgo/ginkgo.hpp>

#include <gko_tpl/hypre/detail/policy.hpp>
#include <gko_tpl/hypre/detail/vectors.hpp>


TEST(PlacedParVector, WritesThroughToTheCallersMemory)
{
    std::vector<double> values{1.0, 2.0, 3.0, 4.0};
    {
        gko::ext::hypre::detail::placed_vector vec{
            MPI_COMM_SELF, 4, 0, 4, values.data(), HYPRE_MEMORY_HOST};
        ASSERT_EQ(hypre_ParVectorSetConstantValues(vec.get(), 7.0), 0);
    }
    // hypre wrote into the vector this test owns, and destroying the wrapper
    // did not free it.
    ASSERT_EQ(values[0], 7.0);
    ASSERT_EQ(values[3], 7.0);
}


TEST(PlacedParVector, RePointsAtAnotherArrayWithoutCopying)
{
    // What apply reuses instead of rebuilding per solve: nothing is copied,
    // and the previous array is neither written nor freed.
    std::vector<double> first{1.0, 2.0, 3.0, 4.0};
    std::vector<double> second{5.0, 6.0, 7.0, 8.0};
    {
        gko::ext::hypre::detail::placed_vector vec{
            MPI_COMM_SELF, 4, 0, 4, first.data(), HYPRE_MEMORY_HOST};
        ASSERT_EQ(hypre_ParVectorSetConstantValues(vec.get(), 7.0), 0);

        vec.set_values(second.data());

        ASSERT_EQ(hypre_ParVectorSetConstantValues(vec.get(), 9.0), 0);
    }
    // The second array is the one hypre wrote after the re-pointing, and the
    // first kept what it held before it.
    ASSERT_EQ(first[0], 7.0);
    ASSERT_EQ(first[3], 7.0);
    ASSERT_EQ(second[0], 9.0);
    ASSERT_EQ(second[3], 9.0);
}


#if !defined(HYPRE_USING_GPU)
TEST(ProcessPolicy, RejectsADeviceLocationOnAHostOnlyHypre)
{
    // Exercises the rejection with just the memory location a device
    // executor would produce, no device executor needed.
    gko::ext::hypre::detail::active_memory_location().reset();

    ASSERT_THROW(
        gko::ext::hypre::detail::set_process_policy(HYPRE_MEMORY_DEVICE),
        gko::InvalidStateError);

    // Nothing was installed, so a host solver can still start afterwards.
    ASSERT_FALSE(gko::ext::hypre::detail::active_memory_location().has_value());
    gko::ext::hypre::detail::active_memory_location().reset();
}
#endif


TEST(ProcessPolicy, RejectsASecondConflictingPolicy)
{
    gko::ext::hypre::detail::active_memory_location().reset();
    gko::ext::hypre::detail::set_process_policy(HYPRE_MEMORY_HOST);
    // Setting the same policy again is what a second host solver does.
    ASSERT_NO_THROW(
        gko::ext::hypre::detail::set_process_policy(HYPRE_MEMORY_HOST));
    ASSERT_THROW(
        gko::ext::hypre::detail::set_process_policy(HYPRE_MEMORY_DEVICE),
        gko::InvalidStateError);
    gko::ext::hypre::detail::active_memory_location().reset();
}

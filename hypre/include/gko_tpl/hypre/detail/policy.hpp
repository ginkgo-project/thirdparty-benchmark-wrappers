// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_TPL_HYPRE_DETAIL_POLICY_HPP_
#define GKO_TPL_HYPRE_DETAIL_POLICY_HPP_


#include <optional>

#include <HYPRE_utilities.h>

#include <ginkgo/core/base/exception_helpers.hpp>

#include <gko_tpl/hypre/detail/error.hpp>


namespace gko {
namespace ext {
namespace hypre {
namespace detail {


// hypre's memory location and execution policy are process-global, so every
// solver in a process shares them. The one in force is remembered here; the
// function-local static gives one instance across translation units.
inline std::optional<HYPRE_MemoryLocation>& active_memory_location()
{
    static std::optional<HYPRE_MemoryLocation> location;
    return location;
}


// Sets hypre's process-wide policy, or throws if a policy is already in force
// and differs. Running a solver under the wrong policy would silently produce
// wrong results or crash inside hypre, so the conflict is reported instead.
inline void set_process_policy(HYPRE_MemoryLocation location)
{
    auto& active = active_memory_location();
    if (active && *active != location) {
        GKO_INVALID_STATE(
            "hypre's memory location and execution policy are process-global, "
            "and this process already runs with the other one; a "
            "gko::ext::hypre solver on a host executor and one on a device "
            "executor cannot coexist");
    }
    // A hypre built without GPU support silently maps HYPRE_MEMORY_DEVICE to
    // HOST, so it would read Ginkgo's device pointers as host memory. This is
    // a build-time property, identical on every rank, so throwing here
    // cannot make ranks diverge.
    if (location == HYPRE_MEMORY_DEVICE) {
#if !defined(HYPRE_USING_GPU)
        GKO_INVALID_STATE(
            "a gko::ext::hypre solver on a Ginkgo device executor needs a "
            "hypre built with GPU support, but this hypre was built without "
            "it (HYPRE_USING_GPU is not defined in its HYPRE_config.h); it "
            "would treat the device pointers as host memory instead of "
            "reporting an error. Use a host executor, or rebuild hypre with "
            "CUDA/HIP/SYCL support");
#endif
    }
    if (!active) {
        GKO_TPL_ASSERT_NO_HYPRE_ERRORS(HYPRE_SetMemoryLocation(location));
        GKO_TPL_ASSERT_NO_HYPRE_ERRORS(HYPRE_SetExecutionPolicy(
            location == HYPRE_MEMORY_DEVICE ? HYPRE_EXEC_DEVICE
                                            : HYPRE_EXEC_HOST));
        active = location;
    }
}


}  // namespace detail
}  // namespace hypre
}  // namespace ext
}  // namespace gko


#endif  // GKO_TPL_HYPRE_DETAIL_POLICY_HPP_

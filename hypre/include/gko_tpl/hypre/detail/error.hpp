// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_TPL_HYPRE_DETAIL_ERROR_HPP_
#define GKO_TPL_HYPRE_DETAIL_ERROR_HPP_


#include <string>

#include <HYPRE_utilities.h>

#include <ginkgo/core/base/exception.hpp>


namespace gko {
namespace ext {
namespace hypre {
namespace detail {


// hypre's description of an error code. HYPRE_DescribeError writes into a
// caller-provided buffer; hypre's own uses are bounded well below this size.
inline std::string describe_error(HYPRE_Int error_code)
{
    char buffer[256] = {};
    HYPRE_DescribeError(error_code, buffer);
    return std::string{buffer};
}


}  // namespace detail
}  // namespace hypre
}  // namespace ext
}  // namespace gko


/**
 * Throws gko::InvalidStateError, naming the call, hypre's error code and its
 * description, if the given hypre call does not return 0.
 *
 * hypre also keeps a global error flag, which the macro clears, so that an
 * error raised here is not reported again by the next unrelated call.
 */
#define GKO_TPL_ASSERT_NO_HYPRE_ERRORS(_hypre_call)                         \
    do {                                                                    \
        const HYPRE_Int _hypre_error = (_hypre_call);                       \
        if (_hypre_error != 0) {                                            \
            const auto _hypre_description =                                 \
                ::gko::ext::hypre::detail::describe_error(_hypre_error);    \
            HYPRE_ClearAllErrors();                                         \
            throw ::gko::InvalidStateError(                                 \
                __FILE__, __LINE__, __func__,                               \
                std::string("hypre error code ") +                          \
                    std::to_string(static_cast<int>(_hypre_error)) + " (" + \
                    _hypre_description + ") returned by " #_hypre_call);    \
        }                                                                   \
    } while (false)


#endif  // GKO_TPL_HYPRE_DETAIL_ERROR_HPP_

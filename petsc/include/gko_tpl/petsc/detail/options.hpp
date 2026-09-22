// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_TPL_PETSC_DETAIL_OPTIONS_HPP_
#define GKO_TPL_PETSC_DETAIL_OPTIONS_HPP_


#include <atomic>
#include <cctype>
#include <cstdint>
#include <sstream>
#include <string>


namespace gko {
namespace ext {
namespace petsc {
namespace detail {


// Source of the unique options prefix of each Ksp instance. It is an inline
// variable, so that every translation unit refers to the same counter; a
// counter with internal linkage would make the Ksp constructors instantiated
// in different translation units differ, which violates the ODR.
inline std::atomic<std::uint64_t> next_instance_id{0};


// Rewrites every option name "-name" in `options` to "-<prefix>name". A token
// is an option name if it starts with '-' followed by a letter; all other
// tokens, including negative numbers, are values and stay unchanged.
inline std::string prefix_options(const std::string& options,
                                  const std::string& prefix)
{
    std::istringstream tokens{options};
    std::string token;
    std::string result;
    while (tokens >> token) {
        if (token.size() > 1 && token[0] == '-' &&
            std::isalpha(static_cast<unsigned char>(token[1]))) {
            token = "-" + prefix + token.substr(1);
        }
        if (!result.empty()) {
            result += ' ';
        }
        result += token;
    }
    return result;
}


}  // namespace detail
}  // namespace petsc
}  // namespace ext
}  // namespace gko


#endif  // GKO_TPL_PETSC_DETAIL_OPTIONS_HPP_

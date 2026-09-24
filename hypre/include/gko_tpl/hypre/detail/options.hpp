// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_TPL_HYPRE_DETAIL_OPTIONS_HPP_
#define GKO_TPL_HYPRE_DETAIL_OPTIONS_HPP_


#include <map>
#include <string>

#include <HYPRE_utilities.h>

#include <ginkgo/core/base/exception_helpers.hpp>


namespace gko {
namespace ext {
namespace hypre {
namespace detail {


// The names accepted for BoomerAMG's algorithm choices, and hypre's integer
// for each, as documented at HYPRE_BoomerAMGSetCoarsenType,
// HYPRE_BoomerAMGSetInterpType and HYPRE_BoomerAMGSetRelaxType. Only the
// options the benchmarks use are listed; anything else goes through the
// amg_setup callback.
inline const std::map<std::string, HYPRE_Int>& coarsen_types()
{
    static const std::map<std::string, HYPRE_Int> types{{"CLJP", 0},
                                                        {"Ruge-Stueben", 3},
                                                        {"Falgout", 6},
                                                        {"PMIS", 8},
                                                        {"HMIS", 10}};
    return types;
}


inline const std::map<std::string, HYPRE_Int>& interp_types()
{
    static const std::map<std::string, HYPRE_Int> types{
        {"classical", 0}, {"direct", 3},   {"multipass", 4}, {"ext+i", 6},
        {"ext+i-cc", 7},  {"standard", 8}, {"ext", 14}};
    return types;
}


inline const std::map<std::string, HYPRE_Int>& relax_types()
{
    static const std::map<std::string, HYPRE_Int> types{
        {"Jacobi", 0},
        {"hybrid-GS-forward", 3},
        {"hybrid-GS-backward", 4},
        {"hybrid-symm-GS", 6},
        {"l1-symm-GS", 8},
        {"Gaussian-elimination", 9},
        {"l1-GS-forward", 13},
        {"l1-GS-backward", 14},
        {"Chebyshev", 16},
        {"l1-Jacobi", 18}};
    return types;
}


inline HYPRE_Int lookup(const std::map<std::string, HYPRE_Int>& types,
                        const std::string& name, const std::string& what)
{
    const auto found = types.find(name);
    if (found != types.end()) {
        return found->second;
    }
    std::string accepted;
    for (const auto& [key, value] : types) {
        accepted += accepted.empty() ? "" : ", ";
        accepted += key;
    }
    GKO_INVALID_STATE("unknown " + what + " '" + name +
                      "'; accepted: " + accepted);
}


inline HYPRE_Int coarsen_type_id(const std::string& name)
{
    return lookup(coarsen_types(), name, "coarsening type");
}


inline HYPRE_Int interp_type_id(const std::string& name)
{
    return lookup(interp_types(), name, "interpolation type");
}


inline HYPRE_Int relax_type_id(const std::string& name)
{
    return lookup(relax_types(), name, "relaxation type");
}


}  // namespace detail
}  // namespace hypre
}  // namespace ext
}  // namespace gko


#endif  // GKO_TPL_HYPRE_DETAIL_OPTIONS_HPP_

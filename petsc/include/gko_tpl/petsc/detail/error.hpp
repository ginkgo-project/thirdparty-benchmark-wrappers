// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_TPL_PETSC_DETAIL_ERROR_HPP_
#define GKO_TPL_PETSC_DETAIL_ERROR_HPP_


#include <string>

#include <petscsys.h>

#include <ginkgo/core/base/exception.hpp>


/**
 * Throws gko::InvalidStateError, naming the call and PETSc's error code, if
 * the given PETSc call does not return PETSC_SUCCESS.
 */
#define GKO_TPL_ASSERT_NO_PETSC_ERRORS(_petsc_call)                  \
    do {                                                             \
        const PetscErrorCode _petsc_error = (_petsc_call);           \
        if (_petsc_error != PETSC_SUCCESS) {                         \
            throw ::gko::InvalidStateError(                          \
                __FILE__, __LINE__, __func__,                        \
                std::string("PETSc error code ") +                   \
                    std::to_string(static_cast<int>(_petsc_error)) + \
                    " returned by " #_petsc_call);                   \
        }                                                            \
    } while (false)


#endif  // GKO_TPL_PETSC_DETAIL_ERROR_HPP_

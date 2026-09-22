# gko_tpl_find_petsc([REQUIRED] [HINTS <prefix>...])
#
# Finds PETSc through pkg-config and provides the imported target
# PkgConfig::PETSC, plus PETSC_FOUND, PETSC_VERSION and PETSC_PREFIX.
#
# PETSc installs PETSc.pc and/or petsc.pc under $PETSC_DIR/lib/pkgconfig for a
# prefix install, and under $PETSC_DIR/$PETSC_ARCH/lib/pkgconfig for an
# in-place build. Those are searched first, then the HINTS prefixes, then
# PKG_CONFIG_PATH. An existing PkgConfig::PETSC, e.g. created by a Ginkgo
# built in the same tree, is reused.
function(gko_tpl_find_petsc)
    cmake_parse_arguments(PARSE_ARGV 0 arg "REQUIRED" "" "HINTS")
    if(TARGET PkgConfig::PETSC)
        set(PETSC_FOUND TRUE PARENT_SCOPE)
        return()
    endif()
    find_package(PkgConfig QUIET)
    if(NOT PKG_CONFIG_FOUND)
        if(arg_REQUIRED)
            message(
                FATAL_ERROR
                "PETSc is found through pkg-config, which is missing"
            )
        endif()
        set(PETSC_FOUND FALSE PARENT_SCOPE)
        return()
    endif()
    set(dirs "")
    if(DEFINED ENV{PETSC_DIR})
        if(DEFINED ENV{PETSC_ARCH})
            list(APPEND dirs "$ENV{PETSC_DIR}/$ENV{PETSC_ARCH}/lib/pkgconfig")
        endif()
        list(APPEND dirs "$ENV{PETSC_DIR}/lib/pkgconfig")
    endif()
    foreach(hint IN LISTS arg_HINTS)
        if(hint)
            list(APPEND dirs "${hint}/lib/pkgconfig")
        endif()
    endforeach()
    set(saved_path "$ENV{PKG_CONFIG_PATH}")
    if(saved_path)
        list(APPEND dirs "${saved_path}")
    endif()
    list(JOIN dirs ":" search_path)
    set(ENV{PKG_CONFIG_PATH} "${search_path}")
    if(arg_REQUIRED)
        pkg_search_module(PETSC REQUIRED IMPORTED_TARGET GLOBAL PETSc petsc)
    else()
        pkg_search_module(PETSC QUIET IMPORTED_TARGET GLOBAL PETSc petsc)
    endif()
    set(ENV{PKG_CONFIG_PATH} "${saved_path}")
    set(PETSC_FOUND "${PETSC_FOUND}" PARENT_SCOPE)
    set(PETSC_VERSION "${PETSC_VERSION}" PARENT_SCOPE)
    set(PETSC_PREFIX "${PETSC_PREFIX}" PARENT_SCOPE)
endfunction()

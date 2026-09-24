# gko_tpl_find_hypre([REQUIRED] [HINTS <prefix>...])
#
# Finds hypre and provides the imported target HYPRE::HYPRE, plus HYPRE_FOUND,
# HYPRE_VERSION, HYPRE_PREFIX and HYPRE_USING_CUDA.
#
# A hypre built with CMake installs HYPREConfig.cmake and is found through it,
# with its own dependencies attached. A hypre built by PETSc's --download-hypre
# has no package config: it is found by its headers and library under
# HYPRE_ROOT, for which the PETSc prefix works, since PETSc installs hypre
# into it.
function(gko_tpl_find_hypre)
    cmake_parse_arguments(PARSE_ARGV 0 arg "REQUIRED" "" "HINTS")
    if(TARGET HYPRE::HYPRE)
        set(HYPRE_FOUND TRUE PARENT_SCOPE)
        return()
    endif()
    find_package(HYPRE CONFIG QUIET HINTS ${arg_HINTS})
    if(NOT TARGET HYPRE::HYPRE)
        find_path(
            HYPRE_INCLUDE_DIR
            NAMES HYPRE.h
            HINTS ${arg_HINTS} $ENV{HYPRE_ROOT} ${HYPRE_ROOT}
            PATH_SUFFIXES include
        )
        find_library(
            HYPRE_LIBRARY
            NAMES HYPRE
            HINTS ${arg_HINTS} $ENV{HYPRE_ROOT} ${HYPRE_ROOT}
            PATH_SUFFIXES lib lib64
        )
        if(HYPRE_INCLUDE_DIR AND HYPRE_LIBRARY)
            add_library(HYPRE::HYPRE UNKNOWN IMPORTED GLOBAL)
            set_target_properties(
                HYPRE::HYPRE
                PROPERTIES
                    IMPORTED_LOCATION "${HYPRE_LIBRARY}"
                    INTERFACE_INCLUDE_DIRECTORIES "${HYPRE_INCLUDE_DIR}"
            )
        endif()
    endif()
    if(NOT TARGET HYPRE::HYPRE)
        if(arg_REQUIRED)
            message(
                FATAL_ERROR
                "hypre not found; set HYPRE_ROOT to its installation (a PETSc prefix built with --download-hypre works)"
            )
        endif()
        set(HYPRE_FOUND FALSE PARENT_SCOPE)
        return()
    endif()
    # The configuration header answers what the library was built for.
    get_target_property(dirs HYPRE::HYPRE INTERFACE_INCLUDE_DIRECTORIES)
    find_file(HYPRE_CONFIG_HEADER NAMES HYPRE_config.h HINTS ${dirs})
    if(HYPRE_CONFIG_HEADER)
        file(READ "${HYPRE_CONFIG_HEADER}" config)
        string(
            REGEX MATCH
            "#define HYPRE_RELEASE_VERSION \"([^\"]+)\""
            _match
            "${config}"
        )
        set(HYPRE_VERSION "${CMAKE_MATCH_1}")
        string(
            REGEX MATCH
            "\n#define HYPRE_USING_CUDA[^A-Za-z0-9_]"
            cuda
            "${config}"
        )
        if(cuda)
            set(HYPRE_USING_CUDA TRUE)
            find_package(CUDAToolkit REQUIRED)
            target_link_libraries(HYPRE::HYPRE INTERFACE CUDA::cudart)
        else()
            set(HYPRE_USING_CUDA FALSE)
        endif()
        get_filename_component(dir "${HYPRE_CONFIG_HEADER}" DIRECTORY)
        get_filename_component(HYPRE_PREFIX "${dir}" DIRECTORY)
    endif()
    # 2.32 is the oldest tested version, and the one PETSc bundles; the
    # component targets 3.2 and guards what is newer than 2.32.
    if(HYPRE_VERSION AND HYPRE_VERSION VERSION_LESS 2.32.0)
        message(
            FATAL_ERROR
            "hypre ${HYPRE_VERSION} is older than the tested minimum 2.32.0"
        )
    endif()
    set(HYPRE_FOUND TRUE PARENT_SCOPE)
    set(HYPRE_VERSION "${HYPRE_VERSION}" PARENT_SCOPE)
    set(HYPRE_PREFIX "${HYPRE_PREFIX}" PARENT_SCOPE)
    set(HYPRE_USING_CUDA "${HYPRE_USING_CUDA}" PARENT_SCOPE)
endfunction()

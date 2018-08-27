# - Try to find MFEM
# Once done this will define
#
#  MFEM_FOUND        - system has MFEM
#  MFEM_INCLUDE_DIRS - include directories for MFEM
#  MFEM_LIBRARIES    - libraries for MFEM
#
# Variables used by this module. They can change the default behaviour and
# need to be set before calling find_package:
#
#  MFEM_DIR          - Prefix directory of the MFEM installation
#  MFEM_INCLUDE_DIR  - Include directory of the MFEM installation
#                       (set only if different from ${MFEM_DIR}/include)
#  MFEM_LIB_DIR      - Library directory of the MFEM installation
#                       (set only if different from ${MFEM_DIR}/lib)
#  MFEM_TEST_RUNS    - Skip tests building and running a test
#                       executable linked against MFEM libraries
#  MFEM_LIB_SUFFIX   - Also search for non-standard library names with the
#                       given suffix appended
#
# NOTE: This file was modified from a ParMFEM detection script 

#=============================================================================
# Copyright (C) 2015 Jack Poulson. All rights reserved.
#
# Copyright (C) 2010-2012 Garth N. Wells, Anders Logg, Johannes Ring
# and Florian Rathgeber. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in
#    the documentation and/or other materials provided with the
#    distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#=============================================================================

if(NOT MFEM_INCLUDE_DIR)
  find_path(MFEM_INCLUDE_DIR mfem.hpp
    HINTS ${MFEM_INCLUDE_DIR} ENV MFEM_INCLUDE_DIR ${MFEM_DIR} ENV MFEM_DIR
    PATH_SUFFIXES include
    DOC "Directory where the MFEM header files are located"
  )
endif()

if(MFEM_LIBRARIES)
  set(MFEM_LIBRARY ${MFEM_LIBRARIES})
endif()
if(NOT MFEM_LIBRARY)
  find_library(MFEM_LIBRARY
    NAMES mfem mfem${MFEM_LIB_SUFFIX}
    HINTS ${MFEM_LIB_DIR} ENV MFEM_LIB_DIR ${MFEM_DIR} ENV MFEM_DIR
    PATH_SUFFIXES lib
    DOC "Directory where the MFEM library is located"
  )
endif()

# Standard package handling
include(FindPackageHandleStandardArgs)
if(CMAKE_VERSION VERSION_GREATER 2.8.2)
  find_package_handle_standard_args(MFEM
    REQUIRED_VARS MFEM_LIBRARY MFEM_INCLUDE_DIR # MFEM_TEST_RUNS
    VERSION_VAR MFEM_VERSION_STRING)
else()
  find_package_handle_standard_args(MFEM
    REQUIRED_VARS MFEM_LIBRARY MFEM_INCLUDE_DIR #MFEM_TEST_RUNS
    )
endif()

if(MFEM_FOUND)
  set(MFEM_LIBRARIES ${MFEM_LIBRARY})
  set(MFEM_INCLUDE_DIRS ${MFEM_INCLUDE_DIR})
endif()

mark_as_advanced(MFEM_INCLUDE_DIR MFEM_LIBRARY)

if (MFEM_FOUND AND NOT TARGET mfem::mfem)
    add_library(mfem::mfem INTERFACE IMPORTED)
    set_target_properties(mfem::mfem PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${MFEM_INCLUDE_DIRS}"
        INTERFACE_LINK_LIBRARIES "${MFEM_LIBRARIES}"
        )
endif()

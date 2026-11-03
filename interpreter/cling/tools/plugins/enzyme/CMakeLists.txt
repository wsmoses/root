#-------------------------------------------------------------------------------
# CLING - the C++ LLVM-based InterpreterG :)
#
# This file is dual-licensed: you can choose to license it under the University
# of Illinois Open Source License or the GNU Lesser General Public License. See
# LICENSE.TXT for details.
#-------------------------------------------------------------------------------

include(ExternalProject)

set(enzyme_install_dir ${CMAKE_BINARY_DIR}/etc/cling/)
set(ENZYME_INCLUDE_DIRS ${enzyme_install_dir}/plugins/include PARENT_SCOPE)
set(ENZYME_CXX_FLAGS ${CMAKE_CXX_FLAGS})

set(_enzyme_build_type ${CMAKE_CFG_INTDIR})
if(NOT _enzyme_build_type STREQUAL "" AND NOT _enzyme_build_type STREQUAL ".")
  set(EXTRA_BUILD_ARGS --config $<CONFIG>)
endif()

if(NOT MSVC AND CMAKE_VERSION VERSION_LESS 3.31.1)
  list(APPEND EXTRA_BUILD_ARGS "-j 1")
endif()

set(_ENZYME_LIBRARY_PATH ${CMAKE_CURRENT_BINARY_DIR}/enzyme-prefix/src/enzyme-build/${_enzyme_build_type}/lib${LLVM_LIBDIR_SUFFIX})

if(CMAKE_GENERATOR MATCHES Ninja)
  set(ENZYME_BYPRODUCTS
    ${_ENZYME_LIBRARY_PATH}/${CMAKE_STATIC_LIBRARY_PREFIX}ClangEnzymeStatic${CMAKE_STATIC_LIBRARY_SUFFIX}
  )
endif()

set(_enzyme_extra_cmake_args)
if(APPLE)
  list(APPEND _enzyme_extra_cmake_args -DCMAKE_OSX_SYSROOT=${CMAKE_OSX_SYSROOT})
endif()

if(CMAKE_CXX_STANDARD)
  list(APPEND _enzyme_extra_cmake_args -DCMAKE_CXX_STANDARD=${CMAKE_CXX_STANDARD})
endif()

if(LLVM_FORCE_USE_OLD_TOOLCHAIN)
  list(APPEND _enzyme_extra_cmake_args -DLLVM_FORCE_USE_OLD_TOOLCHAIN=${LLVM_FORCE_USE_OLD_TOOLCHAIN})
endif()

if(zstd_INCLUDE_DIR AND zstd_LIBRARY)
  list(APPEND _enzyme_extra_cmake_args
       -Dzstd_INCLUDE_DIR:PATH=${zstd_INCLUDE_DIR}
       -Dzstd_LIBRARY:FILEPATH=${zstd_LIBRARY})
endif()

list(APPEND _enzyme_extra_cmake_args -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=${CMAKE_INSTALL_RPATH_USE_LINK_PATH})

set(_enzyme_extra_settings
  LOG_DOWNLOAD ON
  LOG_CONFIGURE ON
  LOG_BUILD ON
  LOG_INSTALL ON
  LOG_OUTPUT_ON_FAILURE ON
)

if(DEFINED ENZYME_SOURCE_DIR)
  get_filename_component(_enzyme_abs_src "${ENZYME_SOURCE_DIR}" ABSOLUTE BASE_DIR "${CMAKE_BINARY_DIR}")
  list(APPEND _enzyme_extra_settings
    DOWNLOAD_COMMAND ""
    SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/wrapper
  )
  list(APPEND _enzyme_extra_cmake_args -DENZYME_SRC_DIR=${_enzyme_abs_src})
else()
  list(APPEND _enzyme_extra_settings
    GIT_REPOSITORY https://github.com/EnzymeAD/Enzyme.git
    GIT_TAG v0.0.297
    PATCH_COMMAND ${CMAKE_COMMAND} -E copy
                  ${CMAKE_CURRENT_SOURCE_DIR}/wrapper/CMakeLists.txt
                  <SOURCE_DIR>/CMakeLists.txt
  )
  list(APPEND _enzyme_extra_cmake_args -DENZYME_SRC_DIR=<SOURCE_DIR>)
endif()

set(_enzyme_deps clingInterpreter)
foreach(_dep LLVMTableGen LLVMDemangle LLVMSupport intrinsics_gen)
  if(TARGET ${_dep})
    list(APPEND _enzyme_deps ${_dep})
  endif()
endforeach()

ExternalProject_Add(
  enzyme
  UPDATE_COMMAND ""
  CMAKE_ARGS -G ${CMAKE_GENERATOR}
             -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
             -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
             -DCMAKE_C_FLAGS=${CMAKE_C_FLAGS}
             -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
             -DCMAKE_CXX_FLAGS=${ENZYME_CXX_FLAGS}
             -DCMAKE_CXX_FLAGS_DEBUG=${CMAKE_CXX_FLAGS_DEBUG}
             -DCMAKE_INSTALL_PREFIX=${enzyme_install_dir}/plugins
             -DLLVM_DIR=${LLVM_DIR}
             -DClang_DIR=${CLANG_CMAKE_DIR}
             ${_enzyme_extra_cmake_args}
  BUILD_COMMAND ${CMAKE_COMMAND} --build . ${EXTRA_BUILD_ARGS}
  INSTALL_COMMAND ${CMAKE_COMMAND} --build . ${EXTRA_BUILD_ARGS} --target install
  BUILD_BYPRODUCTS ${ENZYME_BYPRODUCTS}
  ${_enzyme_extra_settings}
  DEPENDS ${_enzyme_deps}
)

add_library(ClangEnzymeStatic IMPORTED STATIC GLOBAL)
add_dependencies(ClangEnzymeStatic enzyme)
set_property(TARGET ClangEnzymeStatic PROPERTY IMPORTED_LOCATION
  ${_ENZYME_LIBRARY_PATH}/${CMAKE_STATIC_LIBRARY_PREFIX}ClangEnzymeStatic${CMAKE_STATIC_LIBRARY_SUFFIX})

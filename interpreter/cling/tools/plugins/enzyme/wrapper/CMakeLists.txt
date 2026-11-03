cmake_minimum_required(VERSION 3.20)
project(ClingEnzymeWrapper LANGUAGES C CXX)

if(POLICY CMP0116)
  cmake_policy(SET CMP0116 NEW)
endif()

if(NOT DEFINED ENZYME_SRC_DIR)
  message(FATAL_ERROR "ENZYME_SRC_DIR must be provided")
endif()

if(EXISTS "${ENZYME_SRC_DIR}/enzyme/Enzyme/Enzyme.cpp")
  set(ENZYME_ROOT "${ENZYME_SRC_DIR}/enzyme")
elseif(EXISTS "${ENZYME_SRC_DIR}/Enzyme/Enzyme.cpp")
  set(ENZYME_ROOT "${ENZYME_SRC_DIR}")
else()
  message(FATAL_ERROR "Cannot locate Enzyme sources under ENZYME_SRC_DIR='${ENZYME_SRC_DIR}'")
endif()

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(EXISTS "${LLVM_DIR}/lib/cmake/llvm/LLVMConfig.cmake")
  set(LLVM_DIR "${LLVM_DIR}/lib/cmake/llvm")
endif()

find_package(LLVM REQUIRED CONFIG PATHS "${LLVM_DIR}" NO_DEFAULT_PATH)
find_package(Clang REQUIRED CONFIG PATHS "${Clang_DIR}" NO_DEFAULT_PATH)

list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(AddLLVM)
include(TableGen)

separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND ${LLVM_DEFINITIONS})
add_definitions(${LLVM_DEFINITIONS_LIST})
separate_arguments(CLANG_DEFINITIONS_LIST NATIVE_COMMAND ${CLANG_DEFINITIONS})
add_definitions(${CLANG_DEFINITIONS_LIST})

add_definitions(
  -DENZYME_VERSION_MAJOR=0
  -DENZYME_VERSION_MINOR=0
  -DENZYME_VERSION_PATCH=79
)

include_directories(${CLANG_INCLUDE_DIRS} ${LLVM_INCLUDE_DIRS})

# Handle optional ZLIB::ZLIB interface dependency on LLVMSupport in un-installed build trees
if(TARGET LLVMSupport)
  get_target_property(_tbl_linked_libs LLVMSupport INTERFACE_LINK_LIBRARIES)
  if(_tbl_linked_libs)
    if("ZLIB::ZLIB" IN_LIST _tbl_linked_libs AND NOT TARGET ZLIB::ZLIB)
      find_package(ZLIB QUIET)
      if(NOT TARGET ZLIB::ZLIB)
        list(REMOVE_ITEM _tbl_linked_libs "ZLIB::ZLIB")
        set_property(TARGET LLVMSupport PROPERTY INTERFACE_LINK_LIBRARIES "${_tbl_linked_libs}")
      endif()
    endif()
  endif()
endif()

# Locate LLVM source include dir containing llvm/Analysis/ScalarEvolution.h
set(LLVM_SRC_IDIR "")
foreach(_idir IN LISTS LLVM_INCLUDE_DIRS)
  if(EXISTS "${_idir}/llvm/Analysis/ScalarEvolution.h")
    set(LLVM_SRC_IDIR "${_idir}")
    break()
  endif()
endforeach()
if(NOT LLVM_SRC_IDIR)
  message(FATAL_ERROR "Could not locate llvm/Analysis/ScalarEvolution.h in LLVM_INCLUDE_DIRS='${LLVM_INCLUDE_DIRS}'")
endif()

file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/include/SCEV")

file(READ "${LLVM_SRC_IDIR}/llvm/Analysis/ScalarEvolution.h" _scev_text)
string(REPLACE "private" "public" _scev_text "${_scev_text}")
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/include/SCEV/ScalarEvolution.h" "${_scev_text}")

file(READ "${LLVM_SRC_IDIR}/llvm/Analysis/TargetLibraryInfo.h" _tli_text)
string(REPLACE "class TargetLibraryInfo {" "class TargetLibraryInfo {public:" _tli_text "${_tli_text}")
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/include/SCEV/TargetLibraryInfo.h" "${_tli_text}")

file(READ "${LLVM_SRC_IDIR}/llvm/Transforms/Utils/ScalarEvolutionExpander.h" _scev_exp_text)
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/include/SCEV/ScalarEvolutionExpander.h" "${_scev_exp_text}")

include_directories(
  "${CMAKE_CURRENT_BINARY_DIR}/include"
  "${CMAKE_CURRENT_BINARY_DIR}/Enzyme"
  "${ENZYME_ROOT}"
  "${ENZYME_ROOT}/Enzyme"
)

# 1. Build enzyme-tblgen
set(LLVM_LINK_COMPONENTS Demangle Support TableGen)
add_tablegen(enzyme-tblgen Enzyme
  "${ENZYME_ROOT}/tools/enzyme-tblgen/blas-tblgen.cpp"
  "${ENZYME_ROOT}/tools/enzyme-tblgen/datastructures.cpp"
  "${ENZYME_ROOT}/tools/enzyme-tblgen/caching.cpp"
  "${ENZYME_ROOT}/tools/enzyme-tblgen/enzyme-tblgen.cpp"
)
set(Enzyme_TABLEGEN_EXE $<TARGET_FILE:enzyme-tblgen>)

# 2. Generate Enzyme .inc files
file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/Enzyme")
set(LLVM_TARGET_DEFINITIONS "${ENZYME_ROOT}/Enzyme/InstructionDerivatives.td")

function(enzyme_gen_inc ofn flag)
  tablegen(Enzyme "Enzyme/${ofn}" ${flag} EXTRA_INCLUDES "${ENZYME_ROOT}/Enzyme")
  set(TABLEGEN_OUTPUT ${TABLEGEN_OUTPUT} "${CMAKE_CURRENT_BINARY_DIR}/Enzyme/${ofn}" PARENT_SCOPE)
endfunction()

enzyme_gen_inc(BinopDerivatives.inc -gen-binop-derivatives)
enzyme_gen_inc(IntrinsicDerivatives.inc -gen-intr-derivatives)
enzyme_gen_inc(CallDerivatives.inc -gen-call-derivatives)
enzyme_gen_inc(InstructionDerivatives.inc -gen-inst-derivatives)
enzyme_gen_inc(BlasDerivatives.inc -gen-blas-derivatives)
enzyme_gen_inc(BlasAttributor.inc -update-blas-declarations)
enzyme_gen_inc(BlasTA.inc -gen-blas-typeanalysis)
enzyme_gen_inc(BlasDiffUse.inc -gen-blas-diffuseanalysis)
add_public_tablegen_target(EnzymeIncGen)

# 3. Bundle virtual filesystem headers (<enzyme/enzyme> et al.)
file(GLOB_RECURSE ENZYME_PUBLIC_INCLUDES CONFIGURE_DEPENDS "${ENZYME_ROOT}/include/*")
set(BUNDLED_INCLUDES_HEADER "${CMAKE_CURRENT_BINARY_DIR}/Enzyme/Clang/bundled_includes.h")
add_custom_command(
  OUTPUT "${BUNDLED_INCLUDES_HEADER}"
  COMMAND bash "${ENZYME_ROOT}/scripts/bundle-includes.sh"
          "${ENZYME_ROOT}/include/enzyme/enzyme"
          "${BUNDLED_INCLUDES_HEADER}"
  DEPENDS ${ENZYME_PUBLIC_INCLUDES} "${ENZYME_ROOT}/scripts/bundle-includes.sh"
  COMMENT "Bundling Enzyme VFS headers"
  VERBATIM
)
add_custom_target(EnzymeBundledIncludes DEPENDS "${BUNDLED_INCLUDES_HEADER}")

# 4. Build single static library ClangEnzymeStatic with ENZYME_RUNPASS + RTTI matching LLVM
file(GLOB ENZYME_CORE_SRCS CONFIGURE_DEPENDS "${ENZYME_ROOT}/Enzyme/*.cpp")
list(FILTER ENZYME_CORE_SRCS EXCLUDE REGEX "/eopt\\.cpp$")

set(ENZYME_ALL_SRCS
  ${ENZYME_CORE_SRCS}
  "${ENZYME_ROOT}/Enzyme/TypeAnalysis/TypeTree.cpp"
  "${ENZYME_ROOT}/Enzyme/TypeAnalysis/TypeAnalysis.cpp"
  "${ENZYME_ROOT}/Enzyme/TypeAnalysis/TypeAnalysisPrinter.cpp"
  "${ENZYME_ROOT}/Enzyme/Clang/EnzymeClang.cpp"
)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib${LLVM_LIBDIR_SUFFIX}")
add_library(ClangEnzymeStatic STATIC ${ENZYME_ALL_SRCS})
add_dependencies(ClangEnzymeStatic EnzymeIncGen EnzymeBundledIncludes)
if(TARGET intrinsics_gen)
  add_dependencies(ClangEnzymeStatic intrinsics_gen)
endif()

target_compile_definitions(ClangEnzymeStatic PRIVATE ENZYME_RUNPASS)
target_include_directories(ClangEnzymeStatic PRIVATE
  "${CMAKE_CURRENT_BINARY_DIR}/Enzyme/Clang"
)
if(NOT MSVC)
  target_compile_options(ClangEnzymeStatic PRIVATE -Wno-error)
endif()

install(TARGETS ClangEnzymeStatic ARCHIVE DESTINATION "lib${LLVM_LIBDIR_SUFFIX}")
install(DIRECTORY "${ENZYME_ROOT}/include/enzyme" DESTINATION "include")

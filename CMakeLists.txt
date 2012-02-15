CMAKE_MINIMUM_REQUIRED(VERSION 2.6 FATAL_ERROR)
CMAKE_POLICY(VERSION 2.6)
FIND_PACKAGE(Torch REQUIRED)

# use the sparse bundle adjustment from loukatis
set(SBA_SOURCE_DIR ${PROJECT_SOURCE_DIR}/sba-1.6)
add_subdirectory(${SBA_SOURCE_DIR})

# use the sparse bundle adjustment example as main template
set(SBA_EXAMPLE_DIR ${SBA_SOURCE_DIR}/demo)
add_subdirectory(${SBA_EXAMPLE_DIR})

include_directories (${SBA_SOURCE_DIR} ${SBA_EXAMPLE_DIR})

set(src sfm.c)
FILE(GLOB files "${CMAKE_CURRENT_SOURCE_DIR}/examples/*txt")
set(luasrc init.lua ${files})

# call our library sfm b/c sba target is used in sba-1.6
add_torch_package (sfm "${src}" "${luasrc}" "SFM toolkit")
target_link_libraries (sfm luaT TH sbaexample sba)


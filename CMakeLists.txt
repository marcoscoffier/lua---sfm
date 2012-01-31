cmake_minimum_required(VERSION 2.8)

set (CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR})

# infer path for Torch7
string (REGEX REPLACE "(.*)lib/luarocks/rocks.*" "\\1" TORCH_PREFIX "${CMAKE_INSTALL_PREFIX}" )
message (STATUS "Found Torch7, installed in: " ${TORCH_PREFIX})

find_package (Torch REQUIRED)

set (CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# use the sparse bundle adjustment from loukatis
set(SBA_SOURCE_DIR ${PROJECT_SOURCE_DIR}/sba-1.6)
add_subdirectory(${SBA_SOURCE_DIR})

# use the sparse bundle adjustment example as main template
set(SBA_EXAMPLE_DIR ${SBA_SOURCE_DIR}/demo)
add_subdirectory(${SBA_EXAMPLE_DIR})

include_directories (${TORCH_INCLUDE_DIR} ${PROJECT_SOURCE_DIR} 
  ${SBA_SOURCE_DIR} ${SBA_EXAMPLE_DIR})

# call our library sfm b/c sba target is used in sba-1.6
add_library (sfm SHARED sfm.c)

link_directories (${TORCH_LIBRARY_DIR})
set (CMAKE_C_FLAGS "-fPIC")
target_link_libraries (sfm ${TORCH_LIBRARIES} sbaexample sba)

install_files(/lua/sfm init.lua)
install_targets(/lib sfm)


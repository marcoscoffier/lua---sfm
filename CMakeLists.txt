cmake_minimum_required(VERSION 2.8)

set (CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR})

# infer path for Torch7
string (REGEX REPLACE "(.*)lib/luarocks/rocks.*" "\\1" TORCH_PREFIX "${CMAKE_INSTALL_PREFIX}" )
message (STATUS "Found Torch7, installed in: " ${TORCH_PREFIX})

find_package (Torch REQUIRED)

set (CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

add_subdirectory (sba-1.6)
# add_subdirectory (sba-1.g/demo)

include_directories (${TORCH_INCLUDE_DIR} ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/sba-1.6)
add_library (sfm SHARED init.c)
link_directories (${TORCH_LIBRARY_DIR})
set (CMAKE_CXX_FLAGS "-fpic")
target_link_libraries (sfm ${TORCH_LIBRARIES} sba-1.6)

install_files(/lua/sfm init.lua)

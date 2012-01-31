cmake_minimum_required(VERSION 2.8)

set (CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR})

# infer path for Torch7
string (REGEX REPLACE "(.*)lib/luarocks/rocks.*" "\\1" TORCH_PREFIX "${CMAKE_INSTALL_PREFIX}" )
message (STATUS "Found Torch7, installed in: " ${TORCH_PREFIX})

find_package (Torch REQUIRED)

set (CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

add_subdirectory (sba-1.6)
# add_subdirectory (sba-1.g/demo)
set(SBA_SOURCE_DIR ${PROJECT_SOURCE_DIR}/sba-1.6/)
set(SBA_DEMO_DIR ${SBA_SOURCE_DIR}/demo)
include_directories (${TORCH_INCLUDE_DIR} ${PROJECT_SOURCE_DIR} 
  ${SBA_SOURCE_DIR} ${SBA_DEMO_DIR})

add_library (sfm SHARED sba_driver.c)
add_library(sbaexample SHARED ${SBA_DEMO_DIR}/eucsbademo.c 
  ${SBA_DEMO_DIR}/imgproj.c 
  ${SBA_DEMO_DIR}/readparams.c 
  ${SBA_DEMO_DIR}/eucsbademo.h 
  ${SBA_DEMO_DIR}/readparams.h)

link_directories (${TORCH_LIBRARY_DIR})
set (CMAKE_C_FLAGS "-fPIC")
target_link_libraries (sfm ${TORCH_LIBRARIES} sbaexample sba)

install_files(/lua/sfm init.lua)
install_targets(/lib sfm)


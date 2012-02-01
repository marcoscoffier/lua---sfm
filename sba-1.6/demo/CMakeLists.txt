# CMake file for sba's demo program

INCLUDE_DIRECTORIES(${SBA_SOURCE_DIR})
LINK_DIRECTORIES(${SBA_SOURCE_DIR} ${LAPACKBLAS_DIR})

# this is the demo code from which we are borrowing many functions.
add_library(sbaexample 
  eucsbademo.c 
  imgproj.c 
  readparams.c 
  eucsbademo.h 
  readparams.h)

# for debugging
add_executable(eucsbademo 
  eucsbademo.c 
  imgproj.c 
  readparams.c 
  eucsbademo.h 
  readparams.h)

set (CMAKE_C_FLAGS "-fpic -g -DUNIXIO")

# libraries the demo depends on
IF(HAVE_F2C)
  TARGET_LINK_LIBRARIES(sbaexample sba ${LAPACK_LIB} ${BLAS_LIB} ${F2C_LIB})
  TARGET_LINK_LIBRARIES(eucsbademo sba ${LAPACK_LIB} ${BLAS_LIB} ${F2C_LIB})
ELSE(HAVE_F2C)
  TARGET_LINK_LIBRARIES(sbaexample sba ${LAPACK_LIB} ${BLAS_LIB} ${F77_LIB} ${I77_LIB})
  TARGET_LINK_LIBRARIES(eucsbademo sba ${LAPACK_LIB} ${BLAS_LIB} ${F77_LIB} ${I77_LIB})
ENDIF(HAVE_F2C)

# make sure that the library is built before the demo
ADD_DEPENDENCIES(sbaexample sba)

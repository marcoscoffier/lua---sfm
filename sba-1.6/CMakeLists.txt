# sba CMake file; 
#  see http://www.cmake.org and 
#  http://www.insightsoftwareconsortium.org/wiki/index.php/CMake_Tutorial

# f2c is sometimes equivalent to libF77 & libI77; in that case, set
# HAVE_F2C to 0
SET(HAVE_F2C 1 CACHE BOOL "Do we have f2c or F77/I77?" )

# the directory where the lapack/blas/f2c libraries reside
SET(LAPACKBLAS_DIR /usr/lib CACHE PATH "Path to lapack/blas libraries")

# actual names for the lapack/blas/f2c libraries
SET(LAPACK_LIB lapack CACHE STRING "The name of the lapack library")
SET(BLAS_LIB blas CACHE STRING "The name of the blas library")
IF(HAVE_F2C)
  SET(F2C_LIB f2c CACHE STRING "The name of the f2c library")
ELSE(HAVE_F2C)
  SET(F77_LIB libF77 CACHE STRING "The name of the F77 library")
  SET(I77_LIB libI77 CACHE STRING "The name of the I77 library")
ENDIF(HAVE_F2C)

############### NO CHANGES BEYOND THIS POINT ##########################

INCLUDE_DIRECTORIES(${SBA_SOURCE_DIR})
# sba library source files
ADD_LIBRARY(sba 
  sba_levmar.c sba_levmar_wrap.c sba_lapack.c sba_crsm.c sba_chkjac.c
  sba.h sba_chkjac.h compiler.h
)

set (CMAKE_C_FLAGS "-fpic -g -DUNIXIO")

IF(HAVE_F2C)
  TARGET_LINK_LIBRARIES(sba ${LAPACK_LIB} ${BLAS_LIB} ${F2C_LIB})
ELSE(HAVE_F2C)
  TARGET_LINK_LIBRARIES(sba ${LAPACK_LIB} ${BLAS_LIB} ${F77_LIB} ${I77_LIB})
ENDIF(HAVE_F2C)

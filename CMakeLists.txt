cmake_minimum_required(VERSION 3.10)
project(cuda_code)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -DMY_DEF=1")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DMY_DEF=1")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DMY_DEF=1")

find_package(CUDA REQUIRED)

# include directories
include_directories(
  include

  ${catkin_INCLUDE_DIRS}
  ${CUDA_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
)

set(srcs
  src/func/addition.cu
  src/func/addup.cu
  src/func/matrix_cuda.cu
  src/func/reduction.cu
  src/func/cuda_eigen.cu

  src/cuda_code_node.cpp
)

set(incs
  include/common/cuda_base.h
  include/func/addition.h
  include/func/addup.h
  include/func/matrix_cuda.h
  include/func/cuda_eigen.h

  include/common/myutility.h
)

cuda_add_library(cuda_code_lib SHARED ${srcs} ${incs})
cuda_add_executable(cuda_code_node ${srcs})
target_link_libraries(cuda_code_node
  ${CUDA_LIBRARIES}
  ${CUDA_CUBLAS_LIBRARIES}
  ${CUDA_curand_LIBRARY}
  ${CUDA_cusolver_LIBRARY}
  ${CUDA_cudart_LIBRARY}

  ${YAML_CPP_LIBRARIES}
  glog
  gflags
)

# ---------------------------------------------------------------------------------------
# Install
# ---------------------------------------------------------------------------------------
install(TARGETS cuda_code_node cuda_code_lib
  DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/install
)

install(DIRECTORY include/cuda_code/ config
  DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/install
  FILES_MATCHING PATTERN "*.h"
)

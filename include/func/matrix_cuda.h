
#ifndef MATRIX_CUDA_H
#define MATRIX_CUDA_H

#include "common/cuda_base.h"

#include <iostream>

struct CudaMatrix
{
  int width;
  int height;
  float *elements;
};

/*check if the compiler is of C++*/
#ifdef __cplusplus
extern "C" bool addition(float *x, float *y, float *z, int n);

__device__ float getElement(CudaMatrix *A, int row, int col);

__device__ void setElement(CudaMatrix *A, int row, int col, float value);

__global__ void matMulKernel(CudaMatrix *A, CudaMatrix *B, CudaMatrix *C);

__global__ void matAddKernel(CudaMatrix *A, CudaMatrix *B, CudaMatrix *C);
#endif

class Matrix
{
 public:
  Matrix( );
  ~Matrix( );
  int width  = 1 << 10;
  int height = 1 << 10;
  int nBytes;
  CudaMatrix A, B, C;

  void matrixMul(CudaMatrix &a, CudaMatrix &b, CudaMatrix &c);
  void matrixAdd(CudaMatrix &a, CudaMatrix &b, CudaMatrix &c);
};

#endif
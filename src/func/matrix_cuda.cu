
#include "func/matrix_cuda.h"


// 获取矩阵A的(row, col)元素
__device__ float getElement(CudaMatrix *A, int row, int col)
{
  return A->elements[row * A->width + col];
}

// 为矩阵A的(row, col)元素赋值
__device__ void setElement(CudaMatrix *A, int row, int col, float value)
{
  A->elements[row * A->width + col] = value;
}

// 矩阵相乘kernel，2-D，每个线程计算一个元素
__global__ void matMulKernel(CudaMatrix *A, CudaMatrix *B, CudaMatrix *C)
{
  float Cvalue = 0.0;
  int row      = threadIdx.y + blockIdx.y * blockDim.y;
  int col      = threadIdx.x + blockIdx.x * blockDim.x;
  for (int i = 0; i < A->width; ++i)
  {
    Cvalue += getElement(A, row, i) * getElement(B, i, col);
  }
  setElement(C, row, col, Cvalue);
}
__global__ void matAddKernel(CudaMatrix *A, CudaMatrix *B, CudaMatrix *C)
{
  int row      = threadIdx.y + blockIdx.y * blockDim.y;
  int col      = threadIdx.x + blockIdx.x * blockDim.x;
  float Cvalue = getElement(A, row, col) + getElement(B, row, col);
  setElement(C, row, col, Cvalue);
}

Matrix::Matrix( )
{
  nBytes     = width * height * sizeof(float);
  A.elements = (float *)malloc(nBytes);
  B.elements = (float *)malloc(nBytes);
  C.elements = (float *)malloc(nBytes);

  // 初始化数据
  A.height = height;
  A.width  = width;
  B.height = height;
  B.width  = width;
  C.height = height;
  C.width  = width;
  for (int i = 0; i < width * height; ++i)
  {
    A.elements[i] = 1.0;
    B.elements[i] = 2.0;
    C.elements[i] = 10.0;
  }
}

Matrix::~Matrix( )
{
  free(A.elements);
  free(B.elements);
  free(C.elements);
}

void Matrix::matrixMul(CudaMatrix &a, CudaMatrix &b, CudaMatrix &c)
{
  CudaMatrix *A, *B, *C;

  // "Managed":
  cudaMallocManaged((void **)&A, sizeof(CudaMatrix));
  cudaMallocManaged((void **)&B, sizeof(CudaMatrix));
  cudaMallocManaged((void **)&C, sizeof(CudaMatrix));
  nBytes = width * height * sizeof(float);
  cudaMallocManaged((void **)&A->elements, nBytes);
  cudaMallocManaged((void **)&B->elements, nBytes);
  cudaMallocManaged((void **)&C->elements, nBytes);

  // (dst目标区域, src数据源, count复制的字节数, 复制的方向)
  cudaMemcpy((void *)A->elements, (void *)a.elements, nBytes, cudaMemcpyHostToDevice);
  cudaMemcpy((void *)B->elements, (void *)b.elements, nBytes, cudaMemcpyHostToDevice);

  // assign or cp from host
  cudaMemcpy((void *)&(A->width), (void *)&(a.width), sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy((void *)&(B->width), (void *)&(b.width), sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy((void *)&(A->height), (void *)&(a.height), sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy((void *)&(B->height), (void *)&(b.height), sizeof(int), cudaMemcpyHostToDevice);
  C->height = height;
  C->width  = width;

  // 定义kernel的执行配置
  dim3 blockSize(32, 32);
  dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                (height + blockSize.y - 1) / blockSize.y);

  // 执行kernel
  matMulKernel<<<gridSize, blockSize>>>(A, B, C);

  // 同步device 保证结果能正确访问
  cudaDeviceSynchronize( );

  // std::cout << "--- " << C->elements[] << std::endl;
  cudaMemcpy(c.elements, C->elements, nBytes, cudaMemcpyDeviceToHost);

  // 检查执行结果
  float maxError = 0.0;
  for (int i = 0; i < width * height; ++i)
  {
    maxError = fmax(maxError, fabs(c.elements[i] - 2 * width));
  }
  std::cout << "mat mul 最大误差: " << maxError << std::endl
            << std::endl
            << std::endl;

  cudaFree(A);
  cudaFree(B);
  cudaFree(C);
}

void Matrix::matrixAdd(CudaMatrix &a, CudaMatrix &b, CudaMatrix &c)
{
  CudaMatrix *A, *B, *C;

  // "Managed":
  cudaMallocManaged((void **)&A, sizeof(CudaMatrix));
  cudaMallocManaged((void **)&B, sizeof(CudaMatrix));
  cudaMallocManaged((void **)&C, sizeof(CudaMatrix));
  nBytes = width * height * sizeof(float);
  cudaMallocManaged((void **)&A->elements, nBytes);
  cudaMallocManaged((void **)&B->elements, nBytes);
  cudaMallocManaged((void **)&C->elements, nBytes);

  // (dst目标区域, src数据源, count复制的字节数, 复制的方向)
  cudaMemcpy((void *)A->elements, (void *)a.elements, nBytes, cudaMemcpyHostToDevice);
  cudaMemcpy((void *)B->elements, (void *)b.elements, nBytes, cudaMemcpyHostToDevice);

  // assign or cp from host
  cudaMemcpy((void *)&(A->width), (void *)&(a.width), sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy((void *)&(B->width), (void *)&(b.width), sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy((void *)&(A->height), (void *)&(a.height), sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy((void *)&(B->height), (void *)&(b.height), sizeof(int), cudaMemcpyHostToDevice);
  C->height = height;
  C->width  = width;

  // 定义kernel的执行配置
  dim3 blockSize(32, 32); // 当块大小超过硬件的极限，并没有报错，而是返回错误值
  dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                (height + blockSize.y - 1) / blockSize.y);

  // 执行kernel
  matAddKernel<<<gridSize, blockSize>>>(A, B, C);

  // 同步device 保证结果能正确访问
  cudaDeviceSynchronize( );

  cudaMemcpy(c.elements, C->elements, nBytes, cudaMemcpyDeviceToHost);

  // 检查执行结果
  float maxError = 0.0;
  for (int i = 0; i < width * height; ++i)
  {
    maxError = fmax(maxError, fabs(c.elements[i] - 3.0));
  }
  std::cout << "mat add 最大误差: " << maxError << std::endl
            << std::endl
            << std::endl;

  cudaFree(A);
  cudaFree(B);
  cudaFree(C);
}
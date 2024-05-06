#ifndef CUDA_EIGEN
#define CUDA_EIGEN

#include <cusolverDn.h>
#include <cuda_runtime.h>

#include "common/cuda_base.h"

class CudaEigen
{
 public:
  CudaEigen( )
  {
    printf(" CudaEigen \n\n");
    cusolverDnHandle_t cusolverH = NULL;
    cudaStream_t stream          = NULL;

    cusolverDnCreate(&cusolverH);
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    cusolverDnSetStream(cusolverH, stream);

    // 假设协方差矩阵的大小为 N x N
    const int N = 5;       // 协方差矩阵的维度
    float *h_A  = nullptr; // 主机上的协方差矩阵
    float *d_A  = nullptr; // 设备上的协方差矩阵
    cudaMalloc((void **)&d_A, sizeof(float) * N * N);

    // 将协方差矩阵复制到设备内存
    cudaMemcpy(d_A, h_A, sizeof(float) * N * N, cudaMemcpyHostToDevice);


    // 计算特征值和特征向量的API调用会根据矩阵类型而变化
    // 例如，对于实数对称矩阵:
    int lwork = 0;
    cusolverDnSsyevd_bufferSize(cusolverH, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER,
                                N, d_A, N, NULL, &lwork); // 计算缓冲区大小
                                                          // 为特征值和特征向量分配内存
    float *d_Work = NULL;
    float *d_W    = NULL; // 存储特征值
    cudaMalloc((void **)&d_Work, sizeof(float) * lwork);
    cudaMalloc((void **)&d_W, sizeof(float) * N);
    int *devInfo = NULL;
    cudaMalloc((void **)&devInfo, sizeof(int));
    cusolverDnSsyevd(cusolverH, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER,
                     N, d_A, N, d_W, d_Work, lwork, devInfo);

    // d_A 现在包含特征向量
    // d_W 包含特征值

    cudaFree(d_Work);
    cudaFree(devInfo);
    cusolverDnDestroy(cusolverH);
    cudaStreamDestroy(stream);
  }
};


#endif
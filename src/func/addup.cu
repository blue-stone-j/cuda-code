#include "func/addup.h"

__global__ void addup(float *x, float *y, int n)
{
  // 获取全局索引
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  // 步长
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride)
  {
    int offset = int(x[i]);
    if (offset > 9)
    {
      printf("%d, %d\n", i, offset);
    }
    /* Atomic operations in CUDA ensure that concurrent operations by multiple threads on the same memory location
       are executed serially, preventing race conditions.
       >> type atomicAdd(type* address, type val);
       address is the pointer to the memory location where the addition is to be performed.
       val is the value to be added to the variable pointed to by address.
       returns the old value that was stored at address before the addition.
     */
    atomicAdd(y + offset, x[i]);
  }
}

extern "C" bool addups(float *x, float *y, float *z, int N)
{
  float *d_x, *d_y, *d_z;

  int nBytes = N * sizeof(float);

  // 在device上申请一定字节大小的显存(device内存)，并进行数据初始化；通常为同步操作
  // 动态分配
  cudaMalloc((void **)&d_x, nBytes);
  cudaMalloc((void **)&d_y, 10 * sizeof(float));
  cudaMalloc((void **)&d_z, nBytes);

  // 初始化内存
  //  cudaError_t cudaMemset(void *devPtr, int value, size_t count)
  cudaMemset(d_y, 0, 10 * sizeof(float));
  cudaMemset(d_z, 0, nBytes);

  // 从host将数据拷贝到device上(host和device之间数据通信);默认为同步操作
  // (dst目标区域, src数据源, count复制的字节数, 复制的方向)
  cudaMemcpy((void *)d_x, (void *)x, nBytes, cudaMemcpyHostToDevice);

  // 调用CUDA的核函数在device上完成指定的运算
  // 用<<<grid, block>>>来指定kernel要执行的线程数量
  // 一个核函数只能有一个grid，一个grid可以有很多个块，每个块可以有很多的线程
  // 不同块内线程不能相互影响！他们是物理隔离的！
  // 一个网格通常被分成二维的块，而每个块常被分成三维的线程
  dim3 blockSize(256);
  dim3 gridSize((N + blockSize.x - 1) / blockSize.x);
  addup<<<gridSize, blockSize>>>(d_x, d_y, N);
  cudaDeviceSynchronize( );

  // 将device上的运算结果拷贝到host上,
  cudaMemcpy(y, d_y, 10 * sizeof(float), cudaMemcpyDeviceToHost);

  // 释放device上分配的内存
  cudaFree(d_x);
  cudaFree(d_y);
  cudaFree(d_z);
  return true;
}

Addup::Addup( )
{
  std::cout << "Addup" << std::endl;
  size_t N   = 1 << 15;
  int nBytes = N * sizeof(float);
  // 申请host内存
  float *x, *y, *z;
  x = (float *)malloc(nBytes);
  y = (float *)calloc(0, 10 * sizeof(float));
  z = (float *)malloc(nBytes);

  // 初始化数据
  for (size_t i = 0; i < N; ++i)
  {
    x[i] = float(i % 100) / 10; // from 0 to a/b
  }

  addups(x, y, z, N);

  for (int i = 1; i < 10; i++)
  {
    std::cout << y[i] - y[i - 1] << ", ";
  }
  std::cout << "check" << std::endl;

  {
    float *sum;
    sum = (float *)calloc(0, 10 * sizeof(float));

    for (size_t i = 0; i < N; i++)
    {
      if (int(x[i]) > 9)
      {
        printf("%lu, %d\n", i, int(x[i]));
      }
      sum[int(x[i])] += x[i];
    }

    for (int i = 1; i < 10; i++)
    {
      std::cout << sum[i] - sum[i - 1] << ", ";
    }
    std::cout << std::endl
              << std::endl;

    free(sum);
  }

  // 释放host内存
  free(x);
  free(y);
  free(z);
}
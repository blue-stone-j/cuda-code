
#include "func/addition.h"

/** __global__
 * 在device上执行，从host中调用（一些特定的GPU也可以从device上调用），返回类型必须是void，不支持可变参数参数，不能成为类成员函数。
 * 注意用__global__定义的kernel是异步的，这意味着host不会等待kernel函数执行完，而是对kernel函数的调用结束就执行下一步。
 */
/**
 * __device__
 * 在device上执行，单仅可以从device中调用，不可以和__global__同时用。不支持可变数量的参数。不支持静态变量。
 *
 */
/**__host__
 * 在host上执行，仅可以从host上调用，一般省略不写，不可以和__global__同时用，但可和__device__，此时函数会在device和host都编译
 */

/**
 * 带有__global__修饰符的函数称为”核函数“，它负责处理GPU内存里的数据，是并行计算发生的地方。
 * bool addition(int a, int b, int *c)充当了CPU和GPU之间数据传输的角色。也就是Host和Device之间的数据传输。
 */

/**一个kernel所启动的所有线程称为一个网格（grid）
 *
 */

__global__ void add(float *x, float *y, float *z, int n)
{
  // 获取全局索引
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  // 步长
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride)
  {
    z[i] = x[i] + y[i];
  }
}

extern "C" bool addition(float *x, float *y, float *z, int N)
{
  float *d_x, *d_y, *d_z;

  int nBytes = N * sizeof(float);

  // 在device上申请一定字节大小的显存(device内存)，并进行数据初始化；通常为同步操作
  // 动态分配
  cudaMalloc((void **)&d_x, nBytes);
  cudaMalloc((void **)&d_y, nBytes);
  cudaMalloc((void **)&d_z, nBytes);

  // 初始化内存
  //  cudaError_t cudaMemset(void *devPtr, int value, size_t count)

  // 从host将数据拷贝到device上(host和device之间数据通信);默认为同步操作
  // (dst目标区域, src数据源, count复制的字节数, 复制的方向)
  cudaMemcpy((void *)d_x, (void *)x, nBytes, cudaMemcpyHostToDevice);
  cudaMemcpy((void *)d_y, (void *)y, nBytes, cudaMemcpyHostToDevice);



  // 调用CUDA的核函数在device上完成指定的运算
  // 用<<<grid, block>>>来指定kernel要执行的线程数量
  // 一个核函数只能有一个grid，一个grid可以有很多个块，每个块可以有很多的线程
  // 不同块内线程不能相互影响！他们是物理隔离的！
  // 一个网格通常被分成二维的块，而每个块常被分成三维的线程
  dim3 blockSize(256);
  dim3 gridSize((N + blockSize.x - 1) / blockSize.x);
  add<<<gridSize, blockSize>>>(d_x, d_y, d_z, N);

  cudaDeviceSynchronize( );
  // 将device上的运算结果拷贝到host上,
  cudaMemcpy(z, d_z, nBytes, cudaMemcpyDeviceToHost);

  // 释放device上分配的内存
  cudaFree(d_x);
  cudaFree(d_y);
  cudaFree(d_z);
  return true;
}
NumAdd::NumAdd( )
{
  int N      = 1 << 20;
  int nBytes = N * sizeof(float);
  // 申请host内存
  float *x, *y, *z;
  x = (float *)malloc(nBytes);
  y = (float *)malloc(nBytes);
  z = (float *)malloc(nBytes);

  // 初始化数据
  for (int i = 0; i < N; ++i)
  {
    x[i] = 10.0;
    y[i] = 20.0;
  }

  addition(x, y, z, N);
  float maxError = 0.0;
  for (int i = 0; i < N; i++)
  {
    maxError = fmax(maxError, fabs(z[i] - 30.0));
  }
  std::cout << "val add 最大误差: " << maxError << std::endl
            << std::endl
            << std::endl;


  // 释放host内存
  free(x);
  free(y);
  free(z);
}

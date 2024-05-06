#include "func/reduction.h"


// 当一个线程束中所有的线程都执行if或者，都执行else时，不存在性能下降；只有当线程束内有分歧产生分支的时候，性能才会急剧下降。
__global__ void reduceNeighbored(int *g_idata, int *g_odata, unsigned int n)
{
  // first method
  // // set thread ID
  // unsigned int tid = threadIdx.x;
  // // boundary check
  // if (tid >= n)
  // {
  //   return;
  // }
  // // convert global data pointer to the local point of this block
  // int *idata = g_idata + blockIdx.x * blockDim.x;
  // // in-place reduction in global memory
  // for (int stride = 1; stride < blockDim.x; stride *= 2)
  // {
  //   if ((tid % (2 * stride)) == 0)
  //   {
  //     idata[tid] += idata[tid + stride];
  //   }
  //   // synchronize within block
  //   // 可以到达__syncthreads()的线程需要其他可以到达该点的线程，而不是等待块内所有其他线程。
  //   __syncthreads( );
  // }
  // // 在某时刻T，SM上只执行一个线程束
  // // write result for this block to global mem
  // if (tid == 0)
  // {
  //   g_odata[blockIdx.x] = idata[0];
  // }

  // second method
  // unsigned int tid = threadIdx.x;
  // unsigned idx     = blockIdx.x * blockDim.x + threadIdx.x;
  // // convert global data pointer to the local point of this block
  // int *idata = g_idata + blockIdx.x * blockDim.x;
  // if (idx > n)
  // {
  //   return;
  // }
  // // in-place reduction in global memory
  // for (int stride = 1; stride < blockDim.x; stride *= 2)
  // {
  //   // convert tid into local array index
  //   int index = 2 * stride * tid;
  //   if (index < blockDim.x)
  //   {
  //     idata[index] += idata[index + stride];
  //   }
  //   __syncthreads( );
  // }
  // // write result for this block to global men
  // if (tid == 0)
  // {
  //   g_odata[blockIdx.x] = idata[0];
  // }

  // third method
  unsigned int tid = threadIdx.x;
  unsigned idx     = blockIdx.x * blockDim.x + threadIdx.x;
  // convert global data pointer to the local point of this block
  int *idata = g_idata + blockIdx.x * blockDim.x;
  if (idx >= n)
  {
    return;
  }
  // in-place reduction in global memory
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
  {
    if (tid < stride)
    {
      idata[tid] += idata[tid + stride];
    }
    __syncthreads( );
  }
  // write result for this block to global men
  if (tid == 0)
  {
    g_odata[blockIdx.x] = idata[0];
  }
}

extern "C" bool adds(int *idata_host, int *odata_host, int size, int blocksize)
{
  // execution configuration
  dim3 block(blocksize, 1);
  dim3 grid((size - 1) / block.x + 1, 1);
  printf("grid %d block %d \n", grid.x, block.x);

  // device memory
  int *idata_dev = NULL;
  int *odata_dev = NULL;
  int bytes      = size * sizeof(int);

  cudaMalloc((void **)&idata_dev, bytes);
  cudaMalloc((void **)&odata_dev, grid.x * sizeof(int));

  // kernel reduceNeighbored
  cudaMemcpy(idata_dev, idata_host, bytes, cudaMemcpyHostToDevice);

  reduceNeighbored<<<grid, block>>>(idata_dev, odata_dev, size);

  // 让主机等待设备端执行
  cudaDeviceSynchronize( );
  cudaMemcpy(odata_host, odata_dev, grid.x * sizeof(int), cudaMemcpyDeviceToHost);
  cudaFree(idata_dev);
  cudaFree(odata_dev);

  return true;
}


Reduction::Reduction( )
{
  printf(" Reduction \n");
  printf("array size %d  \n", size);
  int blocksize = 1024;


  // allocate host memory
  bytes      = size * sizeof(int);
  idata_host = (int *)malloc(bytes);
  odata_host = (int *)malloc(((size - 1) / blocksize + 1) * sizeof(int));
  tmp        = (int *)malloc(bytes);

  // initialize the array
  initialData_int(idata_host, size);

  memcpy(tmp, idata_host, bytes);
  // std::cout << "here" << std::endl;
  adds(idata_host, odata_host, size, blocksize);

  gpu_sum = 0;
  for (int i = 0; i < (size - 1) / blocksize + 1; i++)
  {
    gpu_sum += odata_host[i];
  }


  printf("gpu sum:%d \n\n\n", gpu_sum);

  // free host memory

  free(idata_host);
  free(odata_host);

  // reset device
  /* 这句话包含了隐式同步，GPU和CPU执行程序是异步的，核函数调用后成立刻会到主机线程继续，而不管GPU端核函数是否执行完毕，所以上面的程序就是
     GPU刚开始执行，CPU已经退出程序了，所以我们要等GPU执行完了，再退出主机线程。
   */
  cudaDeviceReset( );
}

Reduction::~Reduction( )
{}

void Reduction::initialData_int(int *ip, int size)
{
  time_t t;
  srand((unsigned)time(&t));
  for (int i = 0; i < size; i++)
  {
    ip[i] = 2;

    // ip[i] = int(rand( ) & 0xff);
  }
}

#ifndef CUDABASE_H
#define CUDABASE_H

#include <cstdio>

#define CHECK(call)                                                    \
  {                                                                    \
    const cudaError_t error = call;                                    \
    if (error != cudaSuccess)                                          \
    {                                                                  \
      printf("ERROR: %s:%d,", __FILE__, __LINE__);                     \
      printf("code:%d,reason:%s\n", error, cudaGetErrorString(error)); \
      exit(1);                                                         \
    }                                                                  \
  }


// #include "driver_types.h"
#include "cuda_runtime_api.h"


// nvprof: 运行时间的命令行分析工具
// CUDA工具包中提供一个叫做UCDA占用率计算器的电子表格，填上相关数据可以帮你自动计算网格参数：

#endif
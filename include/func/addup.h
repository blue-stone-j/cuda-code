#ifndef AADDUP_H
#define AADDUP_H

#include "common/cuda_base.h"

#include <iostream>

class Addup
{
 public:
  Addup( );
};

/**
 * addup.h
 * 修饰符extern "C"是CUDA和C++混合编程时必须的
 */

/*check if the compiler is of C++*/
#ifdef __cplusplus
extern "C" bool addups(float *x, float *y, float *z, int N);

#endif

#endif
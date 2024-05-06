

#ifndef INCLUDES_ADDITION_H_
#define INCLUDES_ADDITION_H_

#include "common/cuda_base.h"

#include <iostream>

class NumAdd
{
 public:
  NumAdd( );
};

/**
 * addition.h
 * 修饰符extern "C"是CUDA和C++混合编程时必须的
 */

/*check if the compiler is of C++*/
#ifdef __cplusplus
extern "C" bool addition(float *x, float *y, float *z, int n);

#endif



#endif /* INCLUDES_ADDITION_H_ */
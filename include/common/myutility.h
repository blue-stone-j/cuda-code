#ifndef MYUTILITY_H
#define MYUTILITY_H

#include <deque>
#include <iostream>
#include <thread>
#include <vector>

#include <eigen3/Eigen/Geometry>

#include "common/read_params.h"

#include <sys/time.h> // only for linux



/******************/
enum class SensorType
{
  OUSTER,
  HESAI
};

struct LidarParam
{
  SensorType sensor;
  int id;
  std::string frame;
  std::string topic;
  int vertical;
  int horizon;
  Eigen::Affine3d extrans;
  Eigen::Affine3d pose_last = Eigen::Affine3d::Identity( );
};

double cpuSecond( )
{
  struct timeval tp;
  gettimeofday(&tp, NULL); // <sys/time.h>, 从1970年1月1日0点以来到现在的秒数
  return ((double)tp.tv_sec + (double)tp.tv_usec * 1e-6);
}

/******************/

#endif
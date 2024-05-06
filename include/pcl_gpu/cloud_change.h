#ifndef CLOUDCHANGE_H
#define CLOUDCHANGE_H

#include <iostream>

#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/features/feature.h>
#include <pcl/search/impl/search.hpp>
#include <pcl/features/normal_3d.h>
#include <pcl/registration/gicp.h>

#include <pcl/gpu/features/features.hpp>

#include "common/cuda_base.h"

/*check if the compiler is of C++*/
#ifdef __cplusplus
extern "C" bool cloud2GPU(pcl::gpu::DeviceArray<pcl::PointXYZRGB> &cloud_device);

#endif

class CloudChange
{
 public:
  CloudChange( );
  ~CloudChange( );
};

#endif
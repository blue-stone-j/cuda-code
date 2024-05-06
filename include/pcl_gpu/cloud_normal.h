
#ifndef CLOUDNORMAL_H
#define CLOUDNORMAL_H

#include <iostream>
#include <fstream>
#include <string>
#include <iostream>
#include <cstdlib>
#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/principal_curvatures.h>
#include <pcl/gpu/features/features.hpp>

class CloudNormal
{
 public:
  CloudNormal( );
  ~CloudNormal( );
  bool getModelCurvatures(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, int k);
};


#endif

#include "pcl_gpu/cloud_change.h"

__global__ void change_points(pcl::gpu::PtrSz<pcl::PointXYZRGB> cloud_device)
{
  cloud_device[0].x += 1;
  pcl::PointXYZRGB q = cloud_device.data[0];
  printf("x=%f, y=%f, z=%f, r=%d, g=%d, b=%d \n\n\n", q.x, q.y, q.z, q.r, q.g, q.b);
}



extern "C" bool cloud2GPU(pcl::gpu::DeviceArray<pcl::PointXYZRGB> &cloud_device)
{
  change_points<<<1, 1>>>(cloud_device);
  return true;
}


CloudChange::CloudChange( )
{
  pcl::PointCloud<pcl::PointXYZRGB> cloud;
  pcl::gpu::DeviceArray<pcl::PointXYZRGB> cloud_device;

  cloud.width    = 1;
  cloud.height   = 1;
  cloud.is_dense = false;
  cloud.points.resize(cloud.width * cloud.height);

  std::vector<float> point_val;

  for (size_t i = 0; i < 3 * cloud.points.size( ); ++i)
  {
    point_val.push_back(1024 * rand( ) / (RAND_MAX + 1.0f));
  }

  for (size_t i = 0; i < cloud.points.size( ); ++i)
  {
    cloud.points[i].x = point_val[3 * i];
    cloud.points[i].y = point_val[3 * i + 1];
    cloud.points[i].z = point_val[3 * i + 2];
  }

  std::cout << "cloud.points=" << cloud.points[0] << std::endl;

  cloud_device.upload(cloud.points);

  cloud2GPU(cloud_device);

  cloud_device.download(cloud.points);

  std::cout << "cloud.points=" << cloud.points[0] << std::endl;
}

CloudChange::~CloudChange( )
{}
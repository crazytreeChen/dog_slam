#include "lidar_3d_relocalizer/pcd_map_loader.hpp"
#include <pcl/io/pcd_io.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/common/centroid.h>
#include <pcl/common/transforms.h>
#include <iostream>

namespace lidar_3d_relocalizer
{

bool PcdMapLoader::load(const std::string& file_path, float voxel_size)
{
  map_raw_.reset(new PointCloudT());
  map_ds_.reset(new PointCloudT());

  // PCL 原生支持 ascii / binary / binary_compressed
  int ret = pcl::io::loadPCDFile<PointT>(file_path, *map_raw_);
  if (ret != 0 || map_raw_->empty())
  {
    std::cerr << "[PcdMapLoader] Failed to load PCD: " << file_path << std::endl;
    map_raw_.reset();
    return false;
  }

  std::cout << "[PcdMapLoader] Loaded map with " << map_raw_->size() << " points from "
            << file_path << std::endl;

  computeMapCenter();

  // 降采样
  if (voxel_size > 1e-4f)
  {
    pcl::VoxelGrid<PointT> vg;
    vg.setInputCloud(map_raw_);
    vg.setLeafSize(voxel_size, voxel_size, voxel_size);
    vg.filter(*map_ds_);
    std::cout << "[PcdMapLoader] Downsampled map: " << map_ds_->size()
              << " points (voxel=" << voxel_size << "m)" << std::endl;
  }
  else
  {
    map_ds_ = map_raw_;
  }

  return true;
}

bool PcdMapLoader::transformMap(
    const Eigen::Isometry3f& T_target_source)
{
  if (!map_ds_ || map_ds_->empty()) return false;

  Eigen::Matrix4f T = T_target_source.matrix();

  PointCloudT transformed;
  pcl::transformPointCloud(*map_ds_, transformed, T);
  *map_ds_ = transformed;

  std::cout << "[PcdMapLoader] Transformed map (" << T(0,3) << ", "
            << T(1,3) << ", " << T(2,3) << ")" << std::endl;
  return true;
}

void PcdMapLoader::computeMapCenter()
{
  if (!map_raw_ || map_raw_->empty()) return;
  Eigen::Vector4f centroid;
  pcl::compute3DCentroid(*map_raw_, centroid);
  map_center_ = Eigen::Vector3f(centroid[0], centroid[1], centroid[2]);
}

}  // namespace lidar_3d_relocalizer

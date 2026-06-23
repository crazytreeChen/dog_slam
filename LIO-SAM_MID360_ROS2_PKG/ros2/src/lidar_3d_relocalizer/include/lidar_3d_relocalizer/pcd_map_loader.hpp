#pragma once
/**
 * PCD 地图加载器
 * 启动时加载 PCD 先验地图到内存，支持 binary_compressed 格式
 * 地图始终保持在原始 map 系，不做坐标变换
 * KISS-Matcher 直接做 odom→map 全局匹配
 */

#include <string>
#include <memory>
#include <Eigen/Geometry>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/filters/voxel_grid.h>

namespace lidar_3d_relocalizer
{

using PointT = pcl::PointXYZI;  // x, y, z, intensity — 与 MID360 PCD 格式一致
using PointCloudT = pcl::PointCloud<PointT>;

class PcdMapLoader
{
public:
  PcdMapLoader() = default;
  ~PcdMapLoader() = default;

  /**
   * @brief 加载 PCD 文件（支持 ascii / binary / binary_compressed）
   * @param file_path PCD 文件路径
   * @param voxel_size 降采样分辨率（米），<=0 表示不降采样
   * @return true 加载成功
   */
  bool load(const std::string& file_path, float voxel_size = 0.0f);

  /**
   * @brief 将 PCD 地图变换到指定坐标系
   * @param T_target_source source→target 的变换
   * @return true 变换成功
   */
  bool transformMap(const Eigen::Isometry3f& T_target_source);

  /// 获取降采样后的地图点云（用于全局匹配）
  typename PointCloudT::ConstPtr getDownsampledMap() const { return map_ds_; }

  /// 获取原始地图点云
  typename PointCloudT::ConstPtr getRawMap() const { return map_raw_; }

  /// 获取地图中心点（用于调试）
  Eigen::Vector3f getMapCenter() const { return map_center_; }

private:
  typename PointCloudT::Ptr map_raw_{nullptr};   // 原始地图
  typename PointCloudT::Ptr map_ds_{nullptr};     // 降采样地图（用于匹配）
  Eigen::Vector3f map_center_{Eigen::Vector3f::Zero()};

  void computeMapCenter();
};

}  // namespace lidar_3d_relocalizer

#pragma once
/**
 * 重定位核心：KISS-Matcher 全局匹配 + GICP 精细配准
 */

#include <memory>
#include <optional>
#include <Eigen/Geometry>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include "pcd_map_loader.hpp"
#include <kiss_matcher/core/kiss_matcher/KISSMatcher.hpp>

namespace lidar_3d_relocalizer
{

using PointT = pcl::PointXYZI;
using PointCloudT = pcl::PointCloud<PointT>;

struct RelocalizationResult
{
  Eigen::Isometry3f T_map_odom;  ///< map 系到 odom 系的变换
  double fitness_score;              ///< GICP fitness score（越小越好）
  double translation_m;              ///< 平移量 (m)
  double rotation_deg;               ///< 旋转量 (deg)
  int src_points;                    ///< 源点云点数
  int tgt_points;                    ///< 目标点云点数
  bool success;
};

class RelocalizerCore
{
public:
  RelocalizerCore();

  /**
   * @brief 初始化 KISS-Matcher 和 GICP 参数
   */
  void configure(double kiss_voxel_size,
                double gicp_voxel_size,
                int gicp_max_iter,
                double gicp_trans_eps,
                double gicp_rot_eps);

  /**
   * @brief 执行完整重定位流程（支持搜索窗口约束）
   * @param current_cloud 当前帧点云（odom 系）
   * @param map_cloud    先验地图点云（map 系）
   * @param search_center 搜索中心（用于裁剪地图，仅 XY 平面有效）
   * @param search_radius 搜索半径（m），0 表示不裁剪
   * @return 重定位结果
   *
   * 注意：search_center 在 odom 系，map_cloud 在 map 系。
   * XY 裁剪在 odom 漂移 < search_radius 的前提下近似有效。
   * Z 轴不做过滤，因 odom Z 与 map Z 无对应关系。
   */
  std::optional<RelocalizationResult> relocalize(
      const typename PointCloudT::ConstPtr& current_cloud,
      const typename PointCloudT::ConstPtr& map_cloud,
      const Eigen::Vector3f& search_center = Eigen::Vector3f::Zero(),
      float search_radius = 0.0f);

private:
  // KISS-Matcher 全局匹配
  std::unique_ptr<kiss_matcher::KISSMatcher> kiss_matcher_;

  // 配准参数
  double kiss_voxel_size_;
  double gicp_voxel_size_;
  int gicp_max_iter_;
  double gicp_trans_eps_;
  double gicp_rot_eps_;

  /**
   * @brief 步骤1：KISS-Matcher 全局特征匹配（无初值）
   * @return 粗配准变换 T_map_source（map 系到 source 系）
   */
  std::optional<Eigen::Isometry3f> globalMatch(
      const typename PointCloudT::ConstPtr& source,
      const typename PointCloudT::ConstPtr& target);

  struct FineAlignResult {
    Eigen::Isometry3f transform;
    double fitness_score;
  };

  /**
   * @brief 步骤2：pclomp::GICP 精细配准
   * @param initial_guess 粗配准结果作为初值
   * @return 精配准变换 + fitness score
   */
  std::optional<FineAlignResult> fineAlign(
      const typename PointCloudT::ConstPtr& source,
      const typename PointCloudT::ConstPtr& target,
      const Eigen::Isometry3f& initial_guess);
};

}  // namespace lidar_3d_relocalizer

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
  int kiss_rotation_inliers{0};
  int kiss_translation_inliers{0};
  bool used_gicp{false};
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
                double gicp_rot_eps,
                int min_kiss_translation_inliers = 6,
                bool allow_kiss_fallback = false,
                int min_rotation_inliers = 10,
                bool enable_gicp_fallback = true,
                float gicp_fallback_search_radius = 5.0f,
                bool enable_multi_yaw_search = true,
                int multi_yaw_search_samples = 5,
                double multi_yaw_search_range_deg = 30.0);

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
int min_kiss_translation_inliers_{6};
  bool allow_kiss_fallback_{false};
  int min_rotation_inliers_{10};
  bool enable_gicp_fallback_{true};
  float gicp_fallback_search_radius_{5.0f};
  /// 多方向旋转搜索：在 KISS rotation 附近采样候选角度做 GICP
  bool enable_multi_yaw_search_{true};
  int multi_yaw_search_samples_{5};      // 候选角度数（含 0°）
  double multi_yaw_search_range_deg_{30.0};  // 搜索范围 ±deg

  struct CoarseMatchResult
  {
    Eigen::Isometry3f transform;
    int rotation_inliers{0};
    int translation_inliers{0};
    /// KISS valid=false 但旋转有效时，rotation-only fallback 标志。
    /// 此时 transform.rotation()=KISS 估计的旋转，transform.translation()=centroid alignment。
    /// relocalize() 检测此标志后启用多方向旋转搜索。
    bool rotation_only_fallback{false};
    Eigen::Matrix3d kiss_rotation_only;  // KISS 原始旋转矩阵（用于多方向搜索）
    Eigen::Vector3d src_centroid;        // 源质心（odom 系）
    Eigen::Vector3d tgt_centroid;        // 目标质心（map 系）
  };

  /**
   * @brief 步骤1：KISS-Matcher 全局特征匹配（无初值）
   * @return 粗配准变换 T_map_source（map 系到 source 系）
   */
  std::optional<CoarseMatchResult> globalMatch(
      const typename PointCloudT::ConstPtr& source,
      const typename PointCloudT::ConstPtr& target);

  struct FineAlignResult {
    Eigen::Isometry3f transform;
    double fitness_score;
  };

  /**
   * @brief 步骤2：pclomp::GICP 精细配准
   * @param initial_guess 粗配准结果作为初值
   * @param max_corr_dist 最大对应距离（m），0=使用默认 5.0m
   * @return 精配准变换 + fitness score
   */
  std::optional<FineAlignResult> fineAlign(
      const typename PointCloudT::ConstPtr& source,
      const typename PointCloudT::ConstPtr& target,
      const Eigen::Isometry3f& initial_guess,
      float max_corr_dist = 0.0f);
};

}  // namespace lidar_3d_relocalizer

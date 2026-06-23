#include "lidar_3d_relocalizer/relocalizer_core.hpp"
#include <pclomp/gicp_omp.h>
#include <pcl/filters/voxel_grid.h>
#include <iostream>
#include <limits>
#include <Eigen/Geometry>

namespace lidar_3d_relocalizer
{

RelocalizerCore::RelocalizerCore() = default;

void RelocalizerCore::configure(double kiss_voxel_size,
                               double gicp_voxel_size,
                               int gicp_max_iter,
                               double gicp_trans_eps,
                               double gicp_rot_eps)
{
  kiss_voxel_size_  = kiss_voxel_size;
  gicp_voxel_size_  = gicp_voxel_size;
  gicp_max_iter_    = gicp_max_iter;
  gicp_trans_eps_   = gicp_trans_eps;
  gicp_rot_eps_     = gicp_rot_eps;

  // 初始化 KISS-Matcher（启用 Quatro：利用地面机器人重力方向约束）
  kiss_matcher::KISSMatcherConfig km_config(static_cast<float>(kiss_voxel_size));
  km_config.use_quatro_ = true;   // 地面机器人：假定 Z 轴与重力对齐
  kiss_matcher_ = std::make_unique<kiss_matcher::KISSMatcher>(km_config);
}

// ─── 工具函数：PCL PointCloud → std::vector<Eigen::Vector3f> ───
static std::vector<Eigen::Vector3f> pclToEigenVec(
    const pcl::PointCloud<pcl::PointXYZI>::ConstPtr& cloud)
{
  std::vector<Eigen::Vector3f> out;
  out.reserve(cloud->size());
  for (const auto& pt : *cloud) {
    if (!std::isfinite(pt.x) || !std::isfinite(pt.y) || !std::isfinite(pt.z))
      continue;
    out.emplace_back(pt.x, pt.y, pt.z);
  }
  return out;
}

// ─── 工具函数：Eigen Isometry3f → Matrix4f ───
static Eigen::Matrix4f isoToMatrix4f(const Eigen::Isometry3f& T)
{
  return T.matrix();
}

// ─── 全局匹配（KISS-Matcher） ───
std::optional<Eigen::Isometry3f> RelocalizerCore::globalMatch(
    const PointCloudT::ConstPtr& source,
    const PointCloudT::ConstPtr& target)
{
  if (!source || source->empty() || !target || target->empty()) {
    std::cerr << "[RelocalizerCore] globalMatch: empty cloud" << std::endl;
    return std::nullopt;
  }

  auto src_vec  = pclToEigenVec(source);
  auto tgt_vec = pclToEigenVec(target);

  if (src_vec.size() < 100 || tgt_vec.size() < 100) {
    std::cerr << "[RelocalizerCore] globalMatch: too few points (src="
              << src_vec.size() << ", tgt=" << tgt_vec.size() << ")"
              << std::endl;
    return std::nullopt;
  }

  std::cout << "[RelocalizerCore] KISS-Matcher input: "
            << "src=" << src_vec.size() << " pts, tgt=" << tgt_vec.size() << " pts"
            << std::endl;

  // KISS-Matcher: estimate() returns RegistrationSolution
  auto solution = kiss_matcher_->estimate(src_vec, tgt_vec);

  // 诊断输出：查看 KISS-Matcher 内部各阶段统计
  kiss_matcher_->print();

  if (!solution.valid) {
    std::cerr << "[RelocalizerCore] KISS-Matcher did not converge"
              << std::endl;
    return std::nullopt;
  }

  // RegistrationSolution: .rotation (Matrix3d) + .translation (Vector3d)
  Eigen::Isometry3f T_map_source = Eigen::Isometry3f::Identity();
  T_map_source.rotate(solution.rotation.cast<float>());
  T_map_source.translation() = solution.translation.cast<float>();

  std::cout << "[RelocalizerCore] KISS-Matcher result:\n"
            << "  translation: " << T_map_source.translation().transpose()
            << "\n  rotation R:\n" << T_map_source.rotation() << std::endl;

  return T_map_source;
}

// ─── 精细配准（pclomp::GICP） ───
std::optional<RelocalizerCore::FineAlignResult> RelocalizerCore::fineAlign(
    const PointCloudT::ConstPtr& source,
    const PointCloudT::ConstPtr& target,
    const Eigen::Isometry3f& initial_guess)
{
  if (!source || source->empty() || !target || target->empty())
    return std::nullopt;

  // 只对源点云（机器人当前帧）降采样，目标（地图）已预降采样
  auto src_ds = std::make_shared<PointCloudT>();
  pcl::VoxelGrid<PointT> vg;
  vg.setLeafSize(static_cast<float>(gicp_voxel_size_),
                 static_cast<float>(gicp_voxel_size_),
                 static_cast<float>(gicp_voxel_size_));
  vg.setInputCloud(source); vg.filter(*src_ds);

  if (src_ds->empty()) {
    std::cerr << "[RelocalizerCore] Source cloud empty after downsampling" << std::endl;
    return std::nullopt;
  }

  std::cerr << "[RelocalizerCore] GICP input: src=" << src_ds->size()
            << " pts, tgt=" << target->size() << " pts" << std::endl;

  // pclomp::GICP
  pclomp::GeneralizedIterativeClosestPoint<PointT, PointT> gicp;
  gicp.setMaxCorrespondenceDistance(3.0);   // 室内 3m 足够
  gicp.setMaximumIterations(gicp_max_iter_);
  gicp.setTransformationEpsilon(static_cast<float>(gicp_trans_eps_));
  gicp.setRotationEpsilon(static_cast<float>(gicp_rot_eps_));
  gicp.setEuclideanFitnessEpsilon(static_cast<float>(1e-4));  // 适当放宽

  gicp.setInputSource(src_ds);
  gicp.setInputTarget(target);

  Eigen::Matrix4f init_mat = isoToMatrix4f(initial_guess);
  gicp.align(*src_ds, init_mat);

  if (!gicp.hasConverged()) {
    std::cerr << "[RelocalizerCore] GICP did not converge" << std::endl;
    return std::nullopt;
  }

  Eigen::Isometry3f T_fine = Eigen::Isometry3f::Identity();
  T_fine.matrix() = gicp.getFinalTransformation();

  double fitness = gicp.getFitnessScore();
  std::cout << "[RelocalizerCore] GICP converged. fitness_score="
            << fitness << std::endl;

  FineAlignResult result;
  result.transform = T_fine;
  result.fitness_score = fitness;
  return result;
}

// ─── 裁剪点云到局部区域（仅 XY 平面）───
// 注意：search_center 来自 odom 系，PCD 地图在 map 系，两个坐标系不同。
// XY 裁剪在 odom 漂移 < search_radius 的前提下近似有效；
// Z 轴不做过滤，因为 odom Z 与 map Z 无对应关系。
static PointCloudT::Ptr cropCloudToLocal(
    const PointCloudT::ConstPtr& cloud,
    const Eigen::Vector3f& center, float radius)
{
  if (!cloud || cloud->empty()) return nullptr;

  auto cropped = std::make_shared<PointCloudT>();
  float r2 = radius * radius;

  for (const auto& pt : *cloud) {
    float dx = pt.x - center.x();
    float dy = pt.y - center.y();
    if (dx * dx + dy * dy <= r2) {
      cropped->push_back(pt);
    }
  }

  std::cerr << "[RelocalizerCore] Cropped map: " << cloud->size()
            << " → " << cropped->size() << " pts (XY radius=" << radius
            << "m, center=(" << center.x() << "," << center.y()
            << "))" << std::endl;
  return cropped;
}

// ─── 完整重定位流程（支持搜索窗口约束）───
std::optional<RelocalizationResult> RelocalizerCore::relocalize(
    const PointCloudT::ConstPtr& current_cloud,
    const PointCloudT::ConstPtr& map_cloud,
    const Eigen::Vector3f& search_center,
    float search_radius)
{
  RelocalizationResult result;
  result.success = false;
  result.fitness_score = std::numeric_limits<double>::max();
  result.translation_m = 0.0;
  result.rotation_deg = 0.0;
  result.src_points = current_cloud ? static_cast<int>(current_cloud->size()) : 0;
  result.tgt_points = map_cloud ? static_cast<int>(map_cloud->size()) : 0;

  // 如果指定了搜索窗口，裁剪 PCD 地图到局部区域（仅 XY 平面）
  typename PointCloudT::ConstPtr target_cloud = map_cloud;
  if (search_radius > 0.0f) {
    auto cropped = cropCloudToLocal(map_cloud, search_center, search_radius);
    if (cropped && !cropped->empty()) {
      target_cloud = cropped;
    } else {
      std::cerr << "[RelocalizerCore] Cropped map is empty, using full map" << std::endl;
    }
    result.tgt_points = static_cast<int>(target_cloud->size());
  }

  // 步骤1：KISS-Matcher 全局粗匹配
  auto coarse = globalMatch(current_cloud, target_cloud);
  if (!coarse.has_value()) {
    std::cerr << "[RelocalizerCore] Global match failed" << std::endl;
    return std::nullopt;
  }

  // 步骤2：GICP 精细配准（可选，失败则用粗匹配结果）
  auto fine = fineAlign(current_cloud, target_cloud, coarse.value());
  if (fine.has_value()) {
    result.T_map_odom = fine->transform;
    result.fitness_score = fine->fitness_score;
    std::cerr << "[RelocalizerCore] Using GICP refined result" << std::endl;
  } else {
    result.T_map_odom = coarse.value();
    result.fitness_score = 0.1;
    std::cerr << "[RelocalizerCore] GICP failed, using KISS-Matcher coarse result" << std::endl;
  }

  // 计算变换量
  Eigen::Vector3f translation = result.T_map_odom.translation();
  result.translation_m = static_cast<double>(translation.norm());

  Eigen::Matrix3f rotation = result.T_map_odom.rotation();
  Eigen::AngleAxisf aa(rotation);
  result.rotation_deg = static_cast<double>(std::abs(aa.angle())) * 180.0 / M_PI;

  result.success = true;

  return result;
}

}  // namespace lidar_3d_relocalizer

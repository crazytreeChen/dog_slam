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
                               double gicp_rot_eps,
                               int min_kiss_translation_inliers,
                                bool allow_kiss_fallback,
                                int min_rotation_inliers,
                                bool enable_gicp_fallback,
                                float gicp_fallback_search_radius,
                                bool enable_multi_yaw_search,
                                int multi_yaw_search_samples,
                                double multi_yaw_search_range_deg)
{
  kiss_voxel_size_  = kiss_voxel_size;
  gicp_voxel_size_  = gicp_voxel_size;
  gicp_max_iter_    = gicp_max_iter;
  gicp_trans_eps_   = gicp_trans_eps;
  gicp_rot_eps_     = gicp_rot_eps;
  min_kiss_translation_inliers_ = min_kiss_translation_inliers;
  allow_kiss_fallback_ = allow_kiss_fallback;
  min_rotation_inliers_ = min_rotation_inliers;
  enable_gicp_fallback_ = enable_gicp_fallback;
  gicp_fallback_search_radius_ = gicp_fallback_search_radius;
  enable_multi_yaw_search_ = enable_multi_yaw_search;
  multi_yaw_search_samples_ = multi_yaw_search_samples;
  multi_yaw_search_range_deg_ = multi_yaw_search_range_deg;

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
std::optional<RelocalizerCore::CoarseMatchResult> RelocalizerCore::globalMatch(
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
    // 降级容错：旋转有效但平移无效时，返回 rotation-only fallback 标志。
    // 由 relocalize() 做多方向旋转搜索（multi-yaw search），
    // 而非在此直接做 centroid alignment → GICP。
    // 原因：KISS rotation 在退化环境中可能偏 10-30°，
    // 单一 centroid alignment 初值可能仍不够好。
    size_t rot_inliers = kiss_matcher_->getNumRotationInliers();
    if (rot_inliers >= static_cast<size_t>(min_rotation_inliers_)) {
      // 计算源质心（odom 系）和目标质心（map 系）
      Eigen::Vector3d src_centroid = Eigen::Vector3d::Zero();
      for (const auto& v : src_vec) src_centroid += v.cast<double>();
      src_centroid /= static_cast<double>(src_vec.size());

      Eigen::Vector3d tgt_centroid = Eigen::Vector3d::Zero();
      for (const auto& v : tgt_vec) tgt_centroid += v.cast<double>();
      tgt_centroid /= static_cast<double>(tgt_vec.size());

      std::cerr << "[RelocalizerCore] KISS valid=false but rot_inliers="
                << rot_inliers << " >= " << min_rotation_inliers_
                << ", rotation-only fallback (delegating multi-yaw search to relocalize)"
                << std::endl;
      std::cerr << "  src_centroid (odom): " << src_centroid.transpose() << std::endl;
      std::cerr << "  tgt_centroid (map):  " << tgt_centroid.transpose() << std::endl;

      CoarseMatchResult result;
      result.rotation_only_fallback = true;
      result.kiss_rotation_only = solution.rotation;
      result.src_centroid = src_centroid;
      result.tgt_centroid = tgt_centroid;
      result.rotation_inliers = static_cast<int>(rot_inliers);
      result.translation_inliers = 0;
      // transform 留空，由 relocalize() 填充
      result.transform = Eigen::Isometry3f::Identity();
      return result;
    }
    std::cerr << "[RelocalizerCore] KISS-Matcher did not converge (rot_inliers="
              << rot_inliers << " < " << min_rotation_inliers_ << ")" << std::endl;
    return std::nullopt;
  }

  // RegistrationSolution: .rotation (Matrix3d) + .translation (Vector3d)
  Eigen::Isometry3f T_map_source = Eigen::Isometry3f::Identity();
  T_map_source.rotate(solution.rotation.cast<float>());
  T_map_source.translation() = solution.translation.cast<float>();

  std::cout << "[RelocalizerCore] KISS-Matcher result:\n"
            << "  translation: " << T_map_source.translation().transpose()
            << "\n  rotation R:\n" << T_map_source.rotation() << std::endl;

  CoarseMatchResult result;
  result.transform = T_map_source;
  result.rotation_inliers = static_cast<int>(kiss_matcher_->getNumRotationInliers());
  result.translation_inliers = static_cast<int>(kiss_matcher_->getNumFinalInliers());

  // 质量门控：KISS 报告 valid=true 但 translation_inliers 不足时，
  // 平移不可信，降级为 rotation-only fallback。
  // 这避免了把几乎随机的 KISS 平移喂给 GICP（fitness 可达 20-70）。
  if (result.translation_inliers < min_kiss_translation_inliers_ &&
      result.rotation_inliers >= min_rotation_inliers_) {
    // 计算质心（用于 centroid alignment）
    Eigen::Vector3d src_centroid = Eigen::Vector3d::Zero();
    for (const auto& v : src_vec) src_centroid += v.cast<double>();
    src_centroid /= static_cast<double>(src_vec.size());

    Eigen::Vector3d tgt_centroid = Eigen::Vector3d::Zero();
    for (const auto& v : tgt_vec) tgt_centroid += v.cast<double>();
    tgt_centroid /= static_cast<double>(tgt_vec.size());

    std::cerr << "[RelocalizerCore] KISS valid=true but trans_inliers="
              << result.translation_inliers << " < " << min_kiss_translation_inliers_
              << ", rot_inliers=" << result.rotation_inliers << " >= " << min_rotation_inliers_
              << " → downgrading to rotation-only fallback" << std::endl;
    std::cerr << "  src_centroid (odom): " << src_centroid.transpose() << std::endl;
    std::cerr << "  tgt_centroid (map):  " << tgt_centroid.transpose() << std::endl;

    result.rotation_only_fallback = true;
    result.kiss_rotation_only = solution.rotation;
    result.src_centroid = src_centroid;
    result.tgt_centroid = tgt_centroid;
    // 保留原始 KISS 结果在 transform 中作为参考，但 rotation_only_fallback 优先
  }

  return result;
}

// ─── 精细配准（pclomp::GICP） ───
// 关键改进（参考 KISS-Matcher 官方 SLAM + hdl_graph_slam + VGICP 论文）：
//   1. 双侧同体素降采样——避免 1:80 密度比破坏 covariance 估计
//   2. max_corr_dist 可配置——正常路径 5m，rotation-only fallback 可扩大到 15-20m
//   3. setUseReciprocalCorrespondences(true)——双向 NN 显著降低大地图错误匹配
//   4. eps 回归 PCL 默认（5e-4 / 2e-3）——原 1e-5 严格 50 倍导致 early-stop 失败
//   5. 显式 setCorrespondenceRandomness + setMaximumOptimizerIterations
std::optional<RelocalizerCore::FineAlignResult> RelocalizerCore::fineAlign(
    const PointCloudT::ConstPtr& source,
    const PointCloudT::ConstPtr& target,
    const Eigen::Isometry3f& initial_guess,
    float max_corr_dist)
{
  if (!source || source->empty() || !target || target->empty())
    return std::nullopt;

  // 双侧同体素降采样（关键修复）
  auto src_ds = std::make_shared<PointCloudT>();
  auto tgt_ds = std::make_shared<PointCloudT>();
  pcl::VoxelGrid<PointT> vg;
  vg.setLeafSize(static_cast<float>(gicp_voxel_size_),
                 static_cast<float>(gicp_voxel_size_),
                 static_cast<float>(gicp_voxel_size_));
  vg.setInputCloud(source); vg.filter(*src_ds);
  vg.setInputCloud(target); vg.filter(*tgt_ds);

  if (src_ds->empty()) {
    std::cerr << "[RelocalizerCore] Source cloud empty after downsampling" << std::endl;
    return std::nullopt;
  }
  if (tgt_ds->empty()) {
    std::cerr << "[RelocalizerCore] Target cloud empty after downsampling" << std::endl;
    return std::nullopt;
  }

  std::cerr << "[RelocalizerCore] GICP input: src=" << src_ds->size()
            << " pts, tgt=" << tgt_ds->size()
            << " pts (target downsampled from " << target->size() << ")" << std::endl;

  // max_corr_dist: 默认 5.0m，rotation-only fallback 时可由调用方传入更大值
  if (max_corr_dist <= 0.0f)
    max_corr_dist = 5.0f;

  // pclomp::GICP
  pclomp::GeneralizedIterativeClosestPoint<PointT, PointT> gicp;
  gicp.setMaxCorrespondenceDistance(max_corr_dist);
  gicp.setMaximumIterations(gicp_max_iter_);
  // eps 回归 PCL 默认（5e-4 / 2e-3），原 1e-5 严格 50 倍，early-stop 失败
  gicp.setTransformationEpsilon(5e-4);
  gicp.setRotationEpsilon(2e-3);
  gicp.setEuclideanFitnessEpsilon(1e-5);
  // 显式设置（hdl_graph_slam 惯例）
  gicp.setCorrespondenceRandomness(20);
  gicp.setMaximumOptimizerIterations(20);
  // 关键：双向 NN，显著降低 7694-vs-628k 密度失衡导致的错误匹配
  gicp.setUseReciprocalCorrespondences(true);

  gicp.setInputSource(src_ds);
  gicp.setInputTarget(tgt_ds);

  Eigen::Matrix4f init_mat = isoToMatrix4f(initial_guess);
  gicp.align(*src_ds, init_mat);

  bool converged = gicp.hasConverged();
  double fitness = gicp.getFitnessScore();

  // 即使 hasConverged()==false，fitness 仍然可能有效
  // 不收敛的原因可能是迭代不够（early-stop），但 fitness < 阈值时结果仍可用
  // 真正无效的情况：fitness 为默认最大值或对齐后变换接近 identity
  if (!converged && fitness >= std::numeric_limits<double>::max() - 1.0) {
    std::cerr << "[RelocalizerCore] GICP did not converge and fitness is invalid (max_corr_dist="
              << max_corr_dist << "m, src=" << src_ds->size()
              << ", tgt=" << tgt_ds->size() << ")" << std::endl;
    return std::nullopt;
  }

  Eigen::Isometry3f T_fine = Eigen::Isometry3f::Identity();
  T_fine.matrix() = gicp.getFinalTransformation();

  std::cout << "[RelocalizerCore] GICP " << (converged ? "converged" : "DID NOT converge (using result anyway)")
            << ". fitness=" << fitness << " (max_corr_dist=" << max_corr_dist
            << "m, src=" << src_ds->size()
            << ", tgt=" << tgt_ds->size() << ")" << std::endl;

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

    // GICP 兜底：KISS 完全失败时，用 identity 初值 + 缩小半径尝试 GICP
    if (enable_gicp_fallback_ && gicp_fallback_search_radius_ > 0.0f) {
      std::cerr << "[RelocalizerCore] Attempting GICP fallback with identity initial guess, "
                << "search_radius=" << gicp_fallback_search_radius_ << "m" << std::endl;

      typename PointCloudT::ConstPtr fallback_target = target_cloud;
      if (search_radius > 0.0f) {
        auto cropped = cropCloudToLocal(map_cloud, search_center,
                                         gicp_fallback_search_radius_);
        if (cropped && !cropped->empty()) {
          fallback_target = cropped;
          std::cerr << "[RelocalizerCore] GICP fallback cropped map: "
                    << fallback_target->size() << " pts" << std::endl;
        } else {
          std::cerr << "[RelocalizerCore] GICP fallback crop empty, using previous target"
                    << std::endl;
        }
      }

      Eigen::Isometry3f identity_guess = Eigen::Isometry3f::Identity();
      auto fine = fineAlign(current_cloud, fallback_target, identity_guess);
      if (fine.has_value()) {
        result.T_map_odom = fine->transform;
        result.fitness_score = fine->fitness_score;
        result.used_gicp = true;
        result.kiss_rotation_inliers = 0;
        result.kiss_translation_inliers = 0;

        Eigen::Vector3f translation = result.T_map_odom.translation();
        result.translation_m = static_cast<double>(translation.norm());
        Eigen::Matrix3f rotation = result.T_map_odom.rotation();
        Eigen::AngleAxisf aa(rotation);
        result.rotation_deg = static_cast<double>(std::abs(aa.angle())) * 180.0 / M_PI;
        result.success = true;

        std::cerr << "[RelocalizerCore] GICP fallback succeeded! fitness="
                  << fine->fitness_score << std::endl;
        return result;
      }
      std::cerr << "[RelocalizerCore] GICP fallback also failed" << std::endl;
    }
    return std::nullopt;
  }

  // 步骤2：GICP 精细配准
  // 如果 coarse 是 rotation-only fallback，启用多方向旋转搜索
  // 核心改进：两级 GICP 退火策略
  //   第 1 级：大 max_corr_dist（20m）做粗对齐——桥接 centroid alignment 偏差
  //   第 2 级：小 max_corr_dist（5m）做精修——在粗对齐基础上精细收敛
  if (coarse->rotation_only_fallback && enable_multi_yaw_search_) {
    std::cerr << "[RelocalizerCore] Multi-yaw search enabled: "
              << multi_yaw_search_samples_ << " candidates within ±"
              << multi_yaw_search_range_deg_ << "°" << std::endl;

    // 在 KISS rotation 附近采样候选 yaw 角
    double range_rad = multi_yaw_search_range_deg_ * M_PI / 180.0;
    int n = multi_yaw_search_samples_;
    // 对称采样: [−range, ..., 0, ..., +range]
    double best_fitness = std::numeric_limits<double>::max();
    Eigen::Isometry3f best_transform = Eigen::Isometry3f::Identity();
    bool found = false;

    // ── 两级 GICP 退火的 max_corr_dist ──
    const float coarse_max_corr_dist = 20.0f;   // 第 1 级：桥接 centroid 偏差（~4.5m）
    const float fine_max_corr_dist = 5.0f;       // 第 2 级：标准精细收敛

    for (int i = 0; i < n; ++i) {
      double offset = (n == 1) ? 0.0 : -range_rad + 2.0 * range_rad * i / (n - 1);
      Eigen::Matrix3d R_candidate =
          coarse->kiss_rotation_only *
          Eigen::AngleAxisd(offset, Eigen::Vector3d::UnitZ()).toRotationMatrix();

      // centroid alignment: t = src_centroid - R * tgt_centroid
      Eigen::Vector3d t_candidate =
          coarse->src_centroid - R_candidate * coarse->tgt_centroid;

      Eigen::Isometry3f T_candidate = Eigen::Isometry3f::Identity();
      T_candidate.rotate(R_candidate.cast<float>());
      T_candidate.translation() = t_candidate.cast<float>();

      // 第 1 级：大 max_corr_dist 粗对齐
      auto coarse_align = fineAlign(current_cloud, target_cloud,
                                    T_candidate, coarse_max_corr_dist);
      if (!coarse_align.has_value()) {
        std::cerr << "  yaw[" << i << "/" << n << "] offset="
                  << (offset * 180.0 / M_PI) << "° coarse GICP failed" << std::endl;
        continue;
      }

      // 第 2 级：小 max_corr_dist 精修（以粗对齐结果为初值）
      auto fine = fineAlign(current_cloud, target_cloud,
                           coarse_align->transform, fine_max_corr_dist);
      if (!fine.has_value()) {
        std::cerr << "  yaw[" << i << "/" << n << "] offset="
                  << (offset * 180.0 / M_PI)
                  << "° fine GICP failed (coarse fitness="
                  << coarse_align->fitness_score << ")" << std::endl;
        continue;
      }

      double f = fine->fitness_score;
      std::cerr << "  yaw[" << i << "/" << n << "] offset="
                << (offset * 180.0 / M_PI)
                << "° coarse_fitness=" << coarse_align->fitness_score
                << " → fine_fitness=" << f << std::endl;

      if (f < best_fitness) {
        best_fitness = f;
        best_transform = fine->transform;
        found = true;
      }
    }

    if (found) {
      result.T_map_odom = best_transform;
      result.fitness_score = best_fitness;
      result.used_gicp = true;
      std::cerr << "[RelocalizerCore] Multi-yaw search (2-stage annealing): best fitness="
                << best_fitness << std::endl;
    } else {
      // 多方向搜索全部失败，回退到 KISS rotation + centroid alignment
      std::cerr << "[RelocalizerCore] Multi-yaw search: all GICP failed, "
                << "using KISS rotation + centroid alignment" << std::endl;
      result.T_map_odom = coarse->transform;
      result.fitness_score = 0.1;
      result.used_gicp = false;
    }
  } else if (coarse->rotation_only_fallback) {
    // 多方向搜索关闭时，用 centroid alignment 作为单次 GICP 初值（两级退火）
    Eigen::Isometry3f T_fallback = Eigen::Isometry3f::Identity();
    T_fallback.rotate(coarse->kiss_rotation_only.cast<float>());
    Eigen::Vector3d t = coarse->src_centroid -
        coarse->kiss_rotation_only * coarse->tgt_centroid;
    T_fallback.translation() = t.cast<float>();

    // 第 1 级：20m 粗对齐
    auto coarse_align = fineAlign(current_cloud, target_cloud,
                                  T_fallback, 20.0f);
    if (coarse_align.has_value()) {
      // 第 2 级：5m 精修
      auto fine = fineAlign(current_cloud, target_cloud,
                           coarse_align->transform, 5.0f);
      if (fine.has_value()) {
        result.T_map_odom = fine->transform;
        result.fitness_score = fine->fitness_score;
        result.used_gicp = true;
        std::cerr << "[RelocalizerCore] Rotation-only fallback (2-stage annealing): "
                  << "coarse_fitness=" << coarse_align->fitness_score
                  << " → fine_fitness=" << fine->fitness_score << std::endl;
      } else {
        result.T_map_odom = coarse_align->transform;
        result.fitness_score = coarse_align->fitness_score;
        result.used_gicp = true;
        std::cerr << "[RelocalizerCore] Rotation-only fallback: fine GICP failed, "
                  << "using coarse result (fitness="
                  << coarse_align->fitness_score << ")" << std::endl;
      }
    } else {
      result.T_map_odom = T_fallback;
      result.fitness_score = 0.1;
      result.used_gicp = false;
      std::cerr << "[RelocalizerCore] Rotation-only fallback: coarse GICP also failed, "
                << "using KISS rotation + centroid alignment directly" << std::endl;
    }
  } else {
    // 正常路径：KISS 完全成功，用 coarse->transform 做 GICP 初值
    auto fine = fineAlign(current_cloud, target_cloud, coarse->transform);
    if (fine.has_value()) {
      result.T_map_odom = fine->transform;
      result.fitness_score = fine->fitness_score;
      result.used_gicp = true;
      std::cerr << "[RelocalizerCore] Using GICP refined result" << std::endl;
    } else {
      if (!allow_kiss_fallback_) {
        std::cerr << "[RelocalizerCore] GICP failed and KISS fallback is disabled"
                  << std::endl;
        return std::nullopt;
      }
      if (coarse->translation_inliers < min_kiss_translation_inliers_) {
        std::cerr << "[RelocalizerCore] KISS fallback rejected: translation inliers "
                  << coarse->translation_inliers << " < "
                  << min_kiss_translation_inliers_ << std::endl;
        return std::nullopt;
      }

      result.T_map_odom = coarse->transform;
      result.fitness_score = 0.1;
      result.used_gicp = false;
      std::cerr << "[RelocalizerCore] GICP failed, using KISS-Matcher coarse result" << std::endl;
    }
  }

  result.kiss_rotation_inliers = coarse->rotation_inliers;
  result.kiss_translation_inliers = coarse->translation_inliers;

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

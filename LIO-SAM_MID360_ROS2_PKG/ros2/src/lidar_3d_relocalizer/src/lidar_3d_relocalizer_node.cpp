/**
 * @file   lidar_3d_relocalizer_node.cpp
 * @brief  3D LiDAR 初始位姿估计节点
 *
 * 订阅 LIO 输出的 3D 点云 + odom，
 * 用 KISS-Matcher 做全局特征匹配（无初值），
 * 再用 pclomp::GICP 精细配准，
 * 计算 map→odom 变换后发布 /initialpose 给 AMCL。
 *
 * 设计原则：
 *   - 不发布 map→odom TF（避免与 AMCL 冲突）
 *   - 只做一次性初始位姿估计，成功后停止
 *   - 支持命名空间（frame_id 从参数读取）
 *   - fitness 低于阈值时不发布，回退到 2D 方案
 */

#include <cmath>
#include <memory>
#include <mutex>
#include <string>
#include <vector>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <std_msgs/msg/string.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2/LinearMath/Quaternion.h>
#include <Eigen/Geometry>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/common/transforms.h>
#include <pcl/point_cloud.h>

#include "lidar_3d_relocalizer/pcd_map_loader.hpp"
#include "lidar_3d_relocalizer/cloud_accumulator.hpp"
#include "lidar_3d_relocalizer/relocalizer_core.hpp"
#include "lidar_3d_relocalizer/map_alignment.hpp"

namespace lidar_3d_relocalizer
{

class Lidar3dRelocalizerNode : public rclcpp::Node
{
public:
  explicit Lidar3dRelocalizerNode(const rclcpp::NodeOptions& options)
    : Node("lidar_3d_relocalizer_node", options)
  {
    RCLCPP_INFO(this->get_logger(), "Lidar3dRelocalizerNode starting...");

    // ─── 声明参数 ───
    declare_parameters();

    // tf2 不允许 frame_id 以 "/" 开头，自动去除
    auto strip_leading_slash = [](std::string& s) {
      if (!s.empty() && s[0] == '/') s.erase(0, 1);
    };
    strip_leading_slash(map_frame_);
    strip_leading_slash(odom_frame_);
    strip_leading_slash(base_frame_);

    // ─── 加载 PCD 地图 ───
    map_loader_ = std::make_unique<PcdMapLoader>();
    double map_voxel = this->get_parameter("accum_voxel_size").as_double();
    if (!map_loader_->load(pcd_map_path_, static_cast<float>(map_voxel))) {
      RCLCPP_ERROR(this->get_logger(), "Failed to load PCD map: %s",
                   pcd_map_path_.c_str());
      return;
    }

    // ─── 初始化配准核心 ───
    core_ = std::make_unique<RelocalizerCore>();
    core_->configure(kiss_voxel_size_, gicp_voxel_size_,
                     gicp_max_iter_, gicp_trans_eps_, gicp_rot_eps_,
                     min_kiss_translation_inliers_, allow_kiss_fallback_,
                     min_rotation_inliers_, enable_gicp_fallback_,
                     gicp_fallback_search_radius_,
                     enable_multi_yaw_search_, multi_yaw_search_samples_,
                     multi_yaw_search_range_deg_);

    // ─── 初始化累积器 ───
    accumulator_ = std::make_unique<CloudAccumulator>();

    // ─── TF listener（用于将 body frame 点云变换到 odom 系）───
    tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_unique<tf2_ros::TransformListener>(*tf_buffer_);

    // ─── 订阅话题 ───
    // 使用 BEST_EFFORT QoS 兼容传感器数据（LIO 点云发布者通常用 BEST_EFFORT）
    auto cloud_qos = rclcpp::QoS(rclcpp::KeepLast(10))
                         .best_effort()
                         .durability_volatile();
    cloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
        cloud_topic_, cloud_qos,
        std::bind(&Lidar3dRelocalizerNode::cloudCallback, this,
                  std::placeholders::_1));

    odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
        odom_topic_, 50,
        std::bind(&Lidar3dRelocalizerNode::odomCallback, this,
                  std::placeholders::_1));

    // ─── 发布话题（使用命名空间感知的 topic name）───
    initial_pose_pub_ =
        this->create_publisher<geometry_msgs::msg::PoseWithCovarianceStamped>(
            initialpose_topic_, 10);

    status_pub_ = this->create_publisher<std_msgs::msg::String>(
            status_topic_, 10);

    // ─── CmdVel 发布器（用于原地旋转采集）───
    if (rotate_enable_) {
      cmd_vel_pub_ = this->create_publisher<geometry_msgs::msg::Twist>(
          rotate_cmd_vel_topic_, 10);
    }

    // ─── TF 广播器（仅在 publish_tf=true 时使用）───
    if (publish_tf_) {
      tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
    }

    // ─── 重定位定时器 ───
    double rate_hz = this->get_parameter("relocalize_rate_hz").as_double();
    relocalize_timer_ = this->create_wall_timer(
        std::chrono::milliseconds(static_cast<int>(1000.0 / rate_hz)),
        std::bind(&Lidar3dRelocalizerNode::relocalizeTimerCallback, this));

    // ─── 旋转控制定时器（25Hz，仅在旋转期间工作）───
    if (rotate_enable_) {
      rotation_timer_ = this->create_wall_timer(
          std::chrono::milliseconds(40),
          std::bind(&Lidar3dRelocalizerNode::rotationControlCallback, this));
    }

    // ─── 等待 TF 可用（短超时轮询，避免 tf2 内部刷警告）───
    // PCD 地图保持在原始 map 系不变，KISS-Matcher 做 odom→map 全局匹配
    {
      has_tf_ = false;
      const int max_wait_sec = 15;
      const auto frame_exists = [this](const std::string& frame) {
        return tf_buffer_->allFramesAsString().find(frame) != std::string::npos;
      };
      auto start = this->now();
      while (rclcpp::ok() && !has_tf_) {
        const auto elapsed = (this->now() - start).seconds();
        if (elapsed > max_wait_sec) break;

        if (!frame_exists(odom_frame_) || !frame_exists(base_frame_)) {
          rclcpp::sleep_for(std::chrono::milliseconds(1500));  // 等发布者启动
          continue;
        }

        std::string tf_error;
        if (tf_buffer_->canTransform(odom_frame_, base_frame_,
                                     tf2::TimePointZero,
                                     tf2::durationFromSec(0.0),
                                     &tf_error)) {
          has_tf_ = true;
        } else {
          rclcpp::sleep_for(std::chrono::milliseconds(1500));
        }
      }
    }
    if (has_tf_) {
      RCLCPP_INFO(this->get_logger(), "TF %s→%s available",
                  odom_frame_.c_str(), base_frame_.c_str());
      RCLCPP_INFO(this->get_logger(),
                   "PCD map kept in original map frame (NOT transformed to odom). "
                   "KISS-Matcher will directly find T_map_odom.");
    } else {
      RCLCPP_WARN(this->get_logger(),
                   "TF %s→%s not available after %ds. "
                   "Will use odom pose directly (body frame assumption).",
                   odom_frame_.c_str(), base_frame_.c_str(), 15);
    }

    RCLCPP_INFO(this->get_logger(),
                 "Node initialized. Map: %s, cloud: %s, odom: %s, "
                 "frames: map=%s odom=%s base=%s",
                 pcd_map_path_.c_str(), cloud_topic_.c_str(),
                 odom_topic_.c_str(), map_frame_.c_str(),
                 odom_frame_.c_str(), base_frame_.c_str());
  }

private:
  // ─── 参数变量 ───
  std::string pcd_map_path_;
  std::string cloud_topic_;
  std::string odom_topic_;
  std::string map_frame_;
  std::string odom_frame_;
  std::string base_frame_;
  std::string initialpose_topic_;
  std::string status_topic_;
  double kiss_voxel_size_;
  double gicp_voxel_size_;
  int    gicp_max_iter_;
  double gicp_trans_eps_;
  double gicp_rot_eps_;
  int    min_kiss_translation_inliers_;
  int    accum_frame_count_;
  int    min_relocalize_points_;
  double accum_max_time_s_;
  float  accum_voxel_size_;
  double fitness_thresh_;
  bool   publish_tf_;
  bool   publish_initial_pose_;
  bool   allow_kiss_fallback_;
  int    min_rotation_inliers_;
  bool   enable_gicp_fallback_;
  float  gicp_fallback_search_radius_;
  bool   enable_multi_yaw_search_;
  int    multi_yaw_search_samples_;
  double multi_yaw_search_range_deg_;
  int    max_retry_;

  // ─── 原地旋转采集参数 ───
  bool   rotate_enable_{false};
  double rotate_angular_speed_{0.3};
  double rotate_total_angle_deg_{360.0};
  bool   rotate_obstacle_check_{true};
  double rotate_obstacle_min_dist_{0.5};
  double rotate_obstacle_check_range_deg_{45.0};
  std::string rotate_cmd_vel_topic_{"/cmd_vel"};

  // ─── 组件 ───
  std::unique_ptr<PcdMapLoader>    map_loader_;
  std::unique_ptr<CloudAccumulator> accumulator_;
  std::unique_ptr<RelocalizerCore>  core_;

  // ─── ROS 接口 ───
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_sub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr
      initial_pose_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr status_pub_;
  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_pub_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
  std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
  std::unique_ptr<tf2_ros::TransformListener> tf_listener_;
  rclcpp::TimerBase::SharedPtr relocalize_timer_;
  rclcpp::TimerBase::SharedPtr rotation_timer_;

  // ─── 状态 ───
  geometry_msgs::msg::Pose latest_odom_pose_;
  bool has_odom_{false};
  bool has_tf_{false};
  bool is_localized_{false};
  int retry_count_{0};

  // ─── 旋转采集状态 ───
  enum class RotationPhase { IDLE, ROTATING, STOPPING, DONE };
  RotationPhase rot_phase_{RotationPhase::IDLE};
  double rotation_start_yaw_{0.0};
  double total_rotated_angle_{0.0};
  double last_rotation_yaw_{0.0};
  bool rotation_yaw_init_{false};
  rclcpp::Time rotation_stop_time_;
  Eigen::Isometry3f T_map_odom_;
  pcl::PointCloud<pcl::PointXYZI>::Ptr latest_cloud_{nullptr};
  geometry_msgs::msg::Pose first_odom_pose_;
  bool has_first_odom_{false};
  double map_offset_x_, map_offset_y_, map_offset_yaw_deg_;
  std::string initial_pose_mode_;
  double search_window_radius_;
  std::string alignment_mode_;
  bool alignment_allow_scale_{false};
  std::vector<double> alignment_points_flat_;
  AlignmentTransform alignment_transform_;

  // ─── 参数声明 ───
  void declare_parameters()
  {
    // 坐标系参数
    map_frame_    = this->declare_parameter<std::string>("map_frame", "map");
    odom_frame_   = this->declare_parameter<std::string>("odom_frame", "odom");
    base_frame_   = this->declare_parameter<std::string>("base_frame", "base_footprint");

    // 话题参数
    pcd_map_path_ = this->declare_parameter<std::string>("pcd_map_path", "");
    cloud_topic_  = this->declare_parameter<std::string>("cloud_topic",
                                                         "/cloud_registered_body");
    odom_topic_   = this->declare_parameter<std::string>("odom_topic",
                                                         "/lio/robo/odom");

    // 命名空间感知的发布话题
    std::string ns = this->get_namespace();
    if (!ns.empty() && ns != "/") {
      initialpose_topic_ = ns + "/initialpose";
      status_topic_ = ns + "/relocalizer/status";
    } else {
      initialpose_topic_ = "/initialpose";
      status_topic_ = "/relocalizer/status";
    }
    // 允许参数覆盖
    initialpose_topic_ = this->declare_parameter<std::string>(
        "initialpose_topic", initialpose_topic_);
    status_topic_ = this->declare_parameter<std::string>(
        "status_topic", status_topic_);

    // 配准参数
    kiss_voxel_size_ = this->declare_parameter<double>("kiss_matcher_voxel_size", 0.5);
    gicp_voxel_size_ = this->declare_parameter<double>("gicp_voxel_size", 0.1);
    gicp_max_iter_   = this->declare_parameter<int>("gicp_max_iterations", 50);
    gicp_trans_eps_  = this->declare_parameter<double>("gicp_trans_eps", 1e-8);
    gicp_rot_eps_    = this->declare_parameter<double>("gicp_rot_eps", 1e-8);
    min_kiss_translation_inliers_ =
        this->declare_parameter<int>("min_kiss_translation_inliers", 6);

    // 累积参数
    accum_frame_count_ = this->declare_parameter<int>("accum_frame_count", 20);
    min_relocalize_points_ =
        this->declare_parameter<int>("min_relocalize_points", 0);  // 0=不检查点数，允许单帧匹配
    accum_max_time_s_  = this->declare_parameter<double>("accum_max_time_sec", 2.0);
    accum_voxel_size_  = static_cast<float>(
        this->declare_parameter<double>("accum_voxel_size", 0.05));

    // 质量门控
    fitness_thresh_ = this->declare_parameter<double>("gicp_fitness_threshold", 0.5);

    // 行为控制
    double rate_hz_tmp    = this->declare_parameter<double>("relocalize_rate_hz", 0.5);
    publish_tf_           = this->declare_parameter<bool>("publish_tf", false);
    publish_initial_pose_ = this->declare_parameter<bool>("publish_initial_pose", true);
    allow_kiss_fallback_  = this->declare_parameter<bool>("allow_kiss_fallback", false);

    // KISS-Matcher 降级容错
    min_rotation_inliers_ =
        this->declare_parameter<int>("min_rotation_inliers", 10);
    enable_gicp_fallback_ =
        this->declare_parameter<bool>("enable_gicp_fallback", true);
    gicp_fallback_search_radius_ = static_cast<float>(
        this->declare_parameter<double>("gicp_fallback_search_radius", 5.0));
    // 多方向旋转搜索
    enable_multi_yaw_search_ =
        this->declare_parameter<bool>("enable_multi_yaw_search", true);
    multi_yaw_search_samples_ =
        this->declare_parameter<int>("multi_yaw_search_samples", 7);
    multi_yaw_search_range_deg_ =
        this->declare_parameter<double>("multi_yaw_search_range_deg", 30.0);

    max_retry_            = this->declare_parameter<int>("max_retry", 3);

    // ─── 原地旋转采集参数 ───
    rotate_enable_ = this->declare_parameter<bool>("rotate_enable", false);
    rotate_angular_speed_ = this->declare_parameter<double>("rotate_angular_speed", 0.3);
    rotate_total_angle_deg_ = this->declare_parameter<double>("rotate_total_angle_deg", 360.0);
    rotate_obstacle_check_ = this->declare_parameter<bool>("rotate_obstacle_check", true);
    rotate_obstacle_min_dist_ = this->declare_parameter<double>("rotate_obstacle_min_dist", 0.5);
    rotate_obstacle_check_range_deg_ =
        this->declare_parameter<double>("rotate_obstacle_check_range_deg", 45.0);
    rotate_cmd_vel_topic_ =
        this->declare_parameter<std::string>("rotate_cmd_vel_topic", "/cmd_vel");

    // PCD → 2D 地图坐标偏移（PCD原点在2D地图中的坐标）
    map_offset_x_ = this->declare_parameter<double>("map_offset_x", 0.0);
    map_offset_y_ = this->declare_parameter<double>("map_offset_y", 0.0);
    map_offset_yaw_deg_ = this->declare_parameter<double>("map_offset_yaw_deg", 0.0);
    alignment_mode_ = this->declare_parameter<std::string>("alignment_mode", "offset");
    alignment_allow_scale_ = this->declare_parameter<bool>("alignment_allow_scale", false);
    alignment_points_flat_ = this->declare_parameter<std::vector<double>>(
        "alignment_points", std::vector<double>{});
    initializeAlignment();

    // 发布模式: "2d" 或 "3d"
    initial_pose_mode_ = this->declare_parameter<std::string>("initial_pose_mode", "2d");

    // 搜索窗口半径（裁剪 PCD 地图到局部区域，仅 XY 平面）
    // 注意：search_center 在 odom 系，map_cloud 在 map 系，
    // XY 裁剪在 odom 漂移 < search_radius 的前提下近似有效
    search_window_radius_ = this->declare_parameter<double>("search_window_radius", 30.0);
  }

  // ─── 点云回调 ───
  void initializeAlignment()
  {
    alignment_transform_ =
        MapAlignment::fromOffset(map_offset_x_, map_offset_y_, map_offset_yaw_deg_);

    if (alignment_mode_ != "landmarks") {
      RCLCPP_INFO(this->get_logger(),
                  "Using offset alignment: x=%.3f y=%.3f yaw=%.3fdeg",
                  map_offset_x_, map_offset_y_, map_offset_yaw_deg_);
      return;
    }

    if (alignment_points_flat_.size() % 4 != 0) {
      RCLCPP_WARN(this->get_logger(),
                  "alignment_points size must be a multiple of 4 "
                  "[pcd_x, pcd_y, map_x, map_y]. Falling back to offset alignment.");
      return;
    }

    std::vector<AlignmentPoint> points;
    points.reserve(alignment_points_flat_.size() / 4);
    for (size_t i = 0; i < alignment_points_flat_.size(); i += 4) {
      points.push_back({
          alignment_points_flat_[i],
          alignment_points_flat_[i + 1],
          alignment_points_flat_[i + 2],
          alignment_points_flat_[i + 3],
      });
    }

    std::string error;
    auto estimated =
        MapAlignment::estimateFromLandmarks(points, alignment_allow_scale_, &error);
    if (!estimated.valid) {
      RCLCPP_WARN(this->get_logger(),
                  "Landmark alignment invalid: %s. Falling back to offset alignment.",
                  error.c_str());
      return;
    }

    alignment_transform_ = estimated;
    RCLCPP_INFO(this->get_logger(),
                "Using landmark alignment: points=%zu allow_scale=%s scale=%.6f yaw=%.3fdeg "
                "tx=%.3f ty=%.3f rms=%.4f",
                points.size(),
                alignment_allow_scale_ ? "true" : "false",
                alignment_transform_.scale,
                alignment_transform_.yaw * 180.0 / M_PI,
                alignment_transform_.tx,
                alignment_transform_.ty,
                alignment_transform_.rms_error);
    const double c = std::cos(alignment_transform_.yaw);
    const double s = std::sin(alignment_transform_.yaw);
    const double scale = alignment_transform_.scale;
    RCLCPP_INFO(this->get_logger(),
                "PCD->2D map matrix: [[%.6f, %.6f, %.6f], "
                "[%.6f, %.6f, %.6f], [0, 0, 1]]",
                scale * c,
                -scale * s,
                alignment_transform_.tx,
                scale * s,
                scale * c,
                alignment_transform_.ty);
  }

  void cloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
  {
    if (is_localized_) return;

    auto pcl_cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
    pcl::fromROSMsg(*msg, *pcl_cloud);
    latest_cloud_ = pcl_cloud;

    if (!has_odom_) {
      RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                           "Cloud received, waiting for odom before accumulation");
      return;
    }

    // 存储第一个 odom 姿态
    if (!has_first_odom_) {
      first_odom_pose_ = latest_odom_pose_;
      has_first_odom_ = true;
      RCLCPP_INFO(this->get_logger(),
                   "First odom pose: (%.3f, %.3f, %.1f°)",
                   first_odom_pose_.position.x,
                   first_odom_pose_.position.y,
                   std::atan2(2.0*(first_odom_pose_.orientation.w*first_odom_pose_.orientation.z + first_odom_pose_.orientation.x*first_odom_pose_.orientation.y),
                              1.0 - 2.0*(first_odom_pose_.orientation.y*first_odom_pose_.orientation.y + first_odom_pose_.orientation.z*first_odom_pose_.orientation.z)) * 180.0 / M_PI);
    }

    // 将 body 云变换到 odom 系，然后送入累积器
    if (has_tf_) {
      try {
        auto tf_msg = tf_buffer_->lookupTransform(
            odom_frame_, msg->header.frame_id,
            msg->header.stamp, tf2::durationFromSec(0.1));

        Eigen::Affine3f T = Eigen::Affine3f::Identity();
        T.translation() << tf_msg.transform.translation.x,
                           tf_msg.transform.translation.y,
                           tf_msg.transform.translation.z;
        Eigen::Quaternionf q(
            tf_msg.transform.rotation.w,
            tf_msg.transform.rotation.x,
            tf_msg.transform.rotation.y,
            tf_msg.transform.rotation.z);
        T.rotate(q);

        pcl::PointCloud<pcl::PointXYZI> cloud_in_odom;
        pcl::transformPointCloud(*pcl_cloud, cloud_in_odom, T);

        auto cloud_ros = std::make_shared<sensor_msgs::msg::PointCloud2>();
        pcl::toROSMsg(cloud_in_odom, *cloud_ros);
        cloud_ros->header = msg->header;
        cloud_ros->header.frame_id = odom_frame_;
        accumulator_->addFrame(cloud_ros, latest_odom_pose_, msg->header.stamp);
        return;
      } catch (const tf2::TransformException& ex) {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                             "TF failed: %s", ex.what());
      }
    }

    accumulator_->addFrame(msg, latest_odom_pose_, msg->header.stamp);
  }

  // ─── 里程计回调 ───
  void odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg)
  {
    latest_odom_pose_ = msg->pose.pose;
    has_odom_ = true;
  }

  // ─── 重定位定时器回调 ───
  void relocalizeTimerCallback()
  {
    if (is_localized_) {
      if (publish_tf_) {
        publishTF();
      }
      if (!publish_tf_) {
        relocalize_timer_->cancel();
        RCLCPP_INFO(this->get_logger(),
                     "Relocalization complete, timer stopped.");
      }
      return;
    }

    // ─── 旋转采集阶段：等待旋转完成 ───
    if (rotate_enable_ && rot_phase_ != RotationPhase::DONE) {
      if (rot_phase_ == RotationPhase::IDLE) {
        startRotation();
      }
      // 旋转过程中不执行重定位（由 rotation_timer_ 驱动）
      return;
    }

    tryRelocalize();
  }

  // ─── 执行重定位 ───
  void tryRelocalize()
  {
    if (!has_odom_) {
      RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                           "Waiting for odom before relocalization");
      publishStatus("WAITING_FOR_ODOM");
      return;
    }

    // 用累积云（odom 系，多帧融合更密）
    auto current_cloud = accumulator_->getAccumulatedCloud(
        accum_frame_count_, accum_max_time_s_, accum_voxel_size_);

    if (!current_cloud || current_cloud->empty()) {
      // 回退到最新单帧
      if (latest_cloud_ && !latest_cloud_->empty()) {
        current_cloud = latest_cloud_;
        RCLCPP_INFO(this->get_logger(), "Fallback to single frame (%zu pts)", current_cloud->size());
      } else {
        RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                             "Waiting for cloud before relocalization");
        publishStatus("WAITING_FOR_CLOUD");
        return;
      }
    } else {
      RCLCPP_INFO(this->get_logger(), "Using accumulated cloud (%zu pts)", current_cloud->size());
    }

    if (current_cloud->size() < static_cast<size_t>(min_relocalize_points_)) {
      RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                           "Accumulated cloud has %zu pts, waiting for at least %d pts",
                           current_cloud->size(), min_relocalize_points_);
      publishStatus("WAITING_FOR_DENSE_CLOUD");
      return;
    }

    auto map_cloud = map_loader_->getDownsampledMap();
    if (!map_cloud || map_cloud->empty()) {
      RCLCPP_ERROR(this->get_logger(), "Map cloud is empty");
      publishStatus("MAP_EMPTY");
      return;
    }

    RCLCPP_INFO(this->get_logger(),
                 "Starting relocalization (attempt %d/%d)...",
                 retry_count_ + 1, max_retry_);
    publishStatus("RELOCALIZING");

    // 计算搜索中心：累积云的质心（odom 系）
    Eigen::Vector3f centroid = Eigen::Vector3f::Zero();
    for (const auto& pt : *current_cloud) {
      centroid += pt.getVector3fMap();
    }
    centroid /= static_cast<float>(current_cloud->size());
    Eigen::Vector3f search_center = centroid;

    RCLCPP_INFO(this->get_logger(),
                 "Search center (odom): (%.3f, %.3f, %.3f) radius=%.1fm",
                 search_center.x(), search_center.y(), search_center.z(),
                 search_window_radius_);

    auto result = core_->relocalize(current_cloud, map_cloud,
                                     search_center,
                                     static_cast<float>(search_window_radius_));
    if (!result.has_value() || !result->success) {
      retry_count_++;
      if (retry_count_ >= max_retry_) {
        RCLCPP_WARN(this->get_logger(),
                     "Relocalization failed after %d attempts", max_retry_);
        publishStatus("FAILED");
        relocalize_timer_->cancel();
      } else {
        RCLCPP_WARN(this->get_logger(),
                     "Relocalization failed, will retry (%d/%d)",
                     retry_count_, max_retry_);
        publishStatus("RETRYING");
      }
      return;
    }

    // ─── fitness 质量门控 ───
    double fitness = result->fitness_score;
    if (fitness > fitness_thresh_) {
      RCLCPP_WARN(this->get_logger(),
                   "Relocalization fitness=%.4f > threshold=%.4f, "
                   "result rejected (degenerate environment?)",
                   fitness, fitness_thresh_);
      retry_count_++;
      if (retry_count_ >= max_retry_) {
        publishStatus("FAILED_LOW_FITNESS");
        relocalize_timer_->cancel();
      } else {
        publishStatus("RETRYING_LOW_FITNESS");
      }
      return;
    }

    // ─── 成功 ───
    is_localized_ = true;

    // 累积云在 odom 系，KISS-Matcher 直接返回 T_map_odom
    T_map_odom_ = result->T_map_odom;

    RCLCPP_INFO(this->get_logger(),
                 "T_map_odom=(%.3f, %.3f, %.3f) (src=%d, tgt=%d)",
                 T_map_odom_.translation().x(), T_map_odom_.translation().y(), T_map_odom_.translation().z(),
                 result->src_points, result->tgt_points);

    RCLCPP_INFO(this->get_logger(),
                 "Relocalization SUCCEEDED! fitness=%.4f "
                 "translation=%.3fm rotation=%.1fdeg "
                 "kiss_inliers(rot=%d, trans=%d) used_gicp=%s "
                 "(src=%d pts, tgt=%d pts)",
                 fitness, result->translation_m, result->rotation_deg,
                 result->kiss_rotation_inliers, result->kiss_translation_inliers,
                 result->used_gicp ? "true" : "false",
                 result->src_points, result->tgt_points);
    publishStatus("LOCALIZED");

    // 发布 /initialpose
    if (publish_initial_pose_) {
      publishInitialPose(T_map_odom_, fitness);
    }

    // 发布 TF（仅在明确启用时）
    if (publish_tf_) {
      publishTF();
    }
  }

  // ═══════════════════════════════════════════════════════════════
  // ─── 原地旋转采集（含点云避障）───
  // ═══════════════════════════════════════════════════════════════

  /// 启动旋转采集：清空累积器并开始旋转
  void startRotation()
  {
    if (!has_odom_) {
      RCLCPP_WARN(this->get_logger(),
                   "[旋转采集] 无 odom 数据，跳过旋转");
      rot_phase_ = RotationPhase::DONE;
      return;
    }

    accumulator_->clear();
    rotation_yaw_init_ = false;
    total_rotated_angle_ = 0.0;
    last_rotation_yaw_ = getYawFromOdom();
    rotation_start_yaw_ = last_rotation_yaw_;

    rot_phase_ = RotationPhase::ROTATING;

    RCLCPP_INFO(this->get_logger(),
                 "[旋转采集] 开始原地旋转 %.0f° (%.2f rad/s)，"
                 "避障=%s (检测范围±%.0f°, 最小距离%.2fm)",
                 rotate_total_angle_deg_,
                 std::abs(rotate_angular_speed_),
                 rotate_obstacle_check_ ? "ON" : "OFF",
                 rotate_obstacle_check_range_deg_,
                 rotate_obstacle_min_dist_);
    publishStatus("ROTATING");
  }

  /// 旋转控制回调（25Hz）
  void rotationControlCallback()
  {
    if (rot_phase_ == RotationPhase::IDLE ||
        rot_phase_ == RotationPhase::DONE) {
      return;
    }

    if (!has_odom_) return;

    if (rot_phase_ == RotationPhase::ROTATING) {
      // ── 步骤1：避障检查 ──
      if (rotate_obstacle_check_ &&
          latest_cloud_ && !latest_cloud_->empty()) {
        if (checkObstacleInRotationDirection()) {
          RCLCPP_WARN(this->get_logger(),
                       "[旋转避障] 旋转方向±%.0f° 内检测到障碍物 (距离<%.2fm)，"
                       "停止旋转",
                       rotate_obstacle_check_range_deg_,
                       rotate_obstacle_min_dist_);
          publishCmdVel(0.0);
          rot_phase_ = RotationPhase::STOPPING;
          rotation_stop_time_ = this->now();
          publishStatus("ROTATION_OBSTACLE_STOP");
          return;
        }
      }

      // ── 步骤2：累计旋转角度 ──
      double current_yaw = getYawFromOdom();
      if (!std::isnan(last_rotation_yaw_) && !std::isnan(current_yaw)) {
        double delta = current_yaw - last_rotation_yaw_;
        // 角度归一化到 [-π, π]
        while (delta > M_PI)  delta -= 2.0 * M_PI;
        while (delta < -M_PI) delta += 2.0 * M_PI;
        total_rotated_angle_ += std::abs(delta);
      }
      last_rotation_yaw_ = current_yaw;

      // ── 步骤3：检查是否完成 ──
      double target_rad = rotate_total_angle_deg_ * M_PI / 180.0;
      if (total_rotated_angle_ >= target_rad) {
        RCLCPP_INFO(this->get_logger(),
                     "[旋转采集] 完成 %.0f° 旋转，累积 %zu 帧点云",
                     total_rotated_angle_ * 180.0 / M_PI,
                     accumulator_->bufferSize());
        publishCmdVel(0.0);
        rot_phase_ = RotationPhase::STOPPING;
        rotation_stop_time_ = this->now();
        publishStatus("ROTATION_COMPLETE");
        return;
      }

      // ── 步骤4：继续旋转 ──
      publishCmdVel(rotate_angular_speed_);
    }
    else if (rot_phase_ == RotationPhase::STOPPING) {
      // 等待机器人完全停止（1.5 秒缓冲）
      double elapsed = (this->now() - rotation_stop_time_).seconds();
      if (elapsed > 1.5) {
        rot_phase_ = RotationPhase::DONE;
        RCLCPP_INFO(this->get_logger(),
                     "[旋转采集] 已停止，进入重定位阶段 "
                     "(累积云 %zu 帧)",
                     accumulator_->bufferSize());
        publishStatus("ROTATION_DONE");
      }
    }
  }

  /// 从 odom 四元数提取 yaw 角
  double getYawFromOdom()
  {
    const auto& q = latest_odom_pose_.orientation;
    return std::atan2(2.0 * (q.w * q.z + q.x * q.y),
                       1.0 - 2.0 * (q.y * q.y + q.z * q.z));
  }

  /// 发布角速度到 /cmd_vel
  void publishCmdVel(double angular_z)
  {
    if (!cmd_vel_pub_) return;
    auto msg = std::make_unique<geometry_msgs::msg::Twist>();
    msg->angular.z = angular_z;
    // linear 全为 0（纯原地旋转）
    cmd_vel_pub_->publish(std::move(msg));
  }

  /// 基于最新点云检测旋转方向是否有障碍物
  /// @return true=有障碍物，应立即停止旋转
  bool checkObstacleInRotationDirection()
  {
    if (!latest_cloud_ || latest_cloud_->empty()) return false;

    // 旋转方向符号（正=CCW/左转，负=CW/右转）
    double dir_sign = (rotate_angular_speed_ > 0.0) ? 1.0 : -1.0;

    // 检测扇形角度范围（相对于机器人正前方 X 轴）
    double half_angle = rotate_obstacle_check_range_deg_ * M_PI / 180.0 / 2.0;
    double min_dist2 = rotate_obstacle_min_dist_ * rotate_obstacle_min_dist_;

    // 危险区域：机器人前方旋转方向侧的扇形
    // CCW 旋转 → 检查 [0, half_angle]（右侧/正前）
    // CW  旋转 → 检查 [-half_angle, 0]（左侧/正前）
    double angle_lo = std::min(0.0, dir_sign * half_angle);
    double angle_hi = std::max(0.0, dir_sign * half_angle);

    for (const auto& pt : *latest_cloud_) {
      // 过滤非有限值
      if (!std::isfinite(pt.x) || !std::isfinite(pt.y)) continue;

      double d2 = pt.x * pt.x + pt.y * pt.y;
      if (d2 > min_dist2 || d2 < 0.005) continue;  // 太远或太近（噪声）

      double angle = std::atan2(pt.y, pt.x);
      if (angle >= angle_lo && angle <= angle_hi) {
        RCLCPP_DEBUG(this->get_logger(),
                      "[旋转避障] 障碍点: (%.2f, %.2f) dist=%.2fm angle=%.1f°",
                      pt.x, pt.y, std::sqrt(d2), angle * 180.0 / M_PI);
        return true;
      }
    }

    return false;
  }

  // ─── 发布 map→odom TF ───
  void publishTF()
  {
    if (!publish_tf_ || !tf_broadcaster_) return;

    geometry_msgs::msg::TransformStamped tf;
    tf.header.stamp       = this->now();
    tf.header.frame_id    = map_frame_;
    tf.child_frame_id     = odom_frame_;

    const auto& T = T_map_odom_;
    tf.transform.translation.x = T.translation().x();
    tf.transform.translation.y = T.translation().y();
    tf.transform.translation.z = T.translation().z();

    Eigen::Quaternionf q(T.rotation());
    tf.transform.rotation.x = q.x();
    tf.transform.rotation.y = q.y();
    tf.transform.rotation.z = q.z();
    tf.transform.rotation.w = q.w();

    tf_broadcaster_->sendTransform(tf);
  }

  // ─── 发布 /initialpose（支持 2D/3D 模式）───
  void publishInitialPose(const Eigen::Isometry3f& T, double fitness)
  {
    geometry_msgs::msg::PoseWithCovarianceStamped msg;
    msg.header.stamp    = this->now();
    msg.header.frame_id = map_frame_;

    bool is_3d = (initial_pose_mode_ == "3d");
    auto map_pose = MapAlignment::transformPose(T, alignment_transform_);

    if (is_3d) {
      // ─── 3D 模式：直接使用 PCD 坐标 + 偏移 ───
      msg.pose.pose.position.x = map_pose.x;
      msg.pose.pose.position.y = map_pose.y;
      msg.pose.pose.position.z = T.translation().z();

      Eigen::Quaternionf q(T.rotation());
      Eigen::AngleAxisf aa(T.rotation());
      Eigen::Quaternionf q_offset(Eigen::AngleAxisf(static_cast<float>(alignment_transform_.yaw), Eigen::Vector3f::UnitZ()));
      Eigen::Quaternionf q_final = q_offset * q;

      msg.pose.pose.orientation.x = q_final.x();
      msg.pose.pose.orientation.y = q_final.y();
      msg.pose.pose.orientation.z = q_final.z();
      msg.pose.pose.orientation.w = q_final.w();
    } else {
      // ─── 2D 模式：只用 x, y, yaw ───
      msg.pose.pose.position.x = map_pose.x;
      msg.pose.pose.position.y = map_pose.y;
      msg.pose.pose.position.z = 0.0;

      Eigen::Quaternionf q(Eigen::AngleAxisf(static_cast<float>(map_pose.yaw), Eigen::Vector3f::UnitZ()));
      msg.pose.pose.orientation.x = q.x();
      msg.pose.pose.orientation.y = q.y();
      msg.pose.pose.orientation.z = q.z();
      msg.pose.pose.orientation.w = q.w();
    }

    // 自适应协方差：fitness 越小（越好），协方差越小
    double pos_sigma = std::min(fitness * 0.5 + 0.02, 0.5);
    double yaw_sigma = std::min(fitness * 0.3 + 0.05, 0.3);

    for (int i = 0; i < 36; ++i) msg.pose.covariance[i] = 0.0;
    msg.pose.covariance[0]  = pos_sigma * pos_sigma;   // x
    msg.pose.covariance[7]  = pos_sigma * pos_sigma;   // y
    msg.pose.covariance[14] = 0.10;                     // z (unused)
    msg.pose.covariance[21] = 0.01;                     // roll (unused)
    msg.pose.covariance[28] = 0.01;                     // pitch (unused)
    msg.pose.covariance[35] = yaw_sigma * yaw_sigma;   // yaw

    initial_pose_pub_->publish(msg);

    Eigen::Quaternionf q_out(msg.pose.pose.orientation.w, msg.pose.pose.orientation.x,
                              msg.pose.pose.orientation.y, msg.pose.pose.orientation.z);
    double yaw_out = std::atan2(2.0*(q_out.w()*q_out.z() + q_out.x()*q_out.y()),
                                 1.0 - 2.0*(q_out.y()*q_out.y() + q_out.z()*q_out.z()));
    RCLCPP_INFO(this->get_logger(),
                 "Published /initialpose [%s]: "
                 "2D=(%.3f, %.3f, %.1f°) 3D=(%.3f, %.3f, %.3f, %.1f°) "
                 "PCD_raw=(%.3f, %.3f) align=(%s scale=%.4f yaw=%.1fdeg "
                 "tx=%.2f ty=%.2f rms=%.3f) cov=(%.4f, %.4f) fitness=%.4f",
                 initial_pose_mode_.c_str(),
                 msg.pose.pose.position.x, msg.pose.pose.position.y,
                 yaw_out * 180.0 / M_PI,
                 msg.pose.pose.position.x, msg.pose.pose.position.y,
                 T.translation().z(), yaw_out * 180.0 / M_PI,
                 T.translation().x(), T.translation().y(),
                 alignment_mode_.c_str(),
                 alignment_transform_.scale,
                 alignment_transform_.yaw * 180.0 / M_PI,
                 alignment_transform_.tx,
                 alignment_transform_.ty,
                 alignment_transform_.rms_error,
                 pos_sigma, yaw_sigma, fitness);
  }

  // ─── 发布状态字符串 ───
  void publishStatus(const std::string& status)
  {
    auto msg = std::make_unique<std_msgs::msg::String>();
    msg->data = status;
    status_pub_->publish(std::move(msg));
  }
};

}  // namespace lidar_3d_relocalizer

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<lidar_3d_relocalizer::Lidar3dRelocalizerNode>(
      rclcpp::NodeOptions());
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}

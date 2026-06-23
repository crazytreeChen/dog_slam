#pragma once
/**
 * 多帧点云累积器
 * 订阅 /cloud_registered_body，用 odom 位姿拼接多帧点云
 */

#include <mutex>
#include <deque>
#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

namespace lidar_3d_relocalizer
{

using PointT = pcl::PointXYZI;
using PointCloudT = pcl::PointCloud<PointT>;

struct StampedCloud
{
  rclcpp::Time timestamp;
  typename PointCloudT::Ptr cloud;    // odom 系下的点云
  geometry_msgs::msg::Pose pose;        // 对应时刻的 odom 位姿
};

class CloudAccumulator
{
public:
  CloudAccumulator() = default;

  /**
   * @brief 添加一帧点云 + 对应的 odom 位姿
   * 线程安全
   */
  void addFrame(const sensor_msgs::msg::PointCloud2::SharedPtr cloud_msg,
                const geometry_msgs::msg::Pose& odom_pose,
                const rclcpp::Time& timestamp);

  /**
   * @brief 获取累积拼接后的点云（变换到第一帧坐标系）
   * @param max_frames  最多使用多少帧
   * @param max_time_s 时间窗口（秒）
   * @param voxel_size  降采样分辨率（米）
   * @return 拼接后的点云，失败时返回 nullptr
   */
  typename PointCloudT::Ptr getAccumulatedCloud(int max_frames,
                                                double max_time_s,
                                                float voxel_size = 0.05f);

  /// 清空缓冲区
  void clear();

  /// 当前缓冲区大小
  size_t bufferSize() const;

private:
  mutable std::mutex mutex_;
  std::deque<StampedCloud> buffer_;

  /// 将 odom 位姿转换为 4x4 变换矩阵
  Eigen::Matrix4f poseToMatrix(const geometry_msgs::msg::Pose& pose) const;

  /// 将 PointCloud2 转为 PCL PointCloud（假设 odom 系）
  typename PointCloudT::Ptr pointCloud2ToPcl(
      const sensor_msgs::msg::PointCloud2::SharedPtr cloud_msg) const;
};

}  // namespace lidar_3d_relocalizer

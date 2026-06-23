#include "lidar_3d_relocalizer/cloud_accumulator.hpp"
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/conversions.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <Eigen/Geometry>

namespace lidar_3d_relocalizer
{

using PointT = pcl::PointXYZI;
using PointCloudT = pcl::PointCloud<PointT>;

void CloudAccumulator::addFrame(
    const sensor_msgs::msg::PointCloud2::SharedPtr cloud_msg,
    const geometry_msgs::msg::Pose& odom_pose,
    const rclcpp::Time& timestamp)
{
  std::lock_guard<std::mutex> lock(mutex_);

  auto cloud = pointCloud2ToPcl(cloud_msg);
  if (!cloud || cloud->empty()) return;

  buffer_.push_back({timestamp, cloud, odom_pose});
}

typename PointCloudT::Ptr CloudAccumulator::pointCloud2ToPcl(
    const sensor_msgs::msg::PointCloud2::SharedPtr cloud_msg) const
{
  auto out = std::make_shared<PointCloudT>();
  pcl::fromROSMsg(*cloud_msg, *out);
  return out;
}

Eigen::Matrix4f CloudAccumulator::poseToMatrix(
    const geometry_msgs::msg::Pose& pose) const
{
  Eigen::Isometry3f T = Eigen::Isometry3f::Identity();
  T.translation()  << pose.position.x, pose.position.y, pose.position.z;
  Eigen::Quaternionf q(
      pose.orientation.w,
      pose.orientation.x,
      pose.orientation.y,
      pose.orientation.z);
  T.rotate(q);
  return T.matrix();
}

typename PointCloudT::Ptr CloudAccumulator::getAccumulatedCloud(
    int max_frames, double max_time_s, float voxel_size)
{
  std::lock_guard<std::mutex> lock(mutex_);
  if (buffer_.empty()) return nullptr;

  // 选取时间窗口内的帧
  auto now = buffer_.back().timestamp;
  size_t start_idx = 0;
  for (size_t i = 0; i < buffer_.size(); ++i)
  {
    double dt = (now - buffer_[i].timestamp).seconds();
    if (dt <= max_time_s) { start_idx = i; break; }
  }

  size_t count = std::min(static_cast<size_t>(max_frames),
                           buffer_.size() - start_idx);
  if (count == 0) return nullptr;

  // 直接合并 odom 系的云（不做帧间对齐，保持全局坐标）
  auto accum = std::make_shared<PointCloudT>();

  for (size_t i = start_idx; i < start_idx + count; ++i)
  {
    *accum += *(buffer_[i].cloud);
  }

  // 降采样
  if (voxel_size > 1e-4f && !accum->empty())
  {
    pcl::VoxelGrid<PointT> vg;
    vg.setInputCloud(accum);
    vg.setLeafSize(voxel_size, voxel_size, voxel_size);
    PointCloudT ds;
    vg.filter(ds);
    *accum = ds;
  }

  return accum;
}

void CloudAccumulator::clear()
{
  std::lock_guard<std::mutex> lock(mutex_);
  buffer_.clear();
}

size_t CloudAccumulator::bufferSize() const
{
  std::lock_guard<std::mutex> lock(mutex_);
  return buffer_.size();
}

}  // namespace lidar_3d_relocalizer

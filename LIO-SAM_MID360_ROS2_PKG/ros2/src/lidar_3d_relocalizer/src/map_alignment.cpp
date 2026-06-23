#include "lidar_3d_relocalizer/map_alignment.hpp"

#include <cmath>
#include <limits>

namespace lidar_3d_relocalizer
{
namespace
{

double normalizeAngle(double angle)
{
  while (angle > M_PI) angle -= 2.0 * M_PI;
  while (angle < -M_PI) angle += 2.0 * M_PI;
  return angle;
}

}  // namespace

AlignmentTransform MapAlignment::estimateFromLandmarks(
    const std::vector<AlignmentPoint>& points,
    std::string* error_message)
{
  AlignmentTransform transform;
  if (points.size() < 2) {
    if (error_message) *error_message = "at least two alignment points are required";
    return transform;
  }

  Eigen::Vector2d pcd_center = Eigen::Vector2d::Zero();
  Eigen::Vector2d map_center = Eigen::Vector2d::Zero();
  for (const auto& point : points) {
    pcd_center += Eigen::Vector2d(point.pcd_x, point.pcd_y);
    map_center += Eigen::Vector2d(point.map_x, point.map_y);
  }
  pcd_center /= static_cast<double>(points.size());
  map_center /= static_cast<double>(points.size());

  double a = 0.0;
  double b = 0.0;
  double denom = 0.0;
  for (const auto& point : points) {
    const Eigen::Vector2d p(point.pcd_x, point.pcd_y);
    const Eigen::Vector2d q(point.map_x, point.map_y);
    const Eigen::Vector2d pc = p - pcd_center;
    const Eigen::Vector2d qc = q - map_center;

    a += pc.x() * qc.x() + pc.y() * qc.y();
    b += pc.x() * qc.y() - pc.y() * qc.x();
    denom += pc.squaredNorm();
  }

  if (denom <= std::numeric_limits<double>::epsilon()) {
    if (error_message) *error_message = "pcd alignment points are degenerate";
    return transform;
  }

  const double scale_cos = a / denom;
  const double scale_sin = b / denom;
  transform.scale = std::hypot(scale_cos, scale_sin);
  if (transform.scale <= std::numeric_limits<double>::epsilon()) {
    if (error_message) *error_message = "estimated alignment scale is zero";
    return transform;
  }

  transform.yaw = std::atan2(scale_sin, scale_cos);
  const Eigen::Rotation2Dd rotation(transform.yaw);
  const Eigen::Vector2d translation = map_center - transform.scale * rotation * pcd_center;
  transform.tx = translation.x();
  transform.ty = translation.y();

  double squared_error_sum = 0.0;
  for (const auto& point : points) {
    const Eigen::Vector2d p(point.pcd_x, point.pcd_y);
    const Eigen::Vector2d q(point.map_x, point.map_y);
    const Eigen::Vector2d q_est = transform.scale * rotation * p + translation;
    squared_error_sum += (q_est - q).squaredNorm();
  }
  transform.rms_error = std::sqrt(squared_error_sum / static_cast<double>(points.size()));
  transform.valid = true;

  if (error_message) error_message->clear();
  return transform;
}

AlignmentTransform MapAlignment::fromOffset(double offset_x,
                                            double offset_y,
                                            double offset_yaw_deg)
{
  AlignmentTransform transform;
  transform.scale = 1.0;
  transform.yaw = offset_yaw_deg * M_PI / 180.0;
  transform.tx = offset_x;
  transform.ty = offset_y;
  transform.rms_error = 0.0;
  transform.valid = true;
  return transform;
}

PlanarPose MapAlignment::transformPose(const Eigen::Isometry3f& pcd_pose,
                                       const AlignmentTransform& transform)
{
  const Eigen::Vector2d pcd_xy(pcd_pose.translation().x(), pcd_pose.translation().y());
  const Eigen::Rotation2Dd rotation(transform.yaw);
  const Eigen::Vector2d map_xy =
      transform.scale * rotation * pcd_xy + Eigen::Vector2d(transform.tx, transform.ty);

  const Eigen::Matrix3f pcd_rotation = pcd_pose.rotation();
  const double pcd_yaw = std::atan2(pcd_rotation(1, 0), pcd_rotation(0, 0));

  PlanarPose pose;
  pose.x = map_xy.x();
  pose.y = map_xy.y();
  pose.yaw = normalizeAngle(pcd_yaw + transform.yaw);
  return pose;
}

}  // namespace lidar_3d_relocalizer

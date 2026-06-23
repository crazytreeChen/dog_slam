#pragma once

#include <string>
#include <vector>

#include <Eigen/Geometry>

namespace lidar_3d_relocalizer
{

struct AlignmentPoint
{
  double pcd_x{0.0};
  double pcd_y{0.0};
  double map_x{0.0};
  double map_y{0.0};
};

struct PlanarPose
{
  double x{0.0};
  double y{0.0};
  double yaw{0.0};
};

struct AlignmentTransform
{
  double scale{1.0};
  double yaw{0.0};
  double tx{0.0};
  double ty{0.0};
  double rms_error{0.0};
  bool valid{false};
};

class MapAlignment
{
public:
  static AlignmentTransform estimateFromLandmarks(
      const std::vector<AlignmentPoint>& points,
      std::string* error_message = nullptr);

  static AlignmentTransform estimateFromLandmarks(
      const std::vector<AlignmentPoint>& points,
      bool allow_scale,
      std::string* error_message = nullptr);

  static AlignmentTransform fromOffset(double offset_x,
                                       double offset_y,
                                       double offset_yaw_deg);

  static PlanarPose transformPose(const Eigen::Isometry3f& pcd_pose,
                                  const AlignmentTransform& transform);
};

}  // namespace lidar_3d_relocalizer

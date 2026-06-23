#include "lidar_3d_relocalizer/map_alignment.hpp"

#include <cmath>
#include <gtest/gtest.h>

namespace lidar_3d_relocalizer
{
namespace
{

constexpr double kTolerance = 1e-6;

Eigen::Isometry3f makePose(double x, double y, double yaw)
{
  Eigen::Isometry3f pose = Eigen::Isometry3f::Identity();
  pose.translation() << static_cast<float>(x), static_cast<float>(y), 0.0f;
  pose.rotate(Eigen::AngleAxisf(static_cast<float>(yaw), Eigen::Vector3f::UnitZ()));
  return pose;
}

TEST(MapAlignmentTest, EstimatesSimilarityTransformFromLandmarks)
{
  const double yaw = M_PI / 2.0;
  const std::vector<AlignmentPoint> points = {
    {0.0, 0.0, 10.0, -3.0},
    {1.0, 0.0, 10.0, -1.0},
    {0.0, 1.0, 8.0, -3.0},
  };

  std::string error;
  const auto transform = MapAlignment::estimateFromLandmarks(points, true, &error);

  EXPECT_TRUE(transform.valid) << error;
  EXPECT_NEAR(transform.scale, 2.0, kTolerance);
  EXPECT_NEAR(transform.yaw, yaw, kTolerance);
  EXPECT_NEAR(transform.tx, 10.0, kTolerance);
  EXPECT_NEAR(transform.ty, -3.0, kTolerance);
  EXPECT_NEAR(transform.rms_error, 0.0, kTolerance);
}

TEST(MapAlignmentTest, EstimatesRigidTransformFromNoisyLandmarks)
{
  const std::vector<AlignmentPoint> points = {
    {-1.65958, -3.69628, -3.17743, -3.71891},
    {-11.11288, -11.18596, -10.42140, -10.11060},
    {16.66528, -6.13897, 16.47700, -6.96873},
  };

  std::string error;
  const auto transform = MapAlignment::estimateFromLandmarks(points, false, &error);

  EXPECT_TRUE(transform.valid) << error;
  EXPECT_NEAR(transform.scale, 1.0, kTolerance);
  EXPECT_NEAR(transform.yaw, -2.471985 * M_PI / 180.0, 1e-6);
  EXPECT_NEAR(transform.tx, -0.034788024, 1e-6);
  EXPECT_NEAR(transform.ty, 0.123769691, 1e-6);
  EXPECT_GT(transform.rms_error, 0.0);
}

TEST(MapAlignmentTest, EstimatesTranslationOnlyTransformFromSingleLandmark)
{
  const std::vector<AlignmentPoint> points = {
    {-1.659582, -3.696284, -3.17743, -3.71891},
  };

  std::string error;
  const auto transform = MapAlignment::estimateFromLandmarks(points, false, &error);

  EXPECT_TRUE(transform.valid) << error;
  EXPECT_NEAR(transform.scale, 1.0, kTolerance);
  EXPECT_NEAR(transform.yaw, 0.0, kTolerance);
  EXPECT_NEAR(transform.tx, -1.517848, kTolerance);
  EXPECT_NEAR(transform.ty, -0.022626, kTolerance);
  EXPECT_NEAR(transform.rms_error, 0.0, kTolerance);
}

TEST(MapAlignmentTest, TransformsPcdPoseIntoMapPose)
{
  const AlignmentTransform transform{
    2.0,
    M_PI / 2.0,
    10.0,
    -3.0,
    0.0,
    true,
  };
  const auto pcd_pose = makePose(1.0, 1.0, M_PI / 4.0);

  const auto map_pose = MapAlignment::transformPose(pcd_pose, transform);

  EXPECT_NEAR(map_pose.x, 8.0, kTolerance);
  EXPECT_NEAR(map_pose.y, -1.0, kTolerance);
  EXPECT_NEAR(map_pose.yaw, 3.0 * M_PI / 4.0, kTolerance);
}

TEST(MapAlignmentTest, RejectsDegenerateLandmarks)
{
  const std::vector<AlignmentPoint> points = {
    {1.0, 1.0, 3.0, 4.0},
    {1.0, 1.0, 5.0, 6.0},
  };

  std::string error;
  const auto transform = MapAlignment::estimateFromLandmarks(points, &error);

  EXPECT_FALSE(transform.valid);
  EXPECT_FALSE(error.empty());
}

}  // namespace
}  // namespace lidar_3d_relocalizer

#include "traversability_layer/traversability_layer.hpp"
#include "nav2_costmap_2d/cost_values.hpp"
#include "pluginlib/class_list_macros.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <functional>
#include <set>

PLUGINLIB_EXPORT_CLASS(traversability_layer::TraversabilityLayer, nav2_costmap_2d::Layer)

namespace traversability_layer
{

TraversabilityLayer::TraversabilityLayer()
: height_map_size_x_(0),
  height_map_size_y_(0),
  height_map_valid_(false)
{
}

TraversabilityLayer::~TraversabilityLayer()
{
}

void TraversabilityLayer::onInitialize()
{
  auto node = node_.lock();
  if (!node) {
    throw std::runtime_error("TraversabilityLayer: Unable to lock node!");
  }

  declareParameter("enabled", rclcpp::ParameterValue(true));
  declareParameter("pointcloud_topic", rclcpp::ParameterValue(std::string("/lio/body/cloud")));
  declareParameter("sensor_frame", rclcpp::ParameterValue(std::string("")));
  declareParameter("max_obstacle_height", rclcpp::ParameterValue(2.0));
  declareParameter("min_obstacle_height", rclcpp::ParameterValue(-0.5));
  declareParameter("max_slope_traversable", rclcpp::ParameterValue(45.0));
  declareParameter("slope_cost_start", rclcpp::ParameterValue(15.0));
  declareParameter("step_height_threshold", rclcpp::ParameterValue(0.15));
  declareParameter("slope_cost_scale", rclcpp::ParameterValue(5.0));
  declareParameter("height_cost_scale", rclcpp::ParameterValue(10.0));
  declareParameter("lethal_cost_threshold", rclcpp::ParameterValue(254.0));
  declareParameter("observation_persistence", rclcpp::ParameterValue(5.0));
  declareParameter("clearing_duration", rclcpp::ParameterValue(10.0));
  declareParameter("cloud_buffer_size", rclcpp::ParameterValue(5));
  declareParameter("publish_slope_map", rclcpp::ParameterValue(false));
  declareParameter("enable_raycast_clear", rclcpp::ParameterValue(true));

  node->get_parameter(name_ + ".enabled", enabled_);
  node->get_parameter(name_ + ".pointcloud_topic", pointcloud_topic_);
  node->get_parameter(name_ + ".sensor_frame", sensor_frame_);
  node->get_parameter(name_ + ".max_obstacle_height", max_obstacle_height_);
  node->get_parameter(name_ + ".min_obstacle_height", min_obstacle_height_);
  node->get_parameter(name_ + ".max_slope_traversable", max_slope_traversable_);
  node->get_parameter(name_ + ".slope_cost_start", slope_cost_start_);
  node->get_parameter(name_ + ".step_height_threshold", step_height_threshold_);
  node->get_parameter(name_ + ".slope_cost_scale", slope_cost_scale_);
  node->get_parameter(name_ + ".height_cost_scale", height_cost_scale_);
  node->get_parameter(name_ + ".lethal_cost_threshold", lethal_cost_threshold_);
  node->get_parameter(name_ + ".observation_persistence", observation_persistence_);
  node->get_parameter(name_ + ".clearing_duration", clearing_duration_);
  node->get_parameter(name_ + ".cloud_buffer_size", cloud_buffer_size_);
  node->get_parameter(name_ + ".publish_slope_map", publish_slope_map_);
  node->get_parameter(name_ + ".enable_raycast_clear", enable_raycast_clear_);

  max_slope_traversable_ = max_slope_traversable_ * M_PI / 180.0;
  slope_cost_start_ = slope_cost_start_ * M_PI / 180.0;

  RCLCPP_INFO(
    node->get_logger(),
    "TraversabilityLayer: step_height_threshold=%.3f, max_slope_traversable=%.1f deg, "
    "slope_cost_start=%.1f deg, pointcloud_topic=%s, observation_persistence=%.1fs, "
    "clearing_duration=%.1fs, enable_raycast_clear=%s",
    step_height_threshold_, max_slope_traversable_ * 180.0 / M_PI,
    slope_cost_start_ * 180.0 / M_PI, pointcloud_topic_.c_str(),
    observation_persistence_, clearing_duration_,
    enable_raycast_clear_ ? "true" : "false");

  tf_buffer_ = std::make_shared<tf2_ros::Buffer>(node->get_clock());
  tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

  rclcpp::QoS cloud_qos(rclcpp::KeepLast(static_cast<size_t>(cloud_buffer_size_)));
  cloud_qos.best_effort();
  cloud_sub_ = node->create_subscription<sensor_msgs::msg::PointCloud2>(
    pointcloud_topic_, cloud_qos,
    std::bind(&TraversabilityLayer::pointCloudCallback, this, std::placeholders::_1));

  if (publish_slope_map_) {
    slope_pub_ = node->create_publisher<sensor_msgs::msg::PointCloud2>(
      "slope_map", rclcpp::QoS(1));
  }

  // 让当前图层的网格尺寸、分辨率、原点与主地图（layered_costmap_）完全对齐
  matchSize();

  current_ = true;
}

void TraversabilityLayer::matchSize()
{
  // 获取主地图的属性
  nav2_costmap_2d::Costmap2D * master_grid = layered_costmap_->getCostmap();
  
  // 设置当前图层自身的 Costmap2D 尺寸
  resizeMap(master_grid->getSizeInCellsX(), master_grid->getSizeInCellsY(), 
            master_grid->getResolution(), 
            master_grid->getOriginX(), master_grid->getOriginY());
}

void TraversabilityLayer::pointCloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
{
  std::lock_guard<std::mutex> lock(mutex_);
  static auto clock = std::make_shared<rclcpp::Clock>(RCL_ROS_TIME);

  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::fromROSMsg(*msg, *cloud);

  RCLCPP_INFO_THROTTLE(
    rclcpp::get_logger("traversability_layer"), *clock, 2000,
    "[TraversabilityLayer] Received cloud: %zu points, frame_id=%s",
    cloud->size(), msg->header.frame_id.c_str());

  if (cloud->empty()) {
    RCLCPP_WARN(rclcpp::get_logger("traversability_layer"), "[TraversabilityLayer] Cloud is empty!");
    return;
  }

  std::string target_frame = layered_costmap_->getGlobalFrameID();
  std::string source_frame = msg->header.frame_id;

  if (!sensor_frame_.empty() && sensor_frame_ != "") {
    source_frame = sensor_frame_;
  }

  RCLCPP_INFO_THROTTLE(
    rclcpp::get_logger("traversability_layer"), *clock, 2000,
    "[TraversabilityLayer] TF lookup: target=%s, source=%s",
    target_frame.c_str(), source_frame.c_str());

  geometry_msgs::msg::TransformStamped transform;
  try {
    transform = tf_buffer_->lookupTransform(
      target_frame, source_frame, msg->header.stamp,
      rclcpp::Duration::from_seconds(0.5));
    RCLCPP_INFO_THROTTLE(
      rclcpp::get_logger("traversability_layer"), *clock, 2000,
      "[TraversabilityLayer] TF success: translation=[%.3f, %.3f, %.3f]",
      transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z);
  } catch (tf2::TransformException & ex) {
    RCLCPP_WARN_THROTTLE(
      rclcpp::get_logger("traversability_layer"), *clock, 2000,
      "[TraversabilityLayer] TF transform failed: %s (target=%s, source=%s)",
      ex.what(), target_frame.c_str(), source_frame.c_str());
    return;
  }

  double tx = transform.transform.translation.x;
  double ty = transform.transform.translation.y;
  double tz = transform.transform.translation.z;
  double qx = transform.transform.rotation.x;
  double qy = transform.transform.rotation.y;
  double qz = transform.transform.rotation.z;
  double qw = transform.transform.rotation.w;

  double r00 = 1.0 - 2.0 * (qy * qy + qz * qz);
  double r01 = 2.0 * (qx * qy - qz * qw);
  double r02 = 2.0 * (qx * qz + qy * qw);
  double r10 = 2.0 * (qx * qy + qz * qw);
  double r11 = 1.0 - 2.0 * (qx * qx + qz * qz);
  double r12 = 2.0 * (qy * qz - qx * qw);
  double r20 = 2.0 * (qx * qz - qy * qw);
  double r21 = 2.0 * (qy * qz + qx * qw);
  double r22 = 1.0 - 2.0 * (qx * qx + qy * qy);

  double sensor_x = tx;
  double sensor_y = ty;

  pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud(new pcl::PointCloud<pcl::PointXYZ>);
  transformed_cloud->reserve(cloud->size());

  int filtered_count = 0;
  for (const auto & pt : cloud->points) {
    if (!std::isfinite(pt.x) || !std::isfinite(pt.y) || !std::isfinite(pt.z)) {
      filtered_count++;
      continue;
    }

    double x_global = r00 * pt.x + r01 * pt.y + r02 * pt.z + tx;
    double y_global = r10 * pt.x + r11 * pt.y + r12 * pt.z + ty;
    double z_global = r20 * pt.x + r21 * pt.y + r22 * pt.z + tz;

    double z_robot_relative = z_global - tz;
    if (z_robot_relative > max_obstacle_height_ || z_robot_relative < min_obstacle_height_) {
      filtered_count++;
      continue;
    }

    transformed_cloud->push_back(
      pcl::PointXYZ(
        static_cast<float>(x_global),
        static_cast<float>(y_global),
        static_cast<float>(z_global)));
  }

  RCLCPP_INFO_THROTTLE(
    rclcpp::get_logger("traversability_layer"), *clock, 2000,
    "[TraversabilityLayer] After transform: %zu points kept, %d filtered (z_range=[%.2f, %.2f])",
    transformed_cloud->size(), filtered_count, min_obstacle_height_, max_obstacle_height_);

  processCloud(transformed_cloud, msg->header.stamp);
}

void TraversabilityLayer::processCloud(
  const pcl::PointCloud<pcl::PointXYZ>::Ptr & cloud,
  const rclcpp::Time & stamp)
{
  static auto clock = std::make_shared<rclcpp::Clock>(RCL_ROS_TIME);
  unsigned int size_x = getSizeInCellsX();
  unsigned int size_y = getSizeInCellsY();

  if (size_x == 0 || size_y == 0) {
    RCLCPP_WARN_THROTTLE(
      rclcpp::get_logger("traversability_layer"), *clock, 5000,
      "[TraversabilityLayer] Costmap size is zero! size_x=%u, size_y=%u", size_x, size_y);
    return;
  }

  RCLCPP_INFO_THROTTLE(
    rclcpp::get_logger("traversability_layer"), *clock, 2000,
    "[TraversabilityLayer] processCloud: cloud_size=%zu, costmap_size=%ux%u, resolution=%.3f",
    cloud->size(), size_x, size_y, getResolution());

  if (height_map_.size() != static_cast<size_t>(size_x) * size_y) {
    height_map_.resize(static_cast<size_t>(size_x) * size_y);
    height_map_size_x_ = size_x;
    height_map_size_y_ = size_y;
  }

  double ox = getOriginX();
  double oy = getOriginY();
  double res = getResolution();

  std::vector<std::pair<int, int>> marked_cells;
  marked_cells.reserve(cloud->size());

  std::set<size_t> updated_indices;
  int out_of_bounds_count = 0;

  for (const auto & pt : cloud->points) {
    int mx = static_cast<int>((pt.x - ox) / res);
    int my = static_cast<int>((pt.y - oy) / res);

    if (mx < 0 || mx >= static_cast<int>(size_x) ||
        my < 0 || my >= static_cast<int>(size_y))
    {
      out_of_bounds_count++;
      continue;
    }

    size_t idx = static_cast<size_t>(mx) + static_cast<size_t>(my) * size_x;

    if (updated_indices.find(idx) == updated_indices.end()) {
      auto & cell = height_map_[idx];
      cell.min_z = std::numeric_limits<float>::max();
      cell.max_z = std::numeric_limits<float>::lowest();
      cell.mean_z = 0.0f;
      cell.point_count = 0;
      cell.height_diff = 0.0f;
      cell.slope_x = 0.0f;
      cell.slope_y = 0.0f;
      cell.slope_magnitude = 0.0f;
      cell.has_data = false;
      cell.ray_cleared = false;
      updated_indices.insert(idx);
    }

    auto & cell = height_map_[idx];

    if (pt.z < cell.min_z) cell.min_z = pt.z;
    if (pt.z > cell.max_z) cell.max_z = pt.z;
    cell.mean_z += pt.z;
    cell.point_count++;
    cell.has_data = true;
    cell.last_seen_time = stamp;

    marked_cells.emplace_back(mx, my);
  }

  RCLCPP_INFO_THROTTLE(
    rclcpp::get_logger("traversability_layer"), *clock, 2000,
    "[TraversabilityLayer] Grid update: %zu cells marked, %d out_of_bounds, origin=[%.2f, %.2f]",
    marked_cells.size(), out_of_bounds_count, ox, oy);

  for (size_t idx : updated_indices) {
    auto & cell = height_map_[idx];
    if (cell.has_data && cell.point_count > 0) {
      cell.mean_z /= static_cast<float>(cell.point_count);
      cell.height_diff = cell.max_z - cell.min_z;
    }
  }

  geometry_msgs::msg::TransformStamped sensor_transform;
  std::string target_frame = layered_costmap_->getGlobalFrameID();
  std::string source_frame = sensor_frame_.empty() ? target_frame : sensor_frame_;

  try {
    sensor_transform = tf_buffer_->lookupTransform(
      target_frame, source_frame, stamp,
      rclcpp::Duration::from_seconds(0.1));
    double sx = sensor_transform.transform.translation.x;
    double sy = sensor_transform.transform.translation.y;
    if (enable_raycast_clear_) {
      raycastClear(sx, sy, marked_cells);
    }
  } catch (tf2::TransformException & ex) {
    RCLCPP_DEBUG(
      rclcpp::get_logger("traversability_layer"),
      "TraversabilityLayer: sensor TF for raycast failed: %s", ex.what());
  }

  decayOldObservations(stamp);

  computeSlope();
  height_map_valid_ = true;
}

void TraversabilityLayer::raycastClear(
  double sensor_x, double sensor_y,
  const std::vector<std::pair<int, int>> & marked_cells)
{
  unsigned int size_x = height_map_size_x_;
  unsigned int size_y = height_map_size_y_;
  double ox = getOriginX();
  double oy = getOriginY();
  double res = getResolution();

  int sensor_mx = static_cast<int>((sensor_x - ox) / res);
  int sensor_my = static_cast<int>((sensor_y - oy) / res);

  std::set<size_t> marked_set;
  for (const auto & mc : marked_cells) {
    marked_set.insert(static_cast<size_t>(mc.first) + static_cast<size_t>(mc.second) * size_x);
  }

  for (const auto & mc : marked_cells) {
    int end_mx = mc.first;
    int end_my = mc.second;

    int dx = std::abs(end_mx - sensor_mx);
    int dy = std::abs(end_my - sensor_my);
    int sx = (sensor_mx < end_mx) ? 1 : -1;
    int sy = (sensor_my < end_my) ? 1 : -1;
    int err = dx - dy;

    int cx = sensor_mx;
    int cy = sensor_my;

    while (true) {
      if (cx == end_mx && cy == end_my) {
        break;
      }

      if (cx >= 0 && cx < static_cast<int>(size_x) &&
          cy >= 0 && cy < static_cast<int>(size_y))
      {
        size_t idx = static_cast<size_t>(cx) + static_cast<size_t>(cy) * size_x;
        if (marked_set.find(idx) == marked_set.end()) {
          auto & cell = height_map_[idx];
          if (cell.has_data) {
            cell.has_data = false;
            cell.min_z = std::numeric_limits<float>::max();
            cell.max_z = std::numeric_limits<float>::lowest();
            cell.mean_z = 0.0f;
            cell.point_count = 0;
            cell.height_diff = 0.0f;
            cell.slope_x = 0.0f;
            cell.slope_y = 0.0f;
            cell.slope_magnitude = 0.0f;
            cell.ray_cleared = true;
          }
        }
      }

      int e2 = 2 * err;
      if (e2 > -dy) {
        err -= dy;
        cx += sx;
      }
      if (e2 < dx) {
        err += dx;
        cy += sy;
      }

      if (cx < -1 || cx > static_cast<int>(size_x) ||
          cy < -1 || cy > static_cast<int>(size_y))
      {
        break;
      }
    }
  }
}

void TraversabilityLayer::decayOldObservations(const rclcpp::Time & now)
{
  for (auto & cell : height_map_) {
    if (!cell.has_data) {
      continue;
    }

    double age_sec = (now - cell.last_seen_time).seconds();

    if (age_sec > clearing_duration_) {
      cell.has_data = false;
      cell.min_z = std::numeric_limits<float>::max();
      cell.max_z = std::numeric_limits<float>::lowest();
      cell.mean_z = 0.0f;
      cell.point_count = 0;
      cell.height_diff = 0.0f;
      cell.slope_x = 0.0f;
      cell.slope_y = 0.0f;
      cell.slope_magnitude = 0.0f;
    }
  }
}

void TraversabilityLayer::computeSlope()
{
  unsigned int size_x = height_map_size_x_;
  unsigned int size_y = height_map_size_y_;
  double res = getResolution();

  static auto clock = std::make_shared<rclcpp::Clock>(RCL_ROS_TIME);
  int cells_with_slope = 0;
  float max_slope_mag = 0.0f;
  float max_height_diff = 0.0f;

  for (unsigned int mx = 1; mx < size_x - 1; mx++) {
    for (unsigned int my = 1; my < size_y - 1; my++) {
      auto & cell = height_map_[mx + my * size_x];
      if (!cell.has_data || cell.point_count < 2) {
        continue;
      }

      auto & cell_xp = height_map_[(mx + 1) + my * size_x];
      auto & cell_xm = height_map_[(mx - 1) + my * size_x];
      auto & cell_yp = height_map_[mx + (my + 1) * size_x];
      auto & cell_ym = height_map_[mx + (my - 1) * size_x];

      float dz_dx = 0.0f;
      float dz_dy = 0.0f;
      int count_x = 0;
      int count_y = 0;

      if (cell_xp.has_data && cell_xm.has_data) {
        dz_dx = (cell_xp.mean_z - cell_xm.mean_z) / (2.0f * static_cast<float>(res));
        count_x = 2;
      } else if (cell_xp.has_data) {
        dz_dx = (cell_xp.mean_z - cell.mean_z) / static_cast<float>(res);
        count_x = 1;
      } else if (cell_xm.has_data) {
        dz_dx = (cell.mean_z - cell_xm.mean_z) / static_cast<float>(res);
        count_x = 1;
      }

      if (cell_yp.has_data && cell_ym.has_data) {
        dz_dy = (cell_yp.mean_z - cell_ym.mean_z) / (2.0f * static_cast<float>(res));
        count_y = 2;
      } else if (cell_yp.has_data) {
        dz_dy = (cell_yp.mean_z - cell.mean_z) / static_cast<float>(res);
        count_y = 1;
      } else if (cell_ym.has_data) {
        dz_dy = (cell.mean_z - cell_ym.mean_z) / static_cast<float>(res);
        count_y = 1;
      }

      if (count_x > 0 || count_y > 0) {
        cell.slope_x = dz_dx;
        cell.slope_y = dz_dy;
        cell.slope_magnitude = std::sqrt(dz_dx * dz_dx + dz_dy * dz_dy);
        
        if (cell.slope_magnitude > max_slope_mag) {
          max_slope_mag = cell.slope_magnitude;
        }
        if (cell.height_diff > max_height_diff) {
          max_height_diff = cell.height_diff;
        }
        cells_with_slope++;
      }
    }
  }

  RCLCPP_INFO_THROTTLE(
    rclcpp::get_logger("traversability_layer"), *clock, 2000,
    "[TraversabilityLayer] computeSlope: %d cells with slope, max_slope_mag=%.3f (angle=%.1f deg), max_height_diff=%.3f",
    cells_with_slope, max_slope_mag, std::atan(max_slope_mag) * 180.0 / M_PI, max_height_diff);
}

unsigned char TraversabilityLayer::computeCost(const CellData & cell) const
{
  if (!cell.has_data || cell.point_count < 2) {
    return nav2_costmap_2d::NO_INFORMATION;
  }

  unsigned char height_cost = 0;
  if (cell.height_diff > step_height_threshold_) {
    float excess = cell.height_diff - static_cast<float>(step_height_threshold_);
    float ratio = 1.0f - std::exp(-static_cast<float>(height_cost_scale_) * excess);
    height_cost = static_cast<unsigned char>(
      std::min(static_cast<float>(nav2_costmap_2d::LETHAL_OBSTACLE),
               ratio * static_cast<float>(lethal_cost_threshold_)));
  }

  unsigned char slope_cost = 0;
  float slope_angle = std::atan(cell.slope_magnitude);

  if (slope_angle > max_slope_traversable_) {
    slope_cost = nav2_costmap_2d::LETHAL_OBSTACLE;
  } else if (slope_angle > slope_cost_start_) {
    float excess_angle = slope_angle - static_cast<float>(slope_cost_start_);
    float max_excess = static_cast<float>(max_slope_traversable_) - static_cast<float>(slope_cost_start_);
    if (max_excess > 1e-6f) {
      float ratio = excess_angle / max_excess;
      float cost_f = (1.0f - std::exp(-static_cast<float>(slope_cost_scale_) * ratio)) *
                     static_cast<float>(lethal_cost_threshold_);
      slope_cost = static_cast<unsigned char>(
        std::min(static_cast<float>(nav2_costmap_2d::LETHAL_OBSTACLE), cost_f));
    }
  }

  unsigned char final_cost = std::max(height_cost, slope_cost);
  
  static int compute_debug_count = 0;
  if (final_cost == 0 && compute_debug_count < 10) {
    RCLCPP_INFO(
      rclcpp::get_logger("traversability_layer"),
      "【调试-阈值检查】cost=0: slope_angle=%.3f rad (%.1f deg), slope_cost_start=%.3f rad (%.1f deg), "
      "height_diff=%.3f, step_threshold=%.3f",
      slope_angle, slope_angle * 180.0 / M_PI,
      slope_cost_start_, slope_cost_start_ * 180.0 / M_PI,
      cell.height_diff, step_height_threshold_);
    compute_debug_count++;
  }

  return final_cost;
}

void TraversabilityLayer::updateBounds(
  double robot_x, double robot_y, double robot_yaw,
  double * min_x, double * min_y, double * max_x, double * max_y)
{
  if (!enabled_) {
    return;
  }

  std::lock_guard<std::mutex> lock(mutex_);
  static auto clock = std::make_shared<rclcpp::Clock>(RCL_ROS_TIME);

  unsigned int size_x = getSizeInCellsX();
  unsigned int size_y = getSizeInCellsY();

  if (size_x == 0 || size_y == 0) {
    RCLCPP_DEBUG_THROTTLE(
      rclcpp::get_logger("traversability_layer"), *clock, 5000,
      "[TraversabilityLayer] updateBounds: costmap size is zero, skipping");
    return;
  }

  if (!height_map_valid_) {
    return;
  }

  double ox = getOriginX();
  double oy = getOriginY();
  double res = getResolution();

  for (unsigned int mx = 0; mx < size_x; mx++) {
    for (unsigned int my = 0; my < size_y; my++) {
      const auto & cell = height_map_[mx + my * size_x];
      if (!cell.has_data) {
        continue;
      }

      double wx = ox + (mx + 0.5) * res;
      double wy = oy + (my + 0.5) * res;

      *min_x = std::min(*min_x, wx - res);
      *min_y = std::min(*min_y, wy - res);
      *max_x = std::max(*max_x, wx + res);
      *max_y = std::max(*max_y, wy + res);
    }
  }
}

void TraversabilityLayer::updateCosts(
  nav2_costmap_2d::Costmap2D & master_grid,
  int min_i, int min_j, int max_i, int max_j)
{
  if (!enabled_) {
    return;
  }

  std::lock_guard<std::mutex> lock(mutex_);
  static auto clock = std::make_shared<rclcpp::Clock>(RCL_ROS_TIME);

  if (!height_map_valid_) {
    RCLCPP_DEBUG_THROTTLE(
      rclcpp::get_logger("traversability_layer"), *clock, 5000,
      "[TraversabilityLayer] updateCosts: height_map not valid");
    return;
  }

  unsigned int size_x = getSizeInCellsX();
  unsigned int size_y = getSizeInCellsY();

  unsigned char * master_array = master_grid.getCharMap();

  int cells_with_cost = 0;
  int lethal_cells = 0;

  for (int mx = min_i; mx < max_i; mx++) {
    for (int my = min_j; my < max_j; my++) {
      if (mx < 0 || mx >= static_cast<int>(size_x) ||
          my < 0 || my >= static_cast<int>(size_y))
      {
        continue;
      }

      const auto & cell = height_map_[static_cast<unsigned int>(mx) +
                                       static_cast<unsigned int>(my) * size_x];
      if (!cell.has_data) {
        continue;
      }

      unsigned char cost = computeCost(cell);
      
      static int debug_print_count = 0;
      if (cost > 0 && debug_print_count < 20) {
        RCLCPP_INFO(
          rclcpp::get_logger("traversability_layer"),
          "【调试】成功计算出有效代价: cost=%d, slope_magnitude=%.3f, height_diff=%.3f, slope_angle=%.3f rad (%.1f deg)",
          cost, cell.slope_magnitude, cell.height_diff,
          std::atan(cell.slope_magnitude), std::atan(cell.slope_magnitude) * 180.0 / M_PI);
        debug_print_count++;
      }
      
      if (cost == nav2_costmap_2d::NO_INFORMATION) {
        continue;
      }

      if (cost >= nav2_costmap_2d::LETHAL_OBSTACLE) {
        lethal_cells++;
      }
      cells_with_cost++;

      unsigned int idx = master_grid.getIndex(mx, my);
      unsigned char old_cost = master_array[idx];

      if (old_cost == nav2_costmap_2d::NO_INFORMATION ||
          cost > old_cost) {
        master_array[idx] = cost;
      }
    }
  }

  RCLCPP_INFO_THROTTLE(
    rclcpp::get_logger("traversability_layer"), *clock, 2000,
    "[TraversabilityLayer] updateCosts: %d cells with cost, %d lethal (range=[%d,%d,%d,%d])",
    cells_with_cost, lethal_cells, min_i, min_j, max_i, max_j);

  if (publish_slope_map_ && slope_pub_) {
    auto node = node_.lock();
    if (node) {
      pcl::PointCloud<pcl::PointXYZI>::Ptr slope_cloud(new pcl::PointCloud<pcl::PointXYZI>);
      slope_cloud->header.frame_id = layered_costmap_->getGlobalFrameID();

      double ox = getOriginX();
      double oy = getOriginY();
      double res = getResolution();

      for (unsigned int mx = 0; mx < size_x; mx++) {
        for (unsigned int my = 0; my < size_y; my++) {
          const auto & cell = height_map_[mx + my * size_x];
          if (!cell.has_data) {
            continue;
          }

          pcl::PointXYZI pt;
          pt.x = static_cast<float>(ox + (mx + 0.5) * res);
          pt.y = static_cast<float>(oy + (my + 0.5) * res);
          pt.z = cell.mean_z;
          pt.intensity = cell.slope_magnitude;
          slope_cloud->push_back(pt);
        }
      }

      sensor_msgs::msg::PointCloud2 slope_msg;
      pcl::toROSMsg(*slope_cloud, slope_msg);
      slope_msg.header.stamp = node->now();
      slope_pub_->publish(slope_msg);

      RCLCPP_INFO_THROTTLE(
        rclcpp::get_logger("traversability_layer"), *clock, 2000,
        "[TraversabilityLayer] Published slope_map: %zu points, frame=%s",
        slope_cloud->size(), slope_cloud->header.frame_id.c_str());
    }
  } else {
    RCLCPP_DEBUG_THROTTLE(
      rclcpp::get_logger("traversability_layer"), *clock, 5000,
      "[TraversabilityLayer] slope_map not published: publish_slope_map_=%d, slope_pub_=%p",
      publish_slope_map_, slope_pub_.get());
  }
}

void TraversabilityLayer::reset()
{
  std::lock_guard<std::mutex> lock(mutex_);
  resetMaps();
}

void TraversabilityLayer::resetMaps()
{
  height_map_.clear();
  height_map_valid_ = false;
  height_map_size_x_ = 0;
  height_map_size_y_ = 0;
}

void TraversabilityLayer::activate()
{
  auto node = node_.lock();
  if (!node) {
    return;
  }

  rclcpp::QoS cloud_qos(rclcpp::KeepLast(static_cast<size_t>(cloud_buffer_size_)));
  cloud_qos.best_effort();
  cloud_sub_ = node->create_subscription<sensor_msgs::msg::PointCloud2>(
    pointcloud_topic_, cloud_qos,
    std::bind(&TraversabilityLayer::pointCloudCallback, this, std::placeholders::_1));

  if (publish_slope_map_) {
    slope_pub_ = node->create_publisher<sensor_msgs::msg::PointCloud2>(
      "slope_map", rclcpp::QoS(1));
  }
}

void TraversabilityLayer::deactivate()
{
  cloud_sub_.reset();
  slope_pub_.reset();
}

}  // namespace traversability_layer

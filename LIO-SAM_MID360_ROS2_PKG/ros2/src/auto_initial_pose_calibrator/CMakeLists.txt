cmake_minimum_required(VERSION 3.5)
project(auto_initial_pose_calibrator)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclpy REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_srvs REQUIRED)

# Install launch files
install(
  DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}/
)

# Install configuration files
install(
  DIRECTORY config
  DESTINATION share/${PROJECT_NAME}/
)

# Install auto_initial_pose_calibrator script as executable
install(
  PROGRAMS src/auto_initial_pose_calibrator.py
  DESTINATION lib/${PROJECT_NAME}
)

# Install calib_lib Python subpackage (源码目录, 运行时 lib/<pkg>/ 在 sys.path)
install(
  DIRECTORY src/calib_lib
  DESTINATION lib/${PROJECT_NAME}
)

# Install visualize_scan script
install(
  PROGRAMS scripts/visualize_scan.py
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()

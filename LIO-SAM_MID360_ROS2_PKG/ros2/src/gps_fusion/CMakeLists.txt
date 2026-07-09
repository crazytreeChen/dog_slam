cmake_minimum_required(VERSION 3.5)
project(gps_fusion)

# Default to C++14 if not set
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

# Install Python nodes as executables
install(
  PROGRAMS
    gps_fusion/gps_preprocessor.py
    gps_fusion/gps_test_steps.py
    gps_fusion/trajectory_server.py
    gps_fusion/trajectory_ws_server.py
    gps_fusion/rtk_pose_monitor.py
    gps_fusion/rtk_continuous_injector.py
    scripts/simulate_trajectory.py
    scripts/rtk_simulator.py
    scripts/mock_tf_broadcaster.py
    scripts/lio_simulator.py
    scripts/test_rtk_fusion.sh
  DESTINATION lib/${PROJECT_NAME}
)

# Install scripts
install(
  DIRECTORY scripts/
  DESTINATION lib/${PROJECT_NAME}
  FILES_MATCHING PATTERN "*.py"
  PATTERN "__pycache__" EXCLUDE
)

# Install Python module files
install(
  DIRECTORY gps_fusion/
  DESTINATION lib/python3.10/site-packages/gps_fusion
  FILES_MATCHING PATTERN "*.py"
  PATTERN "__pycache__" EXCLUDE
)

# Install launch files
install(
  DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}/
)

# Install config files
install(
  DIRECTORY config
  DESTINATION share/${PROJECT_NAME}/
)

# Install web interface files
install(
  DIRECTORY web
  DESTINATION share/${PROJECT_NAME}/
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()

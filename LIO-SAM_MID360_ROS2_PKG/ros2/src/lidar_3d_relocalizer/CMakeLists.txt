cmake_minimum_required(VERSION 3.5)
project(lidar_3d_relocalizer)

# C++17 required by kiss_matcher
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Release build
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# ─── Find dependencies ───
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(pcl_ros REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(PCL REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenMP REQUIRED)
if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
endif()

# ─── KISS-Matcher: 内嵌子模块，多路径回退查找 ───
# 优先级: 1) 用户指定 KISS_MATCHER_DIR  2) 包内子模块  3) 已安装的 kiss_matcher_ros
set(KISS_MATCHER_FOUND FALSE)

# 路径1: 用户通过 -DKISS_MATCHER_DIR=xxx 指定
if(KISS_MATCHER_DIR AND EXISTS "${KISS_MATCHER_DIR}/CMakeLists.txt")
  add_subdirectory("${KISS_MATCHER_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/kiss_matcher")
  set(KISS_MATCHER_FOUND TRUE)
  message(STATUS "lidar_3d_relocalizer: using KISS-Matcher from KISS_MATCHER_DIR=${KISS_MATCHER_DIR}")
endif()

# 路径2: 包内 git submodule (推荐)
if(NOT KISS_MATCHER_FOUND)
  set(_KISS_CANDIDATE "${CMAKE_CURRENT_SOURCE_DIR}/KISS-Matcher/cpp/kiss_matcher")
  if(EXISTS "${_KISS_CANDIDATE}/CMakeLists.txt")
    add_subdirectory("${_KISS_CANDIDATE}" "${CMAKE_CURRENT_BINARY_DIR}/kiss_matcher")
    set(KISS_MATCHER_FOUND TRUE)
    message(STATUS "lidar_3d_relocalizer: using in-package KISS-Matcher at ${_KISS_CANDIDATE}")
  endif()
endif()

# 路径3: 已安装的 kiss_matcher_ros 包
if(NOT KISS_MATCHER_FOUND)
  find_package(kiss_matcher_ros QUIET)
  if(kiss_matcher_ros_FOUND)
    set(KISS_MATCHER_FOUND TRUE)
    message(STATUS "lidar_3d_relocalizer: using installed kiss_matcher_ros")
  endif()
endif()

if(NOT KISS_MATCHER_FOUND)
  message(FATAL_ERROR
    "lidar_3d_relocalizer: KISS-Matcher not found!\n"
    "Please do one of:\n"
    "  1) git submodule update --init (recommended)\n"
    "  2) Pass -DKISS_MATCHER_DIR=/path/to/KISS-Matcher/cpp/kiss_matcher\n"
    "  3) Install kiss_matcher_ros: ros2 run install_kiss_matcher")
endif()

# ndt_omp_ros2 provides ndt_omp and pclomp::gicp_omp
# 先找库文件（静态库 .a 或共享库 .so）
find_library(NDT_OMP_LIB ndt_omp
  PATHS
    ${CMAKE_CURRENT_SOURCE_DIR}/../../ndt_omp_ros2
    ${CMAKE_BINARY_DIR}/../ndt_omp_ros2
  NO_DEFAULT_PATH
)
if(NOT NDT_OMP_LIB)
  find_package(ndt_omp_ros2 QUIET)
  if(ndt_omp_ros2_FOUND)
    set(NDT_OMP_LIB ndt_omp)
  else()
    message(FATAL_ERROR "ndt_omp library not found. Build ndt_omp_ros2 first.")
  endif()
endif()
message(STATUS "lidar_3d_relocalizer: ndt_omp lib = ${NDT_OMP_LIB}")

# ─── Include directories ───
# KISS-Matcher: #include <kiss_matcher/core/kiss_matcher/KISSMatcher.hpp>
# 需要的 include 路径是 KISS-Matcher/cpp (kiss_matcher/ 的父目录)
set(KISS_MATCHER_INCLUDE_DIR "")
if(KISS_MATCHER_DIR)
  get_filename_component(KISS_MATCHER_INCLUDE_DIR "${KISS_MATCHER_DIR}/.." ABSOLUTE)
elseif(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/KISS-Matcher/cpp/kiss_matcher/CMakeLists.txt")
  set(KISS_MATCHER_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/KISS-Matcher/cpp")
endif()

include_directories(
  include
  ${PCL_INCLUDE_DIRS}
  ${KISS_MATCHER_INCLUDE_DIR}
)

# ─── Build node ───
add_executable(lidar_3d_relocalizer_node
  src/lidar_3d_relocalizer_node.cpp
  src/pcd_map_loader.cpp
  src/cloud_accumulator.cpp
  src/relocalizer_core.cpp
  src/map_alignment.cpp
)

# ament_target_dependencies: 不包含 kiss_matcher_ros（用 target_link_libraries 直接链接）
ament_target_dependencies(lidar_3d_relocalizer_node
  rclcpp
  sensor_msgs
  nav_msgs
  geometry_msgs
  std_msgs
  tf2_ros
  pcl_ros
  pcl_conversions
)

target_link_libraries(lidar_3d_relocalizer_node
  ${PCL_LIBRARIES}
  Eigen3::Eigen
  OpenMP::OpenMP_CXX
  kiss_matcher::kiss_matcher_core
  ${NDT_OMP_LIB}
)

# ─── Install ───
install(TARGETS
  lidar_3d_relocalizer_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  config
  DESTINATION share/${PROJECT_NAME}
)

install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  ament_add_gtest(test_map_alignment
    test/test_map_alignment.cpp
    src/map_alignment.cpp
  )
  target_include_directories(test_map_alignment PRIVATE include)
  target_link_libraries(test_map_alignment Eigen3::Eigen)
endif()

ament_package()

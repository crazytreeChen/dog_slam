# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository purpose

ROS2 Humble workspace for SLAM + Nav2 navigation on a Livox MID360-equipped quadruped ("dog") robot. Targets RK3588 boards on the robot and Ubuntu desktops/Orin for development. The repo is edited on Windows but **only builds and runs on Ubuntu 22.04 / ROS2 Humble** — bash scripts and absolute Linux paths in `global_config` are the source of truth.

## Build

The colcon workspace lives at `LIO-SAM_MID360_ROS2_PKG/ros2/`. The top-level `LIO-SAM_MID360_ROS2_PKG/build_ros2.sh` first builds the Livox SDK driver via its own `livox_ros_driver2/build.sh humble`, then `colcon build --symlink-install` for each package. Always pass `humble` to the driver build script.

```bash
cd LIO-SAM_MID360_ROS2_PKG
./build_ros2.sh                         # full build
# or, after first build, rebuild a single package:
cd ros2 && colcon build --symlink-install --packages-select <pkg>
source ros2/install/setup.bash
```

`livox_gazebo_ros2_gpu_simulation` is intentionally skipped in `build_ros2.sh`.

Submodules (`Super-LIO`, `zsi_tools/fast_lio_robosenseAiry`) must be initialized: `git submodule update --init --recursive`.

## Run

There is **one launch file** for everything: `nav2_dog_slam lio_nav2_unified.launch.py`. Behavior is selected via environment variables, not launch arguments:

| Var | Values | Effect |
|-----|--------|--------|
| `SLAM_ALGORITHM` | `fast_lio` / `lio_sam` / `point_lio` / `super_lio` / `super_lio_zg` / `super_lio_gazebo` / `no_lio` | Which LIO node starts; also picks the topic remapping table in `LIO_TOPIC_CONFIGS` |
| `MANUAL_BUILD_MAP` | `True` / `False` | Mapping mode (octomap_server or slam_toolbox, no Nav2/web) vs navigation mode |
| `BUILD_TOOL` | `slam_toolbox` / `octomap_server` | Which 2D mapper runs in mapping mode |
| `AUTO_BUILD_MAP` | `True` / `False` | Delays/launches `explore_lite` for autonomous exploration |
| `NAVIGATION_MODE` | `standalone` / `integrated` | Nav2 wiring |

Helper scripts in `LIO-SAM_MID360_ROS2_PKG/scripts/` set these vars and call the launch file:

```bash
./scripts/run_buildmap.sh <slam_algorithm> [slam_toolbox|octomap_server]
./scripts/run_navigation.sh <slam_algorithm>
./scripts/build_and_run.sh <slam_algorithm>   # rebuilds the LIO pkg first
```

These scripts hard-code `WORKSPACE_DIR=/home/ztl/dog_slam/...` — they only work on the RK3588 boards. On a dev machine, `source install/setup.bash` and `ros2 launch nav2_dog_slam lio_nav2_unified.launch.py` directly.

Multi-robot namespacing: `ros2 launch nav2_dog_slam lio_nav2_unified.launch.py ns:=rkbot`. **Only Super-LIO is namespace-aware**; the other LIO algorithms will conflict if used with `ns:=`.

## Architecture

### Per-host configuration (`global_config`)

`ros2/src/global_config/global_config/__init__.py` is the single source of truth for paths, lidar offsets, save dirs, namespace defaults, and which Nav2 params yaml to load. It picks a config dict by `platform.node()` hostname — current entries: `RK3588`, `RK3588ZG` (auto-detected when `ROS_DOMAIN_ID=24` or zenoh RMW), `DESKTOP-4LS1SSN`, `DESKTOP-ypat`, `ywj-B250-D3A`, `orin-nx`. Unknown hosts fall back to `default_config`.

**Side effect at import time**: this module rewrites `nav2_params.yaml`, `mid360.yaml` (FAST-LIO and Point-LIO), and `livox_360.yaml` (Super-LIO) in place — patching `use_sim_time`, `lidar_type`, `map_file_path`, `save_map`, etc. Editing those yaml files manually is futile; change the host config dict instead. To add a new machine, add a new key to `config_by_machine`.

### Unified launcher (`nav2_dog_slam/launch/lio_nav2_unified.launch.py`)

Composes the system from a fixed set of building blocks:

- **LIO**: includes one of `fast_lio/launch/mapping.launch.py`, `point_lio/launch/mapping_mid360.launch.py`, `lio_sam/launch/lio_sam.launch.py`, or Super-LIO's `Livox_mid360.py` / `Livox_mid360_zg.py` / `gazebo_mid360.py`, gated by `IfCondition` on `SLAM_ALGORITHM`.
- **Topic remapping** is keyed off `LIO_TOPIC_CONFIGS[SLAM_ALGORITHM]` — each algorithm publishes registered cloud and odom under different names (`cloud_registered_body` vs `lio/body/cloud` vs `lio_sam/mapping/...`). Downstream nodes (pointcloud_to_laserscan, octomap_server, costmap layers) get remapped accordingly. If you add a new LIO, also add an entry here.
- **Mapping mode**: octomap_server or slam_toolbox (slam_toolbox uses `RewrittenYaml` to namespace-prefix all frame ids when `ns` is set).
- **Navigation mode**: `nav2_map_server` + `nav2_amcl` + lifecycle managers + the standard Nav2 stack from `navigation_launch.py`.
- **Frame names** are computed via `PythonExpression`: empty `ns` → bare `map`/`odom`/`base_footprint`/`base_link`; non-empty `ns` → `<ns>/map` etc. All frame/topic params passed to nodes are these `ns_*` substitutions, not string literals.
- CPU pinning via `prefix=['taskset -c ...']` is intentional for RK3588 big.LITTLE scheduling — preserve it.

### `traversability_layer`

Custom Nav2 costmap plugin (`traversability_layer::TraversabilityLayer`) for 3D traversability from raw point cloud — slope, step height, stair detection. Configured under `local_costmap.plugins` in `nav2_params*.yaml`. Reads `/cloud_registered_body` by default.

### Other packages

- `LIO-SAM_MID360_ROS2_DOG`, `FAST_LIO_ROS2_edit`, `point_lio_ros2`, `Super-LIO` (submodule) — the four LIO implementations.
- `SC_PGO_ROS2` — pose-graph optimization. Super-LIO can output SC-PGO-compatible `Scans/NNNNNN.pcd` + KITTI-format `odom_poses.txt` via `lio.sc_pgo.enable: true` in `livox_360.yaml` / `livox_360_zg.yaml`.
- `lidar_localization_ros2` + `ndt_omp_ros2` — re-localization on a saved PCD.
- `livox_ros_driver2` — Livox SDK2 ROS2 wrapper. Per-host JSON config selected by `LIVOX_MID360_CONFIG` (e.g., `MID360_config_zg.json` for the ZG dog with two lidars).
- `livox_gazebo_garden` — Gazebo Garden simulation worlds + URDF.
- `zsi_tools/` contains the `zg_double_lidar` front+back lidar fusion package and the Robosense Airy LIO submodule.
- `m-explore` — frontier-based autonomous exploration (`ros2 launch m-explore explore.launch.py`).

### Map saving

```bash
ros2 service call /lio_sam/save_map lio_sam/srv/SaveMap "{resolution: 0.05, destination: '/projects/LOAM/'}"
ros2 run nav2_map_server map_saver_cli -t /projected_map -f /path/to/map --fmt png
```

Super-LIO writes incremental PCDs to `<save_map_dir>/PCD/` and merges into `<save_map_dir>/test.pcd` on shutdown; if `dynamic_remove` is enabled, per-frame `filtered_*.pcd` files are stream-merged to avoid OOM.

## Conventions

- Default branch: `main`. The workspace is a working tree of multiple submodules — verify which repo a path belongs to before committing.
- README, comments, commit messages, and `freqcmd.txt` notes are in Chinese; match that style when editing existing files. New code identifiers stay in English.
- `freqcmd.txt` at the repo root is the developer's running notes/scratchpad — read it for context but treat it as informational, not a spec.
- Don't edit the auto-rewritten yaml files (`nav2_params*.yaml`, `mid360.yaml`, `livox_360*.yaml`) for host-specific values; edit `global_config/__init__.py` instead. Other yaml fields can still be edited normally.

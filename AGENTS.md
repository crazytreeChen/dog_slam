# AGENTS.md

本文件约束自动化代理在本工作区中的默认行为，以 Superpowers 为主工作流体系按需激活。

## 指令优先级

1. 当前会话用户的明确要求
2. 仓库自身规则、文档与约定
3. 本文件
4. 相关 Superpowers / skill 流程定义

- 默认以 Superpowers 作为主工作流体系，但不默认启用 full Superpowers。
- 本文件保留个人硬门禁、环境约束、交付偏好与沟通方式。
- 只读分析任务可不进入完整实现流程，但结论必须清晰、可追溯。
- 若用户明确要求 `continue nonstop`，默认持续推进，直到满足验收标准或出现真实阻塞。

## 仓库专有信息

### Repository purpose

ROS2 Humble workspace for SLAM + Nav2 navigation on a Livox MID360-equipped quadruped ("dog") robot. Targets RK3588 boards on the robot and Ubuntu desktops/Orin for development. The repo is edited on Windows but **only builds and runs on Ubuntu 22.04 / ROS2 Humble** — bash scripts and absolute Linux paths in `global_config` are the source of truth.

### Build

The colcon workspace lives at `LIO-SAM_MID360_ROS2_PKG/ros2/`. The top-level `LIO-SAM_MID360_ROS2_PKG/build_ros2.sh` first builds the Livox SDK driver via its own `livox_ros_driver2/build.sh humble`, then `colcon build --symlink-install` for each package. Always pass `humble` to the driver build script.

```bash
cd LIO-SAM_MID360_ROS2_PKG
./build_ros2.sh                         # full build
# or, after first build, rebuild a single package:
cd ros2 && colcon build --symlink-install --packages-select <pkg>
source ros2/install/setup.bash
```

`livox_gazebo_ros2_gpu_simulation` is intentionally skipped in `build_ros2.sh`.

Note: `build_ros2.sh` references `faster_lio` and `direct_lidar_inertial_odometry` packages that no longer exist in the source tree — the build will emit warnings for these but continue. Ignore them.

Submodules (`Super-LIO`, `zsi_tools/fast_lio_robosenseAiry`) must be initialized: `git submodule update --init --recursive`.

### Run

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

### Architecture

#### Per-host configuration (`global_config`)

`ros2/src/global_config/global_config/__init__.py` is the single source of truth for paths, lidar offsets, save dirs, namespace defaults, and which Nav2 params yaml to load. It picks a config dict by `platform.node()` hostname — current entries: `RK3588`, `RK3588ZG` (auto-detected when `ROS_DOMAIN_ID=24` or zenoh RMW), `DESKTOP-4LS1SSN`, `DESKTOP-ypat`, `ywj-B250-D3A`, `orin-nx`. Unknown hosts fall back to `default_config`.

**Side effect at import time**: this module rewrites `nav2_params.yaml`, `mid360.yaml` (FAST-LIO and Point-LIO), and `livox_360.yaml` (Super-LIO) in place — patching `use_sim_time`, `lidar_type`, `map_file_path`, `save_map`, etc. Editing those yaml files manually is futile; change the host config dict instead. To add a new machine, add a new key to `config_by_machine`.

#### Unified launcher (`nav2_dog_slam/launch/lio_nav2_unified.launch.py`)

Composes the system from a fixed set of building blocks:

- **LIO**: includes one of `fast_lio/launch/mapping.launch.py`, `point_lio/launch/mapping_mid360.launch.py`, `lio_sam/launch/lio_sam.launch.py`, or Super-LIO's `Livox_mid360.py` / `Livox_mid360_zg.py` / `gazebo_mid360.py`, gated by `IfCondition` on `SLAM_ALGORITHM`.
- **Topic remapping** is keyed off `LIO_TOPIC_CONFIGS[SLAM_ALGORITHM]` — each algorithm publishes registered cloud and odom under different names (`cloud_registered_body` vs `lio/body/cloud` vs `lio_sam/mapping/...`). Downstream nodes (pointcloud_to_laserscan, octomap_server, costmap layers) get remapped accordingly. If you add a new LIO, also add an entry here.
- **Mapping mode**: octomap_server or slam_toolbox (slam_toolbox uses `RewrittenYaml` to namespace-prefix all frame ids when `ns` is set).
- **Navigation mode**: `nav2_map_server` + `nav2_amcl` + lifecycle managers + the standard Nav2 stack from `navigation_launch.py`.
- **Frame names** are computed via `PythonExpression`: empty `ns` → bare `map`/`odom`/`base_footprint`/`base_link`; non-empty `ns` → `<ns>/map` etc. All frame/topic params passed to nodes are these `ns_*` substitutions, not string literals.
- CPU pinning via `prefix=['taskset -c ...']` is intentional for RK3588 big.LITTLE scheduling — preserve it.

#### `traversability_layer`

Custom Nav2 costmap plugin (`traversability_layer::TraversabilityLayer`) for 3D traversability from raw point cloud — slope, step height, stair detection. Configured under `local_costmap.plugins` in `nav2_params*.yaml`. Reads `/cloud_registered_body` by default.

#### Other packages

- `LIO-SAM_MID360_ROS2_DOG`, `FAST_LIO_ROS2_edit`, `point_lio_ros2`, `Super-LIO` (submodule) — the four LIO implementations.
- `auto_initial_pose_calibrator` — automatic initial pose calibration using scan matching + AMCL convergence. Depends on `global_config` and `robots_dog_msgs`.
- `SC_PGO_ROS2` — pose-graph optimization. Super-LIO can output SC-PGO-compatible `Scans/NNNNNN.pcd` + KITTI-format `odom_poses.txt` via `lio.sc_pgo.enable: true` in `livox_360.yaml` / `livox_360_zg.yaml`.
- `lidar_localization_ros2` + `ndt_omp_ros2` — re-localization on a saved PCD.
- `livox_ros_driver2` — Livox SDK2 ROS2 wrapper. Per-host JSON config selected by `LIVOX_MID360_CONFIG` (e.g., `MID360_config_zg.json` for the ZG dog with two lidars).
- `livox_gazebo_garden` — Gazebo Garden simulation worlds + URDF.
- `zsi_tools/` contains the `zg_double_lidar` front+back lidar fusion package, the Robosense Airy LIO submodule, and `d1max` board configs (RK3588/Orin).
- `m-explore` — frontier-based autonomous exploration (`ros2 launch m-explore explore.launch.py`).

#### Map saving

```bash
ros2 service call /lio_sam/save_map lio_sam/srv/SaveMap "{resolution: 0.05, destination: '/projects/LOAM/'}"
ros2 run nav2_map_server map_saver_cli -t /projected_map -f /path/to/map --fmt png
```

Super-LIO writes incremental PCDs to `<save_map_dir>/PCD/` and merges into `<save_map_dir>/test.pcd` on shutdown; if `dynamic_remove` is enabled, per-frame `filtered_*.pcd` files are stream-merged to avoid OOM.

#### Conventions

- Default branch: `main`. The workspace is a working tree of multiple submodules — verify which repo a path belongs to before committing.
- README, comments, commit messages, and `freqcmd.txt` notes are in Chinese; match that style when editing existing files. New code identifiers stay in English.
- `freqcmd.txt` at the repo root is the developer's running notes/scratchpad — read it for context but treat it as informational, not a spec.
- Don't edit the auto-rewritten yaml files (`nav2_params*.yaml`, `mid360.yaml`, `livox_360*.yaml`) for host-specific values; edit `global_config/__init__.py` instead. Other yaml fields can still be edited normally.

#### Knowledge Base (Obsidian)

所有技术笔记、项目经验、问题排查、工作日报等**知识资产统一写入 Obsidian vault**，路径：

```
/Users/qinglinchen/01-Code/98-Custom/obsdian/
```

**核心规则**：
- **代码文档**（README、AGENTS、CLAUDE、API doc）留在项目仓库内
- **知识笔记**（概念卡片、踩坑记录、问题排查、日报周报月报）全部写入 Obsidian vault
- 禁止在项目仓库中散落 `.md` 知识笔记（`CLAUDE.md` / `AGENTS.md` / 子模块 README 除外）

**Vault 目录映射**：

| 目录 | 写入内容 | 关联 Skill |
|------|----------|------------|
| `01-技术学习/概念卡片/` | 技术概念（ROS2, SLAM, LiDAR, Nav2, PID...） | `tech-knowledge-organizer` |
| `01-技术学习/问题排查/` | Bug 排查、环境问题、异常处理 | `tech-knowledge-organizer` |
| `01-技术学习/语言/` | C++/Python/Shell 等语言笔记 | `tech-knowledge-organizer` |
| `01-技术学习/框架/` | Qt6, FFmpeg, OpenCV 等框架笔记 | `tech-knowledge-organizer` |
| `02-工作项目/进行中/<项目>/` | 项目主文档、经验日志、版本记录 | `tech-knowledge-organizer` |
| `10-工作记录/YYYY/MM月/第WW周/` | 日报 (`MM-DD.md`)、周报 (`周报.md`)、月报 (`月报.md`) | `report-summarizer` |

**笔记规范**：
- 必须包含 YAML frontmatter：`type`, `created`, `tags`
- 强制使用 Obsidian 双链 `[[NoteName]]` 关联已有笔记
- 新增笔记后同步更新对应 MOC 索引页面（`MOC-*.md`）
- 概念卡片遵循 `99-Templates/概念卡片模板.md`
- 问题排查遵循 `99-Templates/问题排查模板.md`
- 经验日志遵循 `99-Templates/经验日志模板.md`

**重要**：日报/周报/月报只写工作内容，不体现个人知识库整理、Obsidian、skill 配置等个人事务。

**当前活跃项目**：`dog_slam` → `02-工作项目/进行中/dog_slam/`

---

## 任务分流与流程选择

### 三类任务

| 类型 | 范围 | 默认流程 |
|------|------|----------|
| **轻量** | 单文件或小范围修改、明确 bug 修复、配置/文案调整、小测试补充、局部文档 | 跳过 brainstorming / writing-plans / review 链，直接实现+定向验证 |
| **中型** | 跨 2-4 文件、新功能、行为变更、重构 | `brainstorming → writing-plans → implementation → verification` |
| **重型** | 跨模块、涉及公共 API/schema/持久化/并发、需求模糊、影响面大 | 完整 Superpowers 流程 + review + 完整验证 |

### 流程升降级

- **升级触发**：影响边界超预期、涉及共享接口/数据/持久化/并发、需求不清晰、验证覆盖不足、任务演变为中大型重构
- **降级触发**：改动局部且边界清晰、不涉及共享核心逻辑、问题已收敛为单点修复、补长计划/测试的成本高于收益
- **总原则**：满足质量要求的最短路径；能走轻量不走重流程；小任务首次最多问 1 个关键问题，中型任务优先一次性给出 2-3 个方案与推荐

---

## 执行与验证纪律

### 推进规则

1. 需求模糊时先澄清目标、约束、验收标准与边界条件
2. 多步任务维护可见任务列表，任意时刻仅保留一个 `in_progress`
3. 先缩小边界再扩展范围，优先局部修改与最小充分实现
4. 若复杂度上升及时升级流程，若已收敛及时降级
5. 遇到新信息应主动修正之前的判断

### 验证规则

- 不得虚构已运行命令、退出码或验证结果
- 关键验证无法执行时必须明确说明原因
- 没有验证证据不得声称"通过""完成""可提交""可合并"

### 授权边界

- **可默认执行**：当前分支内与任务直接相关的应用代码、测试、局部文档，可新增少量配套文件
- **必须确认**：删除文件、大规模重构、shared contract / schema / shared types、根配置 / CI / 依赖 / 环境模板、数据库 / 持久化变更、git 历史与远程操作、基础设施改动

---

## 质量门禁

### 交付前检查（Change Delivery Gate）

在声明完成、准备 commit/push/PR 之前必须满足：

1. 已完成与本次改动直接相关的验证，并如实报告结果
2. 已完成对应质量门禁
3. 若仓库要求更重验证则优先遵循仓库规则
4. 若关键验证无法执行则明确说明原因并降低完成度表述

### 测试分层

| 级别 | 场景 | 内容 |
|------|------|------|
| L0 定向验证 | 局部、低风险、小改动 | 手动/脚本验证关键路径 |
| L1 回归测试 | 中小修复或局部行为变化 | 现有测试全部通过 |
| L2 TDD | 新功能、行为变更、共享逻辑或高风险 | 先写测试再实现 |
| L3 Code Review | 中大型改动 | `requesting-code-review` / `receiving-code-review` |
| L4 Completion Verification | 所有改动 | `verification-before-completion` + Change Delivery Gate |

TDD 不默认强制，按"行为影响、共享范围、回归风险、测试价值"显式判定是否启用。

---

## 代码规范

### 硬性上限

函数 ≤ 50 行、文件 ≤ 300 行、嵌套 ≤ 3、位置参数 ≤ 3、圈复杂度 ≤ 10、禁止魔法数字。

### 编码原则

遵循 SOLID、DRY、关注点分离、YAGNI；命名清晰，边界条件显式处理；优先局部修改与最小充分实现。

### Bug 修复

真实 bug 默认优先 `systematic-debugging`，先确认根因再修复；Bug 报告应写清现象、触发条件、预期、实际、影响范围、严重程度及日志/堆栈/环境信息。

### 测试

优先覆盖关键路径、边界情况和错误路径；断言优先 expected 在前、actual 在后。

### 重构

默认先保持行为不变再提升结构质量；必要时先补测试再重构；若出现循环导入则提取共享逻辑；较大重构先拆分计划。

### Commit 规范

格式 `<type>(scope): <summary>`，summary 中文动词开头、≤ 50 字、不加句号；常用 type：`feat` / `fix` / `refactor` / `docs` / `test` / `chore`，scope 可选。

---

## 沟通与输出

### 沟通风格

- 默认简体中文，可混用英文技术术语；代码标识符英文；注释优先简体中文
- 回答时优先给结论，再补背景、依据与权衡

### 输出模式

| 模式 | 场景 | 结构 |
|------|------|------|
| **执行进度式** | 代码修改、重构、bug 修复、多步任务 | 任务 → 执行计划(已/当前/待) → 当前进度 → 风险/阻塞 → 参考 |
| **分析回答式** | 问答、代码解释、方案对比、架构分析 | 结论 → 关键分析 → 深入剖析(可选) → 风险与权衡(可选) |

技术内容：多行代码/配置/日志优先带语言标识的 Markdown 代码块，仅在确有必要时使用表格，复杂内容后附简短总结。

---

## 并行开发

### 子代理派发

- 子代理模型仅允许 `gpt-5.5`（默认）与 `gpt-5.4`（仅限代码实现/测试修复/局部重构/单模块阅读）
- reasoning_effort 仅允许 `high` 或 `xhigh`，歧义时上调为 `xhigh`
- 派发前判断是否确有委派价值，并说明所选模型与推理等级原因

### 并行准入

仅当任务可拆分为 2-4 个边界清晰、scope 明确、独立验证且无同文件写冲突的子任务时才适合并行。禁止两个子任务修改同一文件、同一配置源或同一 shared types；lockfile / CI / schema / 路由入口 / 公共适配层必须串行或统一收尾。

### 收尾

所有子任务完成后必须统一收尾：汇总改动 → 检查冲突面 → 分析依赖与合并顺序 → 必要时新增 integration task → 运行最终验证 → 输出最终 merge plan。未经用户明确要求，子任务不得自行 merge/rebase/push/删除 worktree。

---

## 技能（Skills）

技能存放位置：`C:\Users\chenql\.agents\skills`（用户路径可能不同）。开始任务前优先判断是否命中 skill。

主干整合方式：

- 实现前：`brainstorming → writing-plans`
- Debug：`systematic-debugging`
- Review：`requesting-code-review` / `receiving-code-review`
- 完成前：`verification-before-completion`
- 高风险行为变更：`test-driven-development`
- 前端设计：`ui-ux-pro-max`
- 并行规划：`codex-parallel-collab`
- 会话收尾：`session-wrap`

在回复中声明本次使用了哪些技能。

---

## 安全规则

- 不运行破坏性命令（如 `git reset`），除非用户明确要求
- 不操作用户未授权的危险删除，临时产物例外
- 不将密钥、凭证、API Key 硬编码进源码
- 数据库访问使用参数化查询
- 不拼接不可信输入到 shell/SQL
- 除非用户明确要求，不终止非当前任务启动的进程

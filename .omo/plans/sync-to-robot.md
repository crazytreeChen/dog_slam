# Work Plan: sync_to_robot.sh

## Objective
Create `LIO-SAM_MID360_ROS2_PKG/scripts/sync_to_robot.sh` — a bash script that incrementally syncs git-modified files from the local dog_slam workspace to a remote robot server (RK3588 board), using rsync over SSH.

## Requirements Summary

| # | Requirement | Detail |
|---|-------------|--------|
| R1 | Incremental sync | Only sync files modified since last git commit (`git diff --name-only HEAD`) |
| R2 | Exclude untracked | Do NOT sync new/untracked files |
| R3 | Exclude build artifacts | Skip `build/`, `install/`, `log/`, `.git`, `__pycache__`, `*.pyc` |
| R4 | First-time full sync | Auto-detect if remote project dir is missing → rsync full workspace (with exclusions) |
| R5 | No post-sync actions | File transfer only; no auto-compile or service restart |
| R6 | Self-configurable | ROBOT_IP, ROBOT_USER, ROBOT_PATH as variables at top of script |
| R7 | Git Bash compatible | Runs on Windows Git Bash (bash + rsync + ssh available) |
| R8 | Script location | `LIO-SAM_MID360_ROS2_PKG/scripts/sync_to_robot.sh` |

## Acceptance Criteria

| AC | Criteria | Verification |
|----|----------|-------------|
| AC1 | Script is bash-parseable | `bash -n sync_to_robot.sh` exit 0 |
| AC2 | No changes case | Displays "no changes to sync" and exits cleanly |
| AC3 | Modified files synced | Only `git diff HEAD` files transferred, build artifacts excluded |
| AC4 | First-time fallback | Remote dir missing → full rsync runs automatically |
| AC5 | SSH failure | Clear error message, non-zero exit |
| AC6 | rsync missing | Early check with clear message |
| AC7 | Config self-contained | All target config in variables at script top |
| AC8 | Dry-run mode | `--dry-run` flag shows what would be synced |

## Technical Approach

### Architecture Decision
**Plain bash + rsync + git** (no Python, no extra deps). Reasons:
- rsync already proven in user's reference script
- git already available (repo is git-tracked)
- bash is the native scripting language for Git Bash
- No need for Python cross-platform complexity

### Core Logic Flow

```
START
  │
  ├─ 1. Check prerequisites (bash, git, rsync, ssh)
  │
  ├─ 2. Parse flags (--dry-run, --help)
  │
  ├─ 3. Load config (ROBOT_IP, ROBOT_USER, ROBOT_PATH)
  │
  ├─ 4. Verify local is a git repo
  │
  ├─ 5. Check remote connectivity
  │     ├─ ssh connection test → FAIL: error + exit
  │     └─ OK: continue
  │
  ├─ 6. Detect first-time sync
  │     ├─ Remote dir NOT exists → FULL SYNC branch
  │     └─ Remote dir exists → INCREMENTAL branch
  │
  ├─ 7. FULL SYNC branch:
  │     └─ rsync -avz --progress --exclude=... ./ → remote
  │
  ├─ 8. INCREMENTAL branch:
  │     ├─ git diff --name-only HEAD
  │     ├─ Filter out excluded paths (build/ install/ log/ .git/ __pycache__/ *.pyc)
  │     ├─ No files left → "no changes" + exit 0
  │     └─ Has files → rsync --files-from=<tempfile> → remote
  │
  └─ 9. Report results
```

### Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Sync method (incremental) | `rsync --files-from=<(filtered list)` | Preserves directory structure, faster than per-file scp |
| Sync method (full) | `rsync -avz --progress --exclude=...` | Same as reference script pattern |
| Path handling | Convert to Unix-style relative paths | `git diff` outputs OS-native paths; normalize for rsync over SSH |
| File list temp | Named temp file, cleaned up on exit | `--files-from` needs a real file in Git Bash; process substitution `<()` may fail on some Windows setups |
| Error handling | `set -e` + explicit checks at key points | Fail fast, don't half-sync |
| Output style | Colored echo (green/red/yellow) with emoji | Consistent with reference script style |
| Config format | Simple bash variables at script top | Self-documenting, no separate config file needed |

### Exclude Patterns

```bash
EXCLUDE_PATTERNS=(
    "build/"
    "install/"
    "log/"
    ".git/"
    "__pycache__/"
    "*.pyc"
)
```

For full sync: rsync `--exclude` flags.
For incremental sync: grep filter against the git diff output using these patterns.

### Edge Cases Handled

| Edge Case | Behavior |
|-----------|----------|
| No uncommitted changes | Display "No changes to sync", exit 0 |
| All changes are in excluded dirs | Display "No eligible files to sync", exit 0 |
| Remote dir missing | Auto full sync with warning message |
| SSH connection timeout | Graceful error, suggest checking network/IP |
| rsync not installed | Early check, clear error |
| Not a git repo | Early check, clear error |
| File paths with spaces | Use while-read loop with null-delimited processing |
| git diff returns deleted files (D status) | Filter out; only sync modified (M) and added (A) files |
| Large number of changed files | rsync handles efficiently; show count before syncing |

## Implementation Plan

### Step 1: Create script skeleton
**File**: `LIO-SAM_MID360_ROS2_PKG/scripts/sync_to_robot.sh`
- Shebang `#!/bin/bash`
- Config variables section
- Prerequisite checks (git, rsync, ssh)
- Flag parsing (--dry-run, --help)

### Step 2: Implement connectivity check and first-sync detection
- `ssh -o ConnectTimeout=5` to test connection
- `ssh ... "test -d $ROBOT_PATH"` to detect existing project

### Step 3: Implement full sync logic
- rsync with `--exclude` for each pattern
- Show progress, file count

### Step 4: Implement incremental sync logic
- `git diff --name-only --diff-filter=AM HEAD` (only Added/Modified, skip Deleted)
- Filter against exclude patterns
- Write filtered list to temp file
- rsync with `--files-from`

### Step 5: Implement output/reporting
- Colored status messages
- Success/failure summary with file count
- Dry-run: show what would happen without transferring

## Files Changed
| File | Action |
|------|--------|
| `LIO-SAM_MID360_ROS2_PKG/scripts/sync_to_robot.sh` | **CREATE** |

## Test Strategy
- No unit test framework applicable (bash script)
- Manual verification:
  1. `bash -n sync_to_robot.sh` for syntax
  2. `bash sync_to_robot.sh --dry-run` to verify file list
  3. `bash sync_to_robot.sh --help` to verify usage output
  4. Actual sync test with a real robot (user responsibility)

## Dependencies
- Git Bash (with rsync, ssh, git)
- SSH key already configured for password-less login to robot
- (Assumption) User has already set up ssh key auth to the robot

## Estimated Complexity
- **Low**. Single bash script, ~100-150 lines. No architecture decisions. Straightforward logic.

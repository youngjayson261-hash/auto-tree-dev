# vLLM Ascend Development Guidelines

## ECHO 提交范围约束（用户明确要求，2026-11-09）

- **无关 CI 清理不得重新混入 ECHO**。此约束跨对话持续有效，适用于 ECHO 分支、提交和 PR，优先于下文通用的格式化结果纳入建议。
- 历史清理 `50f2dfa2`（`chore: repair inherited CI checks and formatting`）已由 `7cf26bc1` 按用户要求回退。`29522800`（`chore: restore repository CI checks`）误将同类清理重新带入，已由 `6277cf5d` 精确回退；不得再次套用这些已回退的清理补丁。
- 不得借“恢复 CI”加入无关的全仓格式化、日志器替换、导入清理或重排、`noqa` 豁免、全局拼写白名单、C++ 命名修改等；另起清理提交也不代表获准混入 ECHO。
- 当前功能引入的问题应定向修复，局部格式遵循仓库要求。若认为必须重新纳入已回退清理，须先列出具体文件、行为影响和必要性，取得用户明确授权；CI 报错本身不构成授权。
- 功能验证与继承 CI 失败分开记录。全量格式检查在隔离副本执行，不直接写回自动修复结果；如实报告当前版本的失败，不复用历史通过结论。
- 每次提交前检查 Git 历史、暂存区和最终差异，保留用户未提交改动，仅暂存本次范围内的文件或补丁。CI 通常不妨碍本地提交；远端必需检查可能阻止 PR 合并，不能据此擅自扩大修改范围。

This document provides instructions for contributors to the vLLM Ascend project. Please read and follow these guidelines to ensure code quality, maintainability, and consistency.

Read linked documentation when it applies to the current task, using the relevant sections. Continue authorized local implementation and validation while preparing the evidence needed for PR review.

---

## Table of Contents

- [Setup and Environment](#setup-and-environment)
    - [Environment Variables](#environment-variables)
    - [Environment Variable Review Requirement](#environment-variable-review-requirement)
- [Testing](#testing)
- [Code Style](#code-style)
    - [Python Conventions](#python-conventions)
    - [Naming Conventions](#naming-conventions)
- [NPU-Specific Considerations](#npu-specific-considerations)
    - [Tensor item() Operations](#tensor-item-operations)
    - [Memory and Performance](#memory-and-performance)
- [Model and Plugin Architecture](#model-and-plugin-architecture)
    - [vLLM Ascend Plugin Architecture](#vllm-ascend-plugin-architecture)
    - [Patching Requirement](#patching-requirement)
    - [Model Runner Changes](#model-runner-changes)
- [Release Notes](#release-notes)
- [Commit Messages and Pull Requests](#commit-messages-and-pull-requests)
    - [Commit Message Format](#commit-message-format)
- [Quick Start for Contributors](#quick-start-for-contributors)
- [References](#references)

---

## Setup and Environment

### Environment Variables

Define environment variables in the centralized `env_variables` dictionary in [vllm_ascend/envs.py](vllm_ascend/envs.py).

**Requirements:**

- Add documentation for each environment variable in the `env_variables` dict comment
- Specify default values and valid ranges
- Indicate whether the variable is sensitive (credentials, keys)

**Never**: Hardcode environment variable names throughout the codebase. Reference them from the central module using `from vllm_ascend import envs`.

### Environment Variable Review Requirement

Explain new environment variables in the PR for code review.

Reviewers must verify:

- The variable name follows the `VLLM_ASCEND_*` naming convention
- Default value is appropriate for all supported hardware
- Documentation is added to the `env_variables` dict
- The variable is needed and existing configuration cannot cover the use case

---

## Testing

Run tests appropriate to the change and complete required checks. Broaden or repeat testing when new changes, failures, or unresolved risks justify it.

When preparing or reviewing a PR, read the [PR testing requirements](docs/source/developer_guide/contribution/testing.md#pr-testing-requirements) for coverage, hardware evidence, and validation reporting. Use the other sections of the same guide for environment setup and test commands as needed.

---

## Code Style

### Python Conventions

- **Imports**: All imports at the top of the file. Valid exceptions:
    - Circular imports (use inline imports)
    - Lazy loading for worker/isolation processes
    - Type-checking imports wrapped in `if TYPE_CHECKING:`

- **Global Variables**: Avoid new global variables. Pass dependencies explicitly through function parameters.

    **Allowed:**
    - Constants named `ALL_UPPER_CASE`
    - Immutable configuration objects

    If mutable module-level state is necessary, explain its lifetime and concurrency implications in the PR.

- **No Magic Numbers**: Use named constants with descriptive names.

- **Descriptive Naming**: Use names that describe functionality, not implementation details.

### Naming Conventions

- **Classes**: `PascalCase` (e.g., `NPUModelRunner`, `AscendSampler`, `ACLGraphManager`)
- **Functions/Methods**: `snake_case` (e.g., `forward_pass`, `compute_attention`)
- **Constants**: `ALL_UPPER_CASE`
- **Variables**: `snake_case` (e.g., `token_ids`, `sequence_lengths`)

---

## NPU-Specific Considerations

### Tensor item() Operations

**Warning**: `tensor.item()` operations cause synchronization overhead on NPU when the `tensor` is on device.

If the `tensor` is a device tensor, calling `item()` triggers a synchronous data transfer from NPU to CPU. This can severely degrade performance in hot paths, causing `AsyncScheduler` to block here.

**Review Requirements:**

1. Profile synchronization introduced in hot paths before merging
2. Consider alternative patterns:
    - Keep values on device when possible
    - Batch operations to reduce sync frequency
    - Use device-side operations (e.g., `torch.argmax`, `torch.sum`)
3. Document when `item()` is unavoidable (e.g., logging, conditional logic)

### Memory and Performance

Additional NPU-specific best practices:

- Avoid CPU-NPU memory transfers in hot paths
- Prefer in-place operations where safe (e.g., `x.add_()`, `x.mul_()`)
- Monitor memory fragmentation, especially for long-running processes
- Test with realistic workloads on actual NPU hardware (Ascend 910B/C)

---

## Model and Plugin Architecture

### vLLM Ascend Plugin Architecture

vLLM Ascend is a **hardware plugin** that integrates with upstream vLLM via the pluggable hardware interface. Prefer existing upstream implementations and extension points. Ascend-specific model adaptations are also registered in the [model registry](vllm_ascend/models/__init__.py); follow existing registration and inheritance patterns when working on these models.

Choose the integration approach that fits the change:

1. **Patching** (in `vllm_ascend/patch/`):
    - `vllm_ascend/patch/platform/` - Platform-level patches (distributed, scheduling)
    - `vllm_ascend/patch/worker/` - Worker-level patches (model-specific behavior)
    - Patch is not the best solution for all cases. Use it when necessary.

2. **Inheritance**:
    - `NPUModelRunner(GPUModelRunner)` - Extend vLLM model runner with NPU-specific behavior
    - `AscendSampler` - Extend vLLM sampler with NPU-specific operations
    - Add NPU-specific components via composition (e.g., `AclGraphManager`)
    - Custom Operators - NPU-specific custom operators (e.g., `AscendRMSNorm`)

3. **External upstream contributions** where appropriate

### Patching Requirement

Explain new patches in the PR for architectural review.

When adding or changing a patch, read the [patch strategy](docs/source/developer_guide/Design_Documents/patch.md) for lifecycle, registration, and removal-plan requirements.

Reviewers must verify:

- The patch targets the correct upstream component
- The patch is minimal and focused
- Performance implications are understood
- A long-term plan exists for upstream contribution

### Model Runner Changes

Explain new `model_runner` behavior in the PR for architectural review.

Reviewers must verify:

- Why existing extension points cannot cover the new behavior
- Performance implications on NPU hardware
- Compatibility with existing model implementations
- Long-term maintainability and test coverage

**NPU Model Runner Files:**

- `vllm_ascend/worker/v1/model_runner.py` - vLLM v1 model runner (isolated V1 zone, removal-tracked)
- `vllm_ascend/worker/v2/model_runner.py` - vLLM v2 model runner
- `vllm_ascend/worker/v1/_310p/model_runner_310p.py` - Ascend 310P model runner (isolated V1 zone, removal-tracked)

---

## Release Notes

Every PR with a user-facing change must ship a release note fragment in the same PR.

**Required for**: new models, hardware, or operators; new features; changes to user-facing
APIs, parameters, environment variables, or configuration; breaking changes, deprecations,
and default behavior changes; dependency and toolchain updates (CANN, torch_npu,
triton-ascend, vLLM alignment); measurable performance improvements; and important bug fixes.

**Not required for**: internal refactors, tests, CI, build scripts, tooling, and
documentation edits.

Add `release_notes/fragments/<PR number>.yaml` following `release_notes/TEMPLATE.yaml`. See
`release_notes/README.md` for the principles and the field rules. When in doubt, add a
fragment: the release manager drops extra entries at release time. Reviewers confirm the
fragment together with the code; there is no CI gate.

---

## Commit Messages and Pull Requests

### Commit Message Format

Follow the [Conventional Commits](https://www.conventionalcommits.org/) format and **must include a sign-off**:

```bash
git commit -s -m "<type>: <summary>" -m "<body - explaining what changed and why>"
```

**Valid Types**: `feat`, `fix`, `perf`, `refactor`, `test`, `docs`, `chore`

### Pull Request Title Format

Use a PR title accepted by the CI check linked in the [PR submission checklist](docs/source/developer_guide/contribution/testing.md#pr-submission-checklist).

### Pull Request Template

When creating a PR, please follow the template in `.github/PULL_REQUEST_TEMPLATE.md` and ensure the following sections are completed:

> **Note**: The PR description will be automatically updated by GitHub Actions to include vLLM version info at the bottom. If you update the PR description via API or CLI, make sure to preserve the `- vLLM version:` and `- vLLM main:` lines.

- **What this PR does / why we need it?** - Clearly describe the changes and their purpose
- **Does this PR introduce _any_ user-facing change?** - Indicate if there are any user-visible changes
- **How was this patch tested?** - Describe how you tested the changes. Examples:
    - Unit tests added/updated: list the test files
    - Manual testing: provide the test steps and commands
    - CI testing: indicate if only CI verification is needed

Document public APIs and reflect user-facing behavior changes in the project docs.

---

## Quick Start for Contributors

For environment setup, use the [contribution guide](docs/source/developer_guide/contribution/index.md#building-and-testing) as needed.

Before pushing, run the required checks for all file types, including Markdown:

```bash
bash format.sh ci
```

Inspect formatter changes and include only those within the authorized task scope. For ECHO, follow the scope constraints at the top of this file and record inherited CI failures separately. Push to your fork and create the PR against the upstream repository, following the commit and PR requirements above.

---

## References

- [vLLM Hardware Plugin RFC](https://github.com/vllm-project/vllm/issues/11162)
- [Documentation](https://docs.vllm.ai/projects/ascend/en/latest/)
- [Contributors Guide](https://docs.vllm.ai/projects/ascend/en/latest/community/contributors.html)

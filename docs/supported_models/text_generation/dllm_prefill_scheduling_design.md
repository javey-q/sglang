# dLLM Prefill Scheduling Design Note

## 状态

已实现。本文描述当前代码中的 dLLM prefill 调度方案。

GPU 实测步骤与合入门槛见
[`dllm_blockwise_attention_gpu_validation.md`](../../developer_guide/dllm_blockwise_attention_gpu_validation.md)。

## 背景与目标

dLLM decode 以模型 `block_size` 为单位处理 mask。原实现让 prefill 也只能每轮推进一个 decode block，长 prompt 会产生大量 scheduler round。当前方案将 prefill chunk 上限与 decode block 解耦，同时保持 decode 的固定 block 语义。

## 核心不变量

- decode 每个请求始终处理一个完整的 `block_size` block。
- pure prefill 只包含真实 prompt token，不跨入末尾 mask。
- pure-prefill frontier 保持 `block_size` 对齐；非对齐 prompt 尾巴与 mask 由后续 decode block 共同处理。
- pure prefill 使用动态长度的普通 extend，不进入固定 shape 的 dLLM decode graph。
- 单轮只调度 prefill 或 decode，不混合两种 phase。

## 配置

`DllmConfig` 从 `--dllm-algorithm-config` YAML 读取可选的 `prefill_block_size`：

```yaml
block_size: 32
prefill_block_size: 512
threshold: 0.95
```

它表示单个请求每轮 pure prefill 的最大 extend 长度；decode 仍使用 `block_size`。未配置时默认等于 `block_size`，保持原有行为。该值必须不小于 `block_size` 且为其整数倍；空 YAML 按默认配置处理。

当前 `prefill_block_size > block_size` 时要求 prefill attention backend 为 FlashInfer，因为 block-wise custom mask 仅在该 backend 中实现；配置为其他 backend 会在启动时直接报错，避免静默使用错误的可见性。`prefill_block_size == block_size` 的原有单 block 行为不受此限制。

## 调度与 phase 判定

每轮 scheduler 先接收 incoming 请求，再调用 `DllmManager.init_next_round(tree_cache)`。该方法对 incoming/staging 请求先完成 prefix-cache match，再调用 `determine_dllm_phase()`。incoming 在真正被 adder 准入前仍保持 `INCOMING_PREFILL`/`INCOMING_DECODE` 生命周期；只有进入 `adder.can_run_list` 后，`process_dllm_incoming_reqs()` 才将其转为对应 staging phase。

因此 phase 判定使用真实的 `prefix_indices`，不会把“非对齐 prompt 尾巴 + mask”的混合区间误判为 staging prefill，也不需要通过受限重试修正 phase。prefill 到达边界后，下一次初始化会直接选择 decode，避免 scheduler 活锁。

本轮存在 prefill 请求时使用 phase-aware 总预算：

```text
prefill: max_running_requests * prefill_block_size
decode:  max_running_requests * block_size
```

同一 phase 内 staging 优先于 incoming。

## chunk 长度、对齐与非对齐尾巴

`PrefillAdder` 同时受 phase cap、总预算、`max_prefill_tokens` 和 KV 可用预算约束，并按 page/block 边界向下对齐。pure prefill 的终点为：

```text
context_end      = len(origin_input_ids) + len(output_ids)
pure_prefill_end = floor(context_end / block_size) * block_size
prefill_end      = min(prefix_len + available, pure_prefill_end)
```

例如 `block_size=32`、prompt 长度为 300 时，pure prefill 最多提交到 288；最后 12 个真实 token 与 20 个 mask 在后续 32-token decode block 中处理。

请求尚未准入时不会推进 `dllm_block_offset`；已提交 chunk 才按实际 `extend_range.length` 推进，避免重复初始化造成 position 漂移。

当对齐后的 pure-prefill 长度为零时，这是正常 phase 边界，不是 token 预算耗尽；资源不足或其他确实无法生成 token 的情况才返回 `NO_TOKEN`。

## Block-wise prefill attention

### 可见性语义

multi-block prefill 不能直接使用普通的双向 attention。设模型 block size 为 `B`，
query 和 key 在请求内的绝对位置分别为 `q`、`k`，其可见性为：

```text
visible(q, k) = floor(k / B) <= floor(q / B)
```

因此：

- 同一 dLLM block 内双向可见；
- query 可以看到所有此前 block；
- query 不能看到任何后续 block；
- 已提交的 prefix 对当前 extend query 全部可见。

例如 `B=2`、一次 extend 包含 6 个 token 时，当前段的 mask 为：

```text
1 1 0 0 0 0
1 1 0 0 0 0
1 1 1 1 0 0
1 1 1 1 0 0
1 1 1 1 1 1
1 1 1 1 1 1
```

`build_dllm_prefill_blockwise_mask()` 使用请求内绝对位置生成 bool mask。每个请求的
二维 mask 按 query-major 顺序 flatten，再按 batch 中的请求顺序拼接，以匹配
FlashInfer custom mask 的布局。异构 batch 中每个请求使用自己的 `prefix_len` 和
`extend_len`，不会在请求间共享 mask 区域。

如果 batch 中每个请求的 extend 都只落在一个 dLLM block 内，则普通 non-causal
attention 已满足上述语义，构造函数返回 `None`，继续使用无 mask fast path。只要任一
请求跨越 block 边界，整个 batch 就进入 custom-mask 路径。

### Ragged 主路径与 prefix cascade

默认 ragged prefill 将 attention 拆为两部分。设历史 prefix 长度为 `P`，当前 extend
长度为 `E`：

1. ragged wrapper 对当前 extend K/V 执行 `E × E` block-wise masked attention；
2. paged wrapper 对已缓存 prefix K/V 执行 `E × P` 的无 mask attention；
3. 使用两部分输出各自的 log-sum-exp，通过 FlashInfer `merge_state` 合并结果。

prefix token 的绝对位置一定早于当前 query，因而 prefix 分支可以整体可见；当前段的
未来 block 则由 `E × E` mask 隔离。该拆分避免为长 prefix 构造完整的
`E × (P + E)` mask，同时保留 radix/paged KV cache。

FlashInfer ragged prefill 使用两个独立的持久 wrapper：

- 无 mask wrapper 使用 `backend="auto"`，在支持的平台上可以选择 FA3；
- custom-mask wrapper 固定使用 `backend="fa2"`，因为当前 FA3 路径不支持该 mask。

planning 时选中的 wrapper 会记录在 `PrefillMetadata`，各 attention layer forward
继续使用同一个实例。这样可以避免 FlashInfer 0.6.12 在首次 plan 后固定 auto backend，
产生“无 mask 首先选择 FA3，后续 masked plan 错误复用 FA3”的顺序依赖。

### Paged correctness fallback

当 ragged prefill 被禁用时，paged wrapper 直接使用覆盖 prefix 与 extend 的
`E × (P + E)` block-wise mask。它与 ragged cascade 具有相同的可见性语义，但 mask
显存和构造开销为 `O(E(P + E))`，因此定位为正确性 fallback，而不是 multi-block
prefill 的默认高性能路径。

当前 multi-block custom mask 只支持 FlashInfer 的 single-wrapper full-attention 路径，
不与 sliding-window wrapper dispatch 或 multi-item scoring 组合；检测到这些组合时会
显式报错，而不是退化为错误的 attention。

## Forward mode 与 CUDA Graph

pure dLLM prefill 使用 `ForwardMode.EXTEND`，并通过 `dllm_config` 和
`is_dllm_prefill` 明确区别于普通模型 prefill。position、extend metadata、custom mask
和 KV 写入均按实际 extend 长度计算；decode 继续使用 `ForwardMode.DLLM_EXTEND`，
固定处理一个 `block_size`。

pure prefill 可以复用普通 prefill 的 Breakable CUDA Graph，但必须同时满足：

- scheduler 明确标记 `is_dllm_prefill=True`，且 forward mode 为 `EXTEND`；
- CUDA + FlashInfer attention backend；
- graph backend 为 Breakable；
- token 数精确命中已 capture bucket；
- 不使用 input embedding 或其他现有 graph 不支持的输入。

要求精确命中 bucket，是为了避免普通 prefill 的向上 padding 改变 dLLM 双向 attention
和 KV 写入语义。不满足条件时安全回退 eager；固定 shape 的 dLLM decode graph 不接收
pure prefill。graph replay 中继续传递 `dllm_config` 和 `is_dllm_prefill`，确保 metadata
初始化仍构造相同的 block-wise mask。

## 相关代码

| 文件 | 职责 |
| --- | --- |
| `python/sglang/srt/dllm/config.py` | `prefill_block_size` 的读取、默认化和校验。 |
| `python/sglang/srt/dllm/mixin/req.py` | cache match 后 phase 判定及已提交 chunk offset。 |
| `python/sglang/srt/dllm/mixin/scheduler.py` | incoming/staging 准备、phase 选择和 forward mode。 |
| `python/sglang/srt/managers/schedule_policy.py` | phase-aware 预算、对齐和 prefill 边界。 |
| `python/sglang/srt/managers/schedule_batch.py` | incoming 实际准入后的 staging 转换。 |
| `python/sglang/srt/model_executor/forward_batch_info.py` | dLLM decode 固定 block position。 |
| `python/sglang/srt/model_executor/runner/prefill_cuda_graph_runner.py` | pure prefill graph 条件及 eager 回退。 |
| `python/sglang/srt/dllm/attention.py` | 构造多 block prefill 的 request-wise flattened block mask。 |
| `python/sglang/srt/layers/attention/flashinfer_backend.py` | 将 block mask 接入 ragged/paged prefill，并保留 prefix cascade。 |

## 验证

`test/registered/unit/managers/test_prefill_adder.py` 覆盖 phase-aware 预算、非对齐尾巴、
decode 固定 block、incoming 生命周期、cache-match 后 phase 判定、未准入请求 offset
不推进、FlashInfer backend gate，以及 Breakable CUDA Graph capability gate。

`test/registered/unit/layers/test_flashinfer_dllm_prefill_mask.py` 共 11 项，覆盖 mask 公式、
异构 batch、单 block fast path、非对齐 prefix、masked/unmasked wrapper 隔离，以及以下
4 项 CUDA + FlashInfer 数值测试：

- production-shaped bf16 GQA ragged 输出与 dense reference 对齐；
- ragged prefix cascade 的输出和 LSE 与 dense reference 对齐；
- 修改未来 block 的 K/V 不影响此前 block 输出；
- paged fallback 输出与 dense reference 对齐。

```text
FLASHINFER_WORKSPACE_BASE=/tmp /root/miniconda3/envs/sglang-src/bin/python \
  test/registered/unit/layers/test_flashinfer_dllm_prefill_mask.py -v

FLASHINFER_WORKSPACE_BASE=/tmp /root/miniconda3/envs/sglang-src/bin/python \
  test/registered/unit/managers/test_prefill_adder.py -v
```

无 CUDA 环境下第一组为 7 passed、4 skipped；GPU 验收要求 11 项全部执行并通过。
第二组结果为 24 tests passed。

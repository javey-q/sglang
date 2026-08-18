# dLLM Prefill 性能优化方案

本文档针对当前分支 PR #31586 的性能风险进行分析，并给出保持短上下文性能、同时加速长上下文的实现方向。

## 1. 结论摘要

当前回归的本质不是简单的 backend 配置名从 `full` 变为 `breakable`，而是 dLLM pure prefill 的有效执行路径发生了变化：

```text
main:
  DLLM_EXTEND
    -> DecodeCudaGraphRunner
    -> Full CUDA graph
    -> 一次 replay 覆盖完整 forward

current branch:
  pure prefill -> EXTEND
    -> PrefillCudaGraphRunner
    -> Breakable CUDA graph
    -> 只捕获 transformer body
    -> LM head / logits processor 仍然 eager 执行
```

短上下文或较小 `prefill_block_size` 下，每轮处理的 token 数较少，BCG 的固定开销、调度器开销和 eager tail 开销无法被摊薄；长上下文下，多块 prefill 减少 scheduler round 数，收益才会超过这些固定开销。

## 2. 代码路径证据

### 2.1 main 与当前分支的 forward mode 差异

`ForwardMode.DLLM_EXTEND` 被 `is_cuda_graph()` 识别，因此会优先尝试 decode CUDA graph；普通 `EXTEND` 不属于 CUDA graph mode。

当前分支在 dLLM scheduler 中选择 pure prefill 时返回 `ForwardMode.EXTEND`：

- `python/sglang/srt/dllm/mixin/scheduler.py:31-75`
- `python/sglang/srt/model_executor/forward_batch_info.py:98-205`

`ModelRunner._forward_raw()` 的顺序是：先尝试 decode graph，再尝试 prefill graph，最后 eager：

- `python/sglang/srt/model_executor/model_runner.py:1388-1465`

因此 pure prefill 从 decode Full graph 转移到了 prefill runner。

### 2.2 当前 BCG prefill 不是完整 forward graph

`PrefillCudaGraphRunner.execute()` 会 monkey-patch `layer_model.forward()`，由 BCG replay transformer body，然后继续执行 outer model：

- `python/sglang/srt/model_executor/runner/prefill_cuda_graph_runner.py:1168-1239`

其中 LM head、logits processor 等 tail 仍在 eager 路径执行。BCG 本身还可能包含多个 graph segment 和 eager break。

### 2.3 dLLM prefill 的 graph eligibility 很严格

当前 dLLM pure prefill 只有在以下条件全部满足时才使用 BCG：

- `forward_mode == EXTEND`；
- `is_dllm_prefill=True`；
- token 数命中 exact bucket；
- backend 必须是 `BreakableCudaGraphBackend`；
- CUDA + FlashInfer；
- 不存在不支持的输入模式。

代码位置：

- `python/sglang/srt/model_executor/runner/prefill_cuda_graph_runner.py:716-780`

因此简单设置 `cuda_graph_config[prefill].backend = full` 目前不能让 dLLM pure prefill 使用 Full graph；现有 eligibility gate 会直接拒绝。

### 2.4 custom mask 当前在 GPU 上逐 request 构造

`build_dllm_prefill_blockwise_mask()` 在 request loop 中创建 GPU `arange`、整除和比较张量：

- `python/sglang/srt/dllm/attention.py:6-64`

FlashInfer backend 在每次 metadata 初始化时调用该函数：

- `python/sglang/srt/layers/attention/flashinfer_backend.py:996-1036`

这会增加小 kernel launch、临时张量分配和 host-device 调度成本，尤其影响小 chunk。

## 3. 现有基准证据

本地矩阵位于：

- `benchmark_results/pr_prefill_perf_matrix_20260720/pivot_input_throughput.tsv`
- `benchmark_results/pr_prefill_perf_matrix_20260720/pivot_mean_ttft_ms.tsv`

H200、TP=1、`num_prompts=16`、`max_concurrency=1` 的 input throughput：

| input tokens | main | pbs32 | pbs128 | pbs512 | pbs1024 | pbs2048 | pbs4096 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 528 | 416 | 451 | 438 | 441 | 441 | 442 |
| 256 | 917 | 774 | 897 | 846 | 850 | 847 | 849 |
| 512 | 1832 | 1545 | 1967 | 1968 | 1982 | 1989 | 1977 |
| 1024 | 2563 | 2251 | 2975 | 3245 | 3250 | 3268 | 3299 |
| 4096 | 4784 | 4251 | 8561 | 10759 | 12420 | 12810 | 12320 |
| 8192 | 5517 | 4543 | 11845 | 18841 | 23248 | 25174 | 25671 |
| 16384 | 5193 | 3757 | 13103 | 22751 | 27713 | 29577 | 29577 |

关键现象：

- pbs32 在 128/256/512 token 输入下明显低于 main；
- pbs128 在 128 token 输入下仍低于 main，但在 512 token 后开始超过 main；
- pbs1024 及以上在长上下文收益显著；
- pbs32 的 graph miss 为 0，但性能仍低于 main，说明问题主要不是 graph 没命中，而是每轮执行路径固定开销过高。

## 4. 优先级 P0：短上下文 legacy fallback

这是最小风险、最建议优先实现的方案。

### 4.1 增加独立的执行路径字段

不要把语义 phase 和执行 backend 混在 `is_dllm_prefill` 中。建议增加：

```python
class DllmExecutionPath(Enum):
    LEGACY_BLOCK = "legacy_block"
    MULTI_BLOCK_PREFILL = "multi_block_prefill"
```

在 `ForwardBatch` 或 dLLM scheduler batch metadata 中同时保留：

```text
dllm_phase: PREFILL / DECODE
dllm_execution_path: LEGACY_BLOCK / MULTI_BLOCK_PREFILL
```

这样 `process_batch_result_dllm()` 仍然可以根据语义 phase 处理结果，而 model runner 可以根据 execution path 选择 `DLLM_EXTEND` 或 `EXTEND`。

### 4.2 默认策略

建议：

```text
prefill_block_size == block_size
    -> 始终 LEGACY_BLOCK

prefill_block_size > block_size
    -> 使用 AUTO 策略
```

`AUTO` 初版可以使用剩余 pure-prefill token 数作为阈值：

```text
remaining_tokens < min_multiblock_tokens
    -> LEGACY_BLOCK / DLLM_EXTEND / Decode Full graph

remaining_tokens >= min_multiblock_tokens
    -> MULTI_BLOCK_PREFILL / EXTEND / Prefill BCG
```

H200 矩阵显示初始阈值可以从 `256` 或 `512` tokens 开始标定。最终更推荐使用成本模型，而不是永久硬编码：

```text
legacy_rounds = ceil(remaining_tokens / block_size)
multi_rounds  = ceil(remaining_tokens / prefill_block_size)

只有当节省的 round 成本
    > BCG replay + scheduler + eager tail 固定成本
时才切换到 multi-block。
```

当长 prompt 进入最后一个较小的残余区间时，应再次切回 legacy path，避免尾部小 chunk 触发 BCG 固定开销。

### 4.3 batch 同质性

一个 forward batch 不能混合 `LEGACY_BLOCK` 与 `MULTI_BLOCK_PREFILL`。建议：

- scheduler 先选择执行路径；
- 只从同一路径的 request 中构造 batch；
- 如果同时存在两类 request，下一轮再处理另一类。

这样可以保留当前“单轮 phase 同质”的正确性约束。

### 4.4 正确性注意事项

legacy path 本质上会恢复旧的 `DLLM_EXTEND` 行为，因此必须单独验证：

- prompt KV 是否只提交真实 token；
- mask block 是否没有提前提交到 radix cache；
- `dllm_block_offset` 是否正确；
- FDFO / LowConfidence 的结果处理是否与 pure prefill 语义一致。

建议不要直接用 `dllm_is_prefill=False` 偷换语义，而是增加独立 execution path 字段。

## 5. 优先级 P1：Full graph 支持 dLLM pure prefill

这是更彻底、但改动较大的方案。

### 5.1 FlashInfer Full-CG metadata

需要让 `init_forward_metadata_out_graph()` 支持：

```text
ForwardMode.EXTEND + is_dllm_prefill=True
```

并为 Full-CG prefill wrapper 增加：

- 静态 `custom_mask_buf`；
- 静态 `mask_indptr`；
- 静态 `qo_indptr` / `kv_indptr`；
- replay 前更新 mask 内容和 indptr。

由于 dLLM block mask 对 padding 非常敏感，仍然必须使用 exact token buckets，不能向上 padding。

### 5.2 捕获完整 outer model

当前 Full prefill runner 也倾向于只捕获 layer body。若目标是恢复 main 的短上下文性能，最终需要捕获完整 outer model，包括：

- transformer body；
- LM head；
- logits processor；
- dLLM 所需的固定输出 buffer。

否则只能减少 BCG segment 数，eager tail 仍会保留。

可考虑复用 decode Full graph 的固定 slot 思路：

- 固定 request slots；
- 未使用 slot 使用 zero-length sentinel；
- exact token bucket；
- replay 时只刷新静态输入和 metadata。

## 6. 优先级 P1：BCG 单请求 full-forward fast path

当前 benchmark 使用 `max_concurrency=1`，适合增加单请求特化：

```text
batch_size == 1
    -> 捕获完整 model forward

batch_size > 1
    -> 保持当前 body-only BCG
```

需要将 graph key 区分为：

```text
(token_bucket, body_only)
(token_bucket, full_forward)
```

这样不会破坏多请求 BCG 的复用，同时可消除短上下文单请求中的 eager LM-head tail。

## 7. 优先级 P2：降低 scheduler round 开销

当前每轮会执行：

```python
self._fetch_waiting_reqs()
self.dllm_manager.init_next_round(self.tree_cache)
is_prefill = bool(self.dllm_manager.get_prefill_requests())
```

`init_next_round(tree_cache)` 会遍历 waiting requests 并重新进行 cache matching / phase preparation。建议：

1. 新 incoming request 只在首次 admission 时做 prefix matching；
2. staging request 只更新必要的 block offset，不重复完整 cache matching；
3. 给 request 增加 `prefix_matched_version` 或 dirty bit；
4. manager 内部维护 prefill/decode 两个队列或计数，避免每轮 list comprehension；
5. 只有 prefix cache、output ids 或 incomplete block 发生变化时重新 determine phase。

该项对长上下文收益有限，但对 32/64 token chunk 的短请求很重要。

## 8. 优先级 P2：优化 custom mask

### 8.1 CPU 构造后一次性上传

`prefix_lens` 和 `extend_lens` 已经通常是 CPU list，可以改为：

```python
query_positions = torch.arange(..., device="cpu")
key_positions = torch.arange(..., device="cpu")
mask = (...).flatten().to(device, non_blocking=True)
```

最好使用 pinned host buffer，并避免每个 request 单独触发 GPU kernel。

### 8.2 模板缓存

可以按以下 key 缓存：

```text
(prefix_len, extend_len, block_size, include_prefix)
```

对于常见的 block-aligned pure prefill chunk，mask pattern 高度重复，可直接复用模板或写入预分配的 GPU buffer。

### 8.3 保留 no-mask fast path

当 extend range 不跨 dLLM block 时，应继续返回 `None`，使用 FlashInfer 的普通 non-causal path。

## 9. 优先级 P3：迁移到 native Block Extend Attention

当前 custom mask 是临时兼容路径。FlashInfer 的 native Block Extend Attention 方向可以：

- 在 register 中计算 block visibility；
- 避免 O(q×k) mask tensor；
- 跳过不可见 KV tile；
- 将 cascade 的多次 kernel launch 合并为单个 block-extend kernel；
- 支持大于 `block_size` 的 chunk。

长期应将 SGLang 的 scheduler-facing multi-block 逻辑保留，但替换 attention operator，不再长期维护显式 custom mask。

## 10. 建议的实现顺序

### Phase A：低风险性能修复

- [ ] 增加 `LEGACY_BLOCK / MULTI_BLOCK_PREFILL` execution path；
- [ ] `prefill_block_size == block_size` 默认走 legacy；
- [ ] 增加 `min_multiblock_tokens` 或 AUTO cost model；
- [ ] 增加短上下文 fallback 单测和 phase/path 单测。

### Phase B：运行时开销优化

- [ ] 统计 scheduler round、tokens/round、BCG segment、eager break；
- [ ] 延迟或缓存 `init_next_round(tree_cache)`；
- [ ] CPU 构造 custom mask；
- [ ] 增加 mask template cache。

### Phase C：完整 graph 路径

- [ ] Full-CG dLLM metadata/custom-mask 支持；
- [ ] Full prefill 完整 outer forward capture；
- [ ] BCG 单请求 full-forward variant；
- [ ] 与 decode Full graph 做端到端对比。

### Phase D：算子升级

- [ ] 集成 FlashInfer native Block Extend Attention；
- [ ] 删除长期 custom-mask workaround；
- [ ] 重新评估最优 `prefill_block_size`。

## 11. 验证矩阵

固定模型、GPU、FlashInfer、seed，至少测试：

```text
prefill_block_size = 32, 128, 512, 1024
input_len          = 128, 256, 512, 1024, 4096, 8192, 16384
max_concurrency    = 1, 4, 16
```

每个 case 记录：

- input throughput；
- request/output throughput；
- mean TTFT；
- scheduler round 数；
- 每轮实际 token 数；
- `DLLM_EXTEND` / `EXTEND` 比例；
- decode Full graph / prefill BCG / eager 比例；
- BCG segment 数和 eager break 数；
- `init_next_round`、`PrefillAdder`、metadata init、mask build、LM-head tail CPU 时间；
- exact bucket miss 原因。

## 12. 验收标准

建议以以下标准作为 PR 性能门槛：

- `input_len <= 512`：吞吐不低于 main，允许误差不超过 2%；
- `input_len >= 4k`：保持当前 multi-block 的主要收益；
- 非对齐长度：无 KV offset、prefix cache、future-block isolation 回归；
- GSM8K 或等价质量测试无显著下降；
- BCG graph miss 为 0 的 case 不应再出现明显的 main 对比回归。

最重要的诊断原则：不能只看 CUDA graph hit/miss。当前 pbs32 已经 graph hit，但仍然慢，说明主要问题是每轮 replay 的固定路径成本，而不是 graph 未命中。


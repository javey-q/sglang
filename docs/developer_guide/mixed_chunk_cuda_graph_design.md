# MIXED / DECODE / EXTEND 与 CUDA Graph 调度设计

本文描述 text-generation scheduler 中三种常用 forward mode（`DECODE`、`EXTEND`、`MIXED`）如何形成 batch，以及它们分别走哪条 CUDA Graph 路径。本文以当前实现为准，重点解释 chunked prefill 与 `enable_mixed_chunk` 的关系。

## 结论速览

| Forward mode | 批次语义 | 主 shape 轴 | 默认 CUDA Graph 路径 |
| --- | --- | --- | --- |
| `DECODE` | 每个运行中请求生成一个 token | request batch size `B` | `DecodeCudaGraphRunner` |
| `EXTEND` | 一个或多个请求推进若干 prompt token | 总 extend token 数 `T` | `PrefillCudaGraphRunner` |
| `MIXED` | prefill 的 extend token 加上每个 decode 请求的一个 token | 总 extend token 数 `T` | `PrefillCudaGraphRunner`，内部作为 `EXTEND` 重放 |

`chunked_prefill_size=C` 是 MIXED 的**总 token 预算**，而不是只给 prefill 的预算。若本轮混入 `D` 个 decode 请求，则调度器会先从 chunk token 预算中扣除 `D`：

```text
prefill token 数 + decode token 数 <= C
prefill token 数                  <= C - D
```

这使 MIXED batch 的总 token 数落在 prefill CUDA Graph 已捕获的 token bucket 内；它不会进入按 decode batch size 捕获的传统 Decode CUDA Graph。

## 1. 调度状态与 mode

`Scheduler.init_chunked_prefill()` 将正数的 `chunked_prefill_size` 视为开启 chunked prefill，并维护：

- `chunked_req`：唯一正在跨 iteration 处理的长 prompt 请求；
- `is_mixed_chunk`：`chunked_prefill_size` 已开启且 `enable_mixed_chunk=True`；
- `running_batch`：已完成 prefill、可持续 decode 的请求。

相关实现：`python/sglang/srt/managers/scheduler.py` 中的 `init_chunked_prefill()`。

三种 mode 定义在 `python/sglang/srt/model_executor/forward_batch_info.py`：

```text
EXTEND：对已有 KV prefix 追加多个 token（通常称 prefill）
DECODE：对每个请求追加一个生成 token
MIXED ：同一 forward 同时包含 EXTEND 行和 DECODE 行
```

从 attention 的角度，MIXED 中 decode 行并不特殊：它只是 `extend_len=1` 的 extend 行；区别只存在于 scheduler 的请求生命周期和结果处理。

## 2. Scheduler 的优先级与 chunk 生命周期

每轮 `get_next_batch_to_run()` 先尝试 `get_new_batch_prefill()`。只有没有可运行的 prefill batch 时，才将 `running_batch` 组织为 `DECODE` batch。因此总策略是：

```text
可运行的 prefill / chunk 存在  -> 优先运行 EXTEND 或 MIXED
没有 prefill                    -> 运行 DECODE
```

### 2.1 长请求首次进入

`PrefillAdder.add_one_req()` 比较请求尚未命中的 input 长度与 `rem_chunk_tokens`。长度超过预算时，将本轮 `extend_range` 截为一个 page-aligned chunk，并将该请求记录为 `new_chunked_req`。中间 chunk 不为未来 decode 预留输出 token；最后一个 chunk 才完成 prefill。

```text
第 1 轮：prefix=0         , extend=[0, C)
第 2 轮：prefix=C         , extend=[C, 2C)
...
最后轮：prefix=(n-1)C    , extend=[(n-1)C, prompt_end)
```

在每个中间 chunk 完成后，下一调度轮会缓存新产生的 KV，再以更新后的 prefix 继续处理。`chunked_req` 被从普通 prefill-to-running 合并中排除，避免尚未完成 prefill 的请求进入 decode batch。

关键路径：

```text
Scheduler.get_next_batch_to_run
  -> stash_chunked_request(chunked_req)
  -> Scheduler._get_new_batch_prefill_raw
  -> chunked_req.init_next_round_input()
  -> PrefillAdder.add_chunked_req(chunked_req)
```

结果处理器通过 `inflight_middle_chunks` 判断中间 chunk；中间 chunk 的 logits 不会作为生成 token 流式返回，计数清零后才允许最终 prefill 结果进入正常生成生命周期。

### 2.2 未开启 mixed chunk

仅开启 chunked prefill 时，长请求会连续产生 `EXTEND` batch。因为 scheduler 以 prefill 优先，已有 decode 请求通常等长 prompt 的全部 chunk 完成后才会得到下一次独立 `DECODE`。

```text
EXTEND(chunk 0) -> EXTEND(chunk 1) -> ... -> EXTEND(last) -> DECODE
```

它的主要收益是限制单次 prefill 的峰值显存和单次执行时间；它不保证 decode token 的低 inter-token latency。

## 3. MIXED batch 如何构造

当 `enable_mixed_chunk=True` 且 running batch 不为空时，scheduler 以当前运行请求数 `D` 创建 `PrefillAdder`：

```python
PrefillAdder(..., chunked_prefill_size, running_bs, ...)
```

`PrefillAdder` 随即执行：

```python
rem_input_tokens -= num_mixed_decode_tokens
rem_chunk_tokens -= num_mixed_decode_tokens
```

所以若 `C=4096`、`D=128`，本轮预填充最多约为 `3968` token，并与 128 个 decode token 合为一个最多 4096-token 的 forward。实际数值还会受 page 对齐、KV 容量、`max_prefill_tokens` 和其他 admission 条件限制。

混合前，prefill 部分已由 `prepare_for_extend()` 生成其 token 和 KV 写入位置。随后 scheduler：

1. 对 `running_batch` 调用 `prepare_for_decode()`，每个请求分配一个新 KV slot；
2. 调用 `new_batch.mix_with_running(running_batch)`；
3. 将 mode 改为 `MIXED`；
4. 将 decode 请求的 `extend_len` 追加为 1；
5. 将 `extend_num_tokens` 增加 `D`。

`resolve_forward_inputs()` 在 forward 前把两部分 token 拼接：prefill token 来自 pinned CPU staging，decode token 来自上一轮输出的 `future_map.output_tokens_buf`。

```text
input_ids = [prefill tokens ...][decode request 0 last token]...[decode request D-1 last token]
```

### 3.1 MIXED 的退化条件

以下情况下 scheduler 不混合，而保持普通 EXTEND：

- prefill 或 running batch 请求需要 logprob；
- prefill batch 使用 `input_embeds`；
- 某些 speculative/DP attention 同步路径要求 prefill、decode 分离；
- 后端或模型配置自动禁用 mixed chunk。

此时 chunked prefill 仍然有效，只是不能在同一模型 forward 内推进已有 decode 请求。

## 4. 为什么固定总 token 数不足以复用 Decode CUDA Graph

CUDA Graph 可以接受**内容**不同的输入，只要这些输入被复制到地址稳定、shape 固定的 buffer。问题不在于 `seq_lens`、`positions` 或 KV index 的数值变化；问题是 MIXED 只固定总 token 数 `T`，并未固定 request 数 `B` 和 ragged 分段结构。

例如都令 `T=4096`：

```text
case A: 4095-token prefill + 1 decode                   -> B=2
case B: 3968-token prefill + 128 decode                 -> B=129
case C: 多个 prefill 请求 + 128 decode                  -> B 可继续变化
```

这会使下列 request-major 元数据的实际 shape 变化：

```text
seq_lens, prefix_lens, extend_seq_lens, extend_start_loc: [B]
req_pool_indices, page-table metadata, last-token indices: [B]
```

而传统 Decode CUDA Graph 的结构是均匀的：`B` 条请求、每条 query length 恒为 1。它按 `B` 捕获静态输入 buffer、attention metadata、logits/sampling 输出；`ForwardMode.is_cuda_graph()` 也只把 `DECODE`（以及特定 speculative/idle mode）送入 Decode runner，明确排除 `MIXED`。

若要让传统 full graph 支持 MIXED，至少还需要固定 `Bmax`，把所有 request-major buffer pad 到上界，并让所有 attention backend、KV 操作和采样路径正确忽略 dummy request。这在理论上可行，但会增加图数、显存、无效工作和后端兼容性负担。

## 5. 哪些算子真正受 ragged metadata 影响

下表区分 token-major compute 与 request-major/ragged compute。

| 算子类别 | 固定 `T` 是否通常足够 | 依赖的动态信息 |
| --- | --- | --- |
| embedding、RMSNorm、residual、SiLU | 是 | token 内容 |
| QKV / output projection、dense MLP GEMM | 是 | 通常只有 token 轴 `T` |
| RoPE | 是 | `positions[T]` 的值可变，shape 不必变 |
| KV 写入 | 常可做到 | `out_cache_loc[T]` 可作为固定长度的动态索引输入 |
| varlen/paged attention | 否 | `B`、每段 `q_len`、`seq_len`、prefix、页表、indptr、最大 KV 长度 |
| attention metadata/plan/workspace | 取决于后端 | 有些 backend 的 plan 或 workspace 由分段/页数决定 |
| last-token logits 选择、sampling、grammar | 否，仅固定 `T` 不够 | 每请求一个结果，shape 主要由 `B` 决定 |
| MoE dispatch | 取决于实现 | router 结果会改变 expert 分桶及 workspace 使用 |

因此，MIXED 的难点不是让 GEMM 支持 `T=C`，而是让 ragged attention 和 request-level 后处理在动态分段下维持可重放的执行契约。

## 6. 两条 CUDA Graph 路线

### 6.1 Decode CUDA Graph：按请求数 bucket

`DecodeCudaGraphRunner` 捕获的主 shape 是 decode batch size。运行时它将真实 batch size 向上 bucket 到已捕获大小，填充固定地址 buffer，图外更新 attention metadata，随后 replay。

```text
DECODE batch (B)
  -> nearest decode capture bucket B'
  -> static input / seq-len / KV-location buffers sized by B'
  -> init_forward_metadata_out_graph
  -> replay decode graph
  -> 只返回真实 B 条请求的结果
```

这条路径适合每请求固定一个 query token 的 decode。若 `disable_cuda_graph_padding=True`，只有精确已捕获的 shape 才可使用图；否则允许向上 padding。

### 6.2 Prefill CUDA Graph：按总 extend token 数 bucket

`PrefillCudaGraphRunner` 的 capture size 表示 token 数。`cuda_graph_config.prefill.bs` 是预捕获 token bucket 列表，默认最大值通常与 `chunked_prefill_size` 对齐。

```text
EXTEND 或 MIXED batch (T)
  -> nearest prefill capture bucket T'
  -> 将 token-major 输入复制/填充到静态 buffer
  -> 用真实 request-major metadata 初始化 attention
  -> replay prefill backend
  -> 将 token-major输出裁回前 T 个 token
```

对 MIXED，runner 在构造静态 `ForwardBatch` 时把 `forward_mode` 和 `global_forward_mode` 从 `MIXED` 规范化为 `EXTEND`。这是合法的，因为 attention 所需语义已经由 `extend_seq_lens`、`extend_prefix_lens`、`extend_start_loc`、`seq_lens`、KV locations 等 metadata 表达；decode 行只是 `extend_len=1`。

Prefill runner 的准入会拒绝某些动态特性，例如 embedding override、部分 input embeds、部分 logprob、hidden-state capture mode 不匹配、DP rank 没有本地 token，或 context-parallel strategy 生效。拒绝时由 `EagerRunner` 执行，不改变调度语义。

## 7. Prefill CUDA Graph 的两个 backend

prefill phase 不支持传统 `full` backend，因为完整 prefill forward 的变长/ragged 特征不适合单个固定 shape 的整体图。当前允许：

| Backend | 执行方式 | 对 MIXED 的意义 |
| --- | --- | --- |
| `breakable` | 在 attention/Mamba 等边界把图切成多个 segment；重放前/段间可重算 metadata | 将难以固定的部分放在图外或 graph break，保留其余 dense compute 的捕获收益 |
| `tc_piecewise` | `torch.compile` 切分模型；可捕获分片图重放，split op 保持 eager | 把动态/不适合捕获的 split op 留在 eager 路径，token-major分片按 token bucket 复用 |
| `disabled` | 不创建 prefill graph runner | EXTEND/MIXED 走 eager |

`tc_piecewise` 的实际 capture/replay 对象由 torch.compile 的分片 backend 管理；PrefillCudaGraphRunner 负责 dummy batch、静态输入 buffer、metadata 生命周期和输出裁剪。`breakable` 则维护多个 CUDA Graph segment，并允许在 graph break 处 eager 执行。

## 8. 端到端时序

下面以 `C=4096`、已有 `D=128` 条 decode 请求、一个超长 prompt 为例。

```text
调度 round k
  1. 计算 prefill budget = 4096 - 128 = 3968
  2. 为长 prompt 选 [prefix, prefix+3968) 的 chunk
  3. 为 128 条运行请求各准备 1 个 decode token/KV slot
  4. 合并为 MIXED，T=3968+128=4096
  5. resolve_forward_inputs 拼接两类 input_ids
  6. Prefill CUDA Graph 选择 token bucket 4096；否则 eager
  7. 中间 chunk：缓存 prompt 新 KV，但不向该 prompt 流式返回首 token
  8. decode 行：正常采样、更新 output_ids、下轮仍留在 running_batch

最后一个 chunk
  1. chunked_req 清空
  2. prompt 取得第一个生成 token
  3. 下一轮它与其他请求一起成为普通 DECODE 行
  4. 没有新的 prefill 时，Decode CUDA Graph 按 B bucket 重放
```

## 9. 源码导航

| 关注点 | 文件 / 符号 |
| --- | --- |
| mode 定义及 Decode CG 准入 mode | `model_executor/forward_batch_info.py`, `ForwardMode` |
| chunked prefill 调度与 MIXED 合并 | `managers/scheduler.py`, `get_next_batch_to_run`, `_get_new_batch_prefill_raw` |
| chunk 截断、预算扣减 | `managers/schedule_policy.py`, `PrefillAdder` |
| EXTEND/MIXED 的 batch metadata | `managers/schedule_batch.py`, `prepare_for_extend`, `mix_with_running` |
| prefill/decode token 拼接 | `managers/overlap_utils.py`, `resolve_forward_inputs` |
| mode 到 runner 的 dispatch | `model_executor/model_runner.py`, `_forward_raw` |
| Decode graph 的 B bucket/replay | `model_executor/runner/decode_cuda_graph_runner.py` |
| Prefill graph 的 T bucket、MIXED→EXTEND | `model_executor/runner/prefill_cuda_graph_runner.py` |
| prefill backend 选择 | `model_executor/runner_backend/utils.py`, `resolve_prefill_backend` |
| breakable / tc_piecewise backend | `model_executor/runner_backend/breakable_cuda_graph_backend.py`, `tc_piecewise_cuda_graph_backend.py` |

## 10. 调试建议

排查某次 forward 实际走哪条路径时，建议按以下顺序记录：

1. scheduler 输出的 `forward_mode`、`batch_size`、`extend_num_tokens`、`extend_lens`；
2. 对 MIXED，确认 `sum(extend_lens) == extend_num_tokens`，并检查 decode 行是否均为 1；
3. 检查实际 token 数是否不超过 `prefill.max_bs` 的最大 capture bucket；
4. 观察 `PrefillCudaGraphRunner.can_run_graph()` 的拒绝原因；
5. 在没有 prefill 时，检查 `DecodeCudaGraphRunner.can_run_graph()` 的 batch-size bucket 和 padding 策略；
6. 将同一 workload 分别以 prefill graph disabled、decode graph disabled、两者 enabled 的配置运行，区分调度问题与图捕获兼容性问题。


## 2. Breakable CUDA Graph

当前项目的 Breakable CUDA Graph 会在 attention 边界主动断图，其定义直接说明了这一点：

python/sglang/srt/model_executor/runner_backend/breakable_cuda_graph_backend.py:56

一次 Transformer forward 被拆成：

CUDA Graph segment 0
  embedding / norm / QKV projection
              │
              ▼
      eager attention
              │
              ▼
CUDA Graph segment 1
  output projection / residual / MLP / next-layer QKV
              │
              ▼
      eager attention
              │
              ▼
CUDA Graph segment 2
  ...

具体来说，RadixAttention 在 Breakable 模式下调用：

python/sglang/srt/layers/radix_attention.py:135

而 attention 函数被 eager_on_graph(True) 包装：

python/sglang/srt/layers/radix_attention.py:256

capture 遇到这个函数时会：

1. 结束当前 CUDA Graph segment；
2. eager 执行 attention；
3. 将 attention 保存为一个 break function；
4. 开始捕获下一个 segment。

Replay 则按如下顺序运行：

segment_0.replay()
attention_break_fn()   # 重新执行
segment_1.replay()
attention_break_fn()   # 重新执行
segment_2.replay()

实现位于 python/sglang/srt/model_executor/runner_backend_utils/breakable_cuda_graph/
breakable_cuda_graph.py:245。

关键区别是：attention_break_fn() 每次 replay 都真的重新调用 attention backend，而不是重放
capture 时录制的 attention kernel。

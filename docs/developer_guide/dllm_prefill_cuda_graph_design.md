# dLLM Prefill CUDA Graph 正式支持方案

## 状态与范围

本文是后续开发交接文档，描述如何在现有 dLLM chunked prefill 实现上正式启用
Prefill CUDA Graph。本文不表示功能已经实现。

首期范围：

- CUDA；
- LLaDA2 / `LowConfidence`；
- FlashInfer attention backend；
- `PrefillCudaGraphRunner` 的 `breakable` backend；
- 纯 dLLM prefill，即输入中不含 mask 的 `ForwardMode.EXTEND`；
- 优先支持精确命中的 token capture bucket。

暂不支持：

- NPU/Ascend；
- 带 mask 的 dLLM decode；
- 将可变长度 prefill 送入固定 block 的 dLLM Decode CUDA Graph；
- 未验证的 HIP、其他 attention backend 和 `tc_piecewise` prefill backend；
- 首期不承诺向上 bucket padding 的正确性。

关联文档：

- `docs/supported_models/text_generation/dllm_prefill_scheduling_design.md`：
  dLLM prefill/decode 调度、chunk 边界和 forward mode；
- `docs/developer_guide/mixed_chunk_cuda_graph_design.md`：普通
  `EXTEND/MIXED` Prefill CUDA Graph 的 token bucket、metadata 和 replay 契约。

## 背景

dLLM prefill chunk 已从 decode `block_size` 中解耦。纯 prefill 使用普通
`ForwardMode.EXTEND`，decode 保留 `ForwardMode.DLLM_EXTEND`：

```text
pure prefill: EXTEND, input 不含 mask, chunk 可为 32/128/512/1024...
decode:       DLLM_EXTEND, 每请求固定一个 block_size block
```

这个拆分是必要的。Decode CUDA Graph 按请求数捕获，并假定每请求固定处理
`block_size` 个 token；大 prefill chunk 不能复用它。

但是当前代码又在 `PrefillCudaGraphRunner.can_run_graph()` 中拒绝所有携带
`dllm_config` 的 batch：

```python
if forward_batch.dllm_config is not None:
    return False
```

结果是纯 prefill 即使满足普通 Prefill CUDA Graph 的 `EXTEND` 契约，也只能走
eager。当前服务仍会在启动时捕获 prefill graphs，所以还额外付出了捕获时间和
显存，却无法在 dLLM prefill 中 replay。

## 已确认的问题与证据

### PBS=32 不再等价于旧版

相同 benchmark 中：

```text
原版：          6022 个 32-token batch，全部 cuda graph=True
改动版 PBS=32： 6022 个 32-token batch，其中 5599 个 prefill graph=False，
                423 个 decode graph=True
```

两版 round 数和 chunk shape 相同，主要差异是 prefill 从 graph replay 变成 eager。
日志中单个稳定的 32-token prefill 大致从 5 ms 增加到 20 ms。2048-token prompt
约有 64 个 chunk，额外开销约 960 ms，与实测 TTFT 差 956.9 ms 基本一致。

原始证据：

- `benchmark_results/prefill_bs32_n32_original_verify/server.log`；
- `benchmark_results/prefill_block_size_compare/server_pbs32.log`；
- `benchmark_results/prefill_block_size_compare/SUMMARY.md`。

### 当前 dLLM 模型已经成功捕获普通 prefill graphs

LLaDA2 服务启动日志显示 `breakable` backend 已成功捕获 4 到 8192 token 的
58 个 prefill buckets，耗时约 7.15 秒、占用约 1.59 GiB graph memory。没有捕获
失败；运行时只是被上述 `dllm_config` gate 拒绝。

这证明至少在当前 CUDA/H200/LLaDA2/FlashInfer 环境中，模型主体能够完成普通
prefill graph capture。仍需通过 replay parity test 验证线上 metadata、KV 写入
和 padding 语义。

## 为什么纯 dLLM prefill 可以复用普通 Prefill CUDA Graph

`LowConfidence.run()` 首先检查 mask。纯 prefill 没有 mask，仅执行一次：

```text
model_runner.forward(forward_batch)
```

然后返回空 `next_token_ids`。算法控制循环位于 graph 外；graph 只需要覆盖普通
model forward 和 KV 写入。

当前普通 prefill graph 已具备：

- 以总 token 数 `T` 选择 capture bucket；
- 静态 `input_ids`、`positions`、`out_cache_loc` buffer；
- replay 前刷新真实 `seq_lens`、`extend_seq_lens`、`extend_prefix_lens`、
  `extend_start_loc` 和 request pool indices；
- 非零 cached prefix；
- Breakable backend 的多请求 replay；
- graph 外执行 request-dependent logits processor。

FlashInfer 的普通 `EXTEND` 路径也满足 LLaDA2 的核心语义：

- `AttentionType.ENCODER_ONLY` 使用非 causal attention；
- `is_dllm_model=True` 时 encoder-only attention 仍保存 KV；
- 普通 extend metadata 使用真实 prefix 和 extend length，而不是固定
  `block_size`；
- prefill 阶段不需要 `full_logits`，因为 `LowConfidence` 会忽略该 forward 的
  logits；带 mask 的 decode 仍使用 `DLLM_EXTEND` 和 full logits。

## 必须维持的不变量

1. 只有纯 prefill 可以进入 Prefill CUDA Graph。
2. 带 mask 的 batch 必须继续使用 `DLLM_EXTEND` 和 Decode CUDA Graph/eager
   fallback，不能进入普通 `EXTEND` graph。
3. decode 每请求的 query length 始终等于 `block_size`。
4. prefill 的 committed frontier 继续满足 page/block 对齐，不能跨入 mask block。
5. graph replay 与 eager 必须产生一致的有效 KV cache；不能只比较最终文本。
6. graph bucket 的 padding token 不能参与真实 token 的双向 attention，也不能覆盖
   有效 KV slot。
7. 不支持的 device/backend/feature 必须安全回退 eager，而不是尝试 replay。

## 推荐设计

### 1. 给 ForwardBatch 增加明确的 phase 标志

不要长期使用以下条件推断纯 prefill：

```python
forward_batch.dllm_config is not None and forward_batch.forward_mode == EXTEND
```

建议从 scheduler/batch 构造阶段传递明确字段，例如：

```python
is_dllm_prefill: bool = False
```

传播路径：

```text
SchedulerDllmMixin._create_dllm_batch
  -> ScheduleBatch
  -> ForwardBatch.init_new
  -> PrefillCudaGraphRunner.can_run_graph/load_batch
```

设置规则：只有 scheduler 已选择 pure prefill phase 且生成 `EXTEND` batch 时为
`True`。decode 和普通请求均为 `False`。

理由：

- 避免未来其他 dLLM 算法使用 `EXTEND` 时被错误放行；
- 避免在 `can_run_graph()` 中再次对 GPU input 做 `mask_id` 检查和同步；
- 测试和 metrics 能直接区分 dLLM prefill/decode；
- 可以明确传入 static `ForwardBatch`，减少 capture/replay guard 差异。

最小验证原型可以先使用
`dllm_config is not None && forward_mode == EXTEND`，但正式提交应增加显式 phase
标志。

### 2. 将全量拒绝改为 capability gate

移除“只要存在 `dllm_config` 就拒绝”的规则，改为：

```text
非 dLLM batch：沿用现有 can_run_graph 逻辑

dLLM batch：
  is_dllm_prefill == True
  forward_mode == EXTEND
  device == CUDA
  attention backend == FlashInfer
  prefill graph backend == breakable
  input_embeds is None
  replace_embeds is None
  不请求不兼容的 logprob/hidden states
  num_tokens <= max_num_tokens
  首期：num_tokens 精确存在于 capture_num_tokens
```

不满足任意条件时返回 `False`，继续走 eager。decode 的 mode 是
`DLLM_EXTEND`，本来就不会进入 ModelRunner 的 prefill graph dispatch。

capability 判断应使用已有 backend/device 类型或统一 capability API，避免依赖容易
变化的字符串。若当前类层次不方便，可先添加局部 helper。

### 3. static ForwardBatch 保留 dLLM prefill 身份

`PrefillCudaGraphRunner.load_batch()` 构造 `static_forward_batch` 时，应复制：

```python
dllm_config=forward_batch.dllm_config
is_dllm_prefill=forward_batch.is_dllm_prefill
```

当前模型主体多数通过 server/model runner 上的 dLLM config 分支，未必直接读取
`ForwardBatch.dllm_config`；仍建议保留，以保证 replay batch 与 live batch 语义完整，
并避免后续 backend/模型代码出现静默差异。

capture dummy batch 可以设置 `is_dllm_prefill=False`，因为捕获的是通用 `EXTEND`
模型主体；如果 Dynamo/backend guard 要求字段一致，应将该字段保持在 graph 外，或
为 dLLM prefill 设置专门 capture context。Breakable backend 优先验证前者。

### 4. 首期只使用精确 token bucket

dLLM pure prefill 长度已对齐到 `lcm(page_size, block_size)`。当前常用
32/128/512/1024 都有精确 capture bucket。

首期增加条件：

```python
num_tokens in self.capture_num_tokens
```

原因是 LLaDA2 chunk 内使用双向 attention。普通 runner 的向上 padding 已服务于
其他模型，但必须专门确认：

- attention metadata 的真实 query boundary 排除 padded tail；
- padded token 的 K/V 不被真实 query 看到；
- padded `out_cache_loc` 不覆盖有效 cache；
- graph 输出裁剪不会掩盖已经发生的 KV 污染。

完成专门 parity test 后，可以取消精确 bucket 限制，复用现有最近 bucket padding。

### 5. 保留 decode graph，不创建新的 dLLM graph 类型

最终 dispatch 应为：

```text
dLLM pure prefill
  -> ForwardMode.EXTEND
  -> PrefillCudaGraphRunner (eligible) / EagerRunner (fallback)

dLLM decode
  -> ForwardMode.DLLM_EXTEND
  -> DecodeCudaGraphRunner (eligible) / EagerRunner (fallback)
```

不建议为每个 `prefill_block_size` 扩展 Decode CUDA Graph；普通 prefill runner 已经
提供 token bucket、动态 prefix metadata 和 eager fallback，复用它的维护成本更低。

## 建议实现顺序

### 阶段 1：最小验证原型

目标：证明当前 H200/LLaDA2/FlashInfer/breakable 下 graph replay 正确且有收益。

1. 临时将 dLLM gate 改成只拒绝非 `EXTEND` dLLM batch，或直接在本地删除 gate；
2. 限定精确 capture bucket；
3. 跑单请求 PBS=32/128/512；
4. 确认服务日志中纯 prefill 为 `cuda graph: True`；
5. 对比 graph disabled 的输出和性能。

该原型用于验证，不应直接作为最终提交。

### 阶段 2：正式 capability 和 phase 表达

1. 增加并传播 `is_dllm_prefill`；
2. 实现 CUDA/FlashInfer/breakable capability gate；
3. static batch 保留 dLLM 字段；
4. 增加 unit tests；
5. 增加 LLaDA2 CUDA integration test；
6. 更新 scheduling design 中“dLLM prefill 禁止 graph”的旧描述。

### 阶段 3：扩大支持范围

按单独验证结果逐项放开：

- 非精确 bucket padding；
- multi-request prefill；
- `tc_piecewise`；
- HIP/其他 attention backend；
- 其他 dLLM algorithm/model。

每项都应由 capability gate 控制，不能因为一个环境通过就默认全平台启用。

## 测试计划

### Unit tests

建议为 runner eligibility 新建聚焦的测试文件，并保留 scheduler phase 测试，覆盖：

1. 普通 batch 继续使用原 can-run 逻辑；
2. CUDA/FlashInfer/breakable 的 dLLM pure prefill + exact bucket 返回 `True`；
3. `DLLM_EXTEND` 返回 `False`（它应走 decode runner）；
4. `is_dllm_prefill=False` 的 dLLM batch 返回 `False`；
5. 非精确 bucket 首期返回 `False`；
6. input embeds、replace embeds、logprob 等现有条件仍返回 `False`；
7. unsupported backend/device 返回 `False`；
8. `static_forward_batch` 保留 dLLM phase/config。

避免只 mock `dllm_config` 后断言结果；测试应同时构造 mode、phase、backend
capability 和 token shape，确保准入规则不会误伤 decode。

### CUDA integration parity

对每个 case 分别运行 prefill graph enabled/disabled，并使用固定 seed：

| 维度 | 建议值 |
| --- | --- |
| PBS | 32, 128, 512, 1024 |
| prompt | 对齐、非对齐、跨多个 chunk |
| prefix | 0、非零 radix-cache hit |
| batch | 1、2、4 请求 |
| output | 至少跨两个 decode block |
| bucket | exact；阶段 3 再加 padded |

比较项：

- 最终 token IDs；
- 每轮 `extend_range`；
- prefill→decode phase transition；
- 首个 decode block 的 full logits 或选定位置 logits；
- 有条件时比较每层有效 KV slice；至少比较最后层或下一 block logits，以发现 KV
  污染；
- `can_run_cuda_graph` 和日志 graph flag。

只比较最终文本不足以证明 KV 正确，因为阈值解码可能掩盖小的中间差异。

### 性能验收

复用：

```text
num_prompts=32
output_len=64
max_concurrency=1
input_len=128/256/512/1024/1536/2048
PBS=32/128/512/1024
固定 seed，逐组 flush cache
```

至少报告：

- TTFT；
- request/input/output throughput；
- pure prefill forward 次数；
- graph hit 次数；
- server startup capture time；
- graph memory usage。

验收底线：

1. correctness parity 通过；
2. PBS=32 不低于原版 `DLLM_EXTEND` graph 基线，或清楚解释剩余差距；
3. PBS=128/512 相比当前 eager 路径无显著回退；
4. decode graph hit 和 decode correctness 不受影响；
5. 不支持的环境稳定 fallback eager。

## 可能遇到的问题

### Breakable capture dummy 与 runtime phase guard

新增 `is_dllm_prefill` 后，若该 Python bool 被 capture backend/Dynamo 当作 guard，通用
capture dummy 的 `False` 与 runtime 的 `True` 可能导致 graph miss。解决方向：

- 不让模型主体读取该字段，只在 runner 准入和 graph 外控制；或
- 为 dLLM prefill 单独设置一致的 capture context；或
- 在 `load_batch()` 前消费 phase 信息，不让它进入 captured closure。

首选第一种，避免为相同模型 shape 重复捕获图。

### full logits 语义

`DLLM_EXTEND` 的 logits processor 返回所有 block token 的 full logits；普通
`EXTEND` 通常只返回请求级 logits。纯 prefill 不使用 logits，因此允许普通
`EXTEND` 输出。必须保证带 mask 的 decode 永远不会误入该路径，否则
`LowConfidence` 会缺少 full logits。

### 双向 attention 与 padding

LLaDA2 的当前 chunk 是 encoder-only/non-causal attention。向上 bucket padding 对这种
语义比 causal prefill 风险更高，必须检查 query/KV indptr 和 padded KV write。精确
bucket 首期限制用于隔离该风险，不代表 padding 永远不可支持。

### 多请求 metadata

Breakable runner 声称支持多请求：捕获的 transformer stack 以 bs=1 replay，外层
model/logits processor 使用 live multi-request metadata。首期应先验证 bs=1，再测试
多个长度相同和不同的 prefill 请求，尤其检查 `extend_start_loc` 和每请求非 causal
attention 边界。

### MoE 数值差异

LLaDA2-mini 是 MoE 模型。graph/eager 可能因 kernel/autotune 路径产生浮点差异。
测试应区分可接受的数值误差与 token/KV 语义错误；最终生成 token parity 应在固定
seed 和确定性允许范围内评估，不能仅以逐元素 bitwise 相等作为唯一标准。

## 可选的短期兼容方案

若正式 Prefill CUDA Graph 开发周期较长，可以先对
`prefill_block_size == block_size` 恢复旧版 `DLLM_EXTEND` decode graph 路径，以
修复 PBS=32 回退。但它不能支持大 prefill chunk，也不能替代本文方案。

短期方案与正式方案应避免同时改变同一 batch 的 mode；最终应优先使用明确的
dispatch：大/可变 prefill 走 Prefill Graph，固定 decode 走 Decode Graph。

## 完成定义

只有满足以下条件才可将状态改为“已实现”：

- 代码中不再全量拒绝 dLLM prefill，而是使用明确 phase/capability gate；
- CUDA/FlashInfer/breakable exact-bucket replay correctness tests 通过；
- LLaDA2 端到端 graph/eager parity 通过；
- PBS benchmark 证明性能回退已消除或显著改善；
- decode 仍保持固定 block 和原 CUDA Graph 路径；
- unsupported backend/device 有测试覆盖的 eager fallback；
- scheduling design、配置说明和测试文档同步更新。

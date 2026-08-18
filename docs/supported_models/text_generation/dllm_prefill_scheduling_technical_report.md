# dLLM 多块 Prefill 调度与块级注意力技术报告

## 摘要

dLLM（diffusion large language model）的 decode 以固定大小的 token block 执行。
在原有实现中，prefill 沿用了同一 `block_size`，使长 prompt 每个调度轮次也只能前进
一个 block。该限制增加了调度、batch 构造和 kernel launch 的次数，并使长上下文场景难以
充分利用 GPU。

本文提出并实现一种多块 prefill 执行模式：将 pure prefill 的最大 chunk 长度
`prefill_block_size` 与 decode 的 `block_size` 解耦。decode 仍严格保持每请求、每轮一个
固定 block 的 dLLM 语义；pure prefill 则可以在单轮中处理多个、按 block 对齐的真实 prompt
token。为保证这种解耦的正确性，系统引入 phase-aware 调度、在 prefix-cache 匹配完成后再做
phase 判定、block-wise prefill attention，以及针对 Breakable CUDA Graph 的精确 bucket
复用规则。

在 2x NVIDIA H200、`inclusionAI/LLaDA2.0-mini`、TP=1 的评测中，长上下文收益随 prompt
长度增长。在 16K 输入下，`prefill_block_size=2048` 相比 main 基线将输入吞吐提升 491%，
平均 TTFT 从 2996 ms 降至 351 ms。GSM8K 的 200 样本评测未显示质量回归；多块 eager 和
Breakable 配置的得分分别为 0.915 和 0.910，均高于基线的 0.895。

## 1. 引言

### 1.1 背景

dLLM（Diffusion LLM）是一种新兴的文本生成范式。与每步只生成一个 token 的自回归模型不同，
dLLM 在 block 粒度并行生成多个 token，并通过多步去噪逐步确定当前 block 的内容。因此，
同一 block 内的 token 必须具有双向可见性，而后续 block 必须完全不可见。这正是 Block
Extend Mask 的语义来源。

dLLM 的生成过程以 block 为基本单位：在 decode 阶段，一个请求需要处理一个由真实 token
和 mask 组成的完整 block。该固定形状对位置计算、KV 写入、专用 CUDA Graph 以及 dLLM
注意力语义都至关重要。

长上下文服务的主要工作量通常发生在 prefill。若 prefill 与 decode 一样每轮只能处理一个
`block_size`，则一个 16K prompt 在 `block_size=32` 时至少需要数百次调度推进。即使单次
计算正确，这一过程仍会产生可观的调度和运行时开销，并限制单轮可提交给 GPU 的工作量。

### 1.2 Block Diffusion 与 Chunked Prefill 的关系

Block Diffusion 的执行流程与 SGLang 已有的 Chunked Prefill 高度相似：二者均将完整序列
划分为若干 chunk 并逐段处理。对当前 chunk 的 query 与历史 KV，二者都使用 full attention；
根本差异只在当前 chunk 内部的 attention：Chunked Prefill 采用 causal mask，Block Diffusion
则要求 bidirectional full attention。

| 计算阶段 | Chunked Prefill | Block Diffusion |
| --- | --- | --- |
| 当前上下文查询（`Q_curr × KV_prev`） | Full Attention | Full Attention（相同） |
| 当前块内部查询（`Q_curr × KV_curr`） | Causal Attention Mask | Full Attention（双向） |

这种相似性使复用 Chunked Prefill 看起来很自然。初始方案是以 dLLM block 切分请求，并对当前
chunk 设置 `causal=False`，以近似 Block Extend Mask。但 `causal=False` 只在 chunk 恰为一个
完整 dLLM block 时才正确：若单个 chunk 跨越多个 dLLM block，普通双向 attention 会令早期
block 看到未来 block，违反 dLLM 语义。因此，复用路径本身无法支持用于减少调度步数的更大
prefill chunk。

### 1.3 问题定义

简单地增大 prefill chunk 不能直接复用普通的 non-causal attention，也不能将动态长度的
prefill 送入固定形状 dLLM decode 图。系统必须同时解决以下问题：

- **执行语义：** decode 必须始终以固定 `block_size` 执行；pure prefill 只能提交真实
  prompt，不能跨入末尾的 mask 区间。
- **注意力可见性：** 一个多 block extend 中，同 block token 双向可见，历史 block 可见，
  未来 block 不可见。
- **调度一致性：** prefix-cache 匹配可能改变请求的阶段；在请求真正准入前推进 offset 会
  造成 position 漂移。
- **图执行正确性：** 普通 prefill CUDA Graph 的向上 padding 会改变 dLLM 的注意力和 KV
  写入语义，因此不能无条件复用。

### 1.4 贡献

本文工作的主要贡献如下：

- 提供 `prefill_block_size`，使多 block pure prefill 可配置且向后兼容。
- 建立 prefill/decode phase 分离的调度模型，分别使用动态 extend 与固定 dLLM block。
- 在 FlashInfer 上实现 request-wise block mask、ragged prefix cascade 及 paged correctness
  fallback。
- 修复 prefix-cache 与 incoming 请求准备顺序导致的两类 phase 误判。
- 通过单元测试、attention 数值测试、模型质量测试及 H200 性能矩阵验证正确性和收益。

## 2. 设计目标与不变量

本设计的首要目标是缩短长 prompt 的调度路径，而不是改变 dLLM decode 的计算定义。系统
始终保持以下不变量：

- decode 对每个请求每轮恰好处理一个完整的 `block_size` block；
- pure prefill 仅包含真实 prompt token，且结束位置按 `block_size` 对齐；
- 非对齐的 prompt 尾部与所需 mask 一起由随后的 decode block 处理；
- 单个调度轮次只运行 prefill 或 decode，不能混合两种 phase；
- pure prefill 使用普通动态长度的 `ForwardMode.EXTEND`，decode 使用固定形状的
  `ForwardMode.DLLM_EXTEND`；
- 未被 `PrefillAdder` 准入的请求不更新 `dllm_block_offset`；
- phase 必须在 prefix-cache 匹配完成后确定。

这组不变量将优化的影响范围限制在 prefill 吞吐路径，同时保护 dLLM 的 decode、KV cache
和 CUDA Graph 语义。

## 3. 多块 Prefill 调度设计

### 3.1 配置与兼容性

`DllmConfig` 从 `--dllm-algorithm-config` 指定的 YAML 中读取可选
`prefill_block_size`：

```yaml
block_size: 32
prefill_block_size: 1024
threshold: 0.95
```

也可通过 `--dllm-prefill-block-size` 覆盖 YAML 值。未配置时，
`prefill_block_size` 默认等于 `block_size`，因而保留旧的单 block 行为。配置值必须满足：

```text
prefill_block_size >= block_size
prefill_block_size % block_size == 0
```

当 `prefill_block_size > block_size` 时，启动参数必须显式选择
`--attention-backend flashinfer`。原因是当前多块 prefill 的 block-wise custom mask 仅在
FlashInfer backend 中实现。对不支持的 backend 在启动时失败，比静默套用错误的注意力语义
更安全；单 block 配置不受此限制。

### 3.2 调度流程与 phase 判定

每轮调度首先接收 incoming 请求，并调用 `DllmManager.init_next_round(tree_cache)`。对于
incoming 和 staging 请求，调度器先完成 prefix-cache match，再根据最终的
`prefix_indices` 调用 `determine_dllm_phase()`。随后才构造 phase-aware 的
`PrefillAdder`、选择本轮预算及 forward mode。

```text
incoming / staging requests
            |
            v
      prefix-cache match
            |
            v
  final phase determination
            |
     +------+------+
     |             |
     v             v
pure prefill     decode
EXTEND           DLLM_EXTEND
dynamic chunk    one fixed block
```

incoming 请求在真正进入 `adder.can_run_list` 前保持
`INCOMING_PREFILL` 或 `INCOMING_DECODE` 生命周期；只有实际被准入后，
`process_dllm_incoming_reqs()` 才将其转换为相应 staging phase。同一 phase 内，调度器优先
选择 staging 请求，以继续已有工作并降低生命周期切换开销。

这一顺序修复了两类错误。第一，旧逻辑可能在 prefix-cache match 更新 `prefix_indices` 前
判定 phase，将“非对齐 prompt 尾部 + mask”错误视为 pure prefill。第二，旧逻辑可能在 incoming
请求尚未完全准备好时就选择 `is_prefill`、adder 和 forward mode；即使后续请求变为 decode，
当前轮已被不可逆地配置为 prefill。新流程将所有这些决策建立在最终 phase 之上。

### 3.3 预算、chunk 与对齐

prefill 与 decode 采用不同的总 token 预算：

```text
prefill budget = max_running_requests * prefill_block_size
decode budget  = max_running_requests * block_size
```

`PrefillAdder` 还同时受 phase cap、`max_prefill_tokens`、可用 KV 容量和 page/block 边界的
约束。对一个请求，令 `context_end` 为真实上下文末端，pure prefill 的上界为：

```text
pure_prefill_end = floor(context_end / block_size) * block_size
prefill_end = min(prefix_len + available, pure_prefill_end)
```

例如 `block_size=32`、prompt 长度为 300 时，pure prefill 最多提交前 288 token。剩余 12 个
真实 token 与 20 个 mask 在下一次固定 32-token decode 中共同处理。若对齐后的 pure prefill
长度为零，这是正常的 phase 边界而不是 token 预算耗尽；只有资源不足等确实无法提交 token
的情形才返回 `NO_TOKEN`。

请求仅在 chunk 实际提交后，按 `extend_range.length` 更新 `dllm_block_offset`。这一规则避免
未准入请求被重复初始化时错误前移，从而防止 position 漂移、重复跳过 token 或调度活锁。

## 4. 块级 Prefill Attention

### 4.1 可见性定义

设 dLLM block 大小为 `B`，请求内 query、key 的绝对位置分别为 `q`、`k`。多块 prefill
必须遵循：

```text
visible(q, k) = floor(k / B) <= floor(q / B)
```

这意味着同一个 block 内双向可见；当前 block 可以访问所有更早的 block；任何未来 block
都必须被隔离。以 `B=2`、extend 长度为 6 为例，extend 内 mask 为：

```text
1 1 0 0 0 0
1 1 0 0 0 0
1 1 1 1 0 0
1 1 1 1 0 0
1 1 1 1 1 1
1 1 1 1 1 1
```

因此，普通全双向 attention 会泄漏未来 block 信息，普通 causal attention 又会错误屏蔽
同 block 的后续 token；二者都不能直接用于多块 dLLM prefill。

### 4.2 Custom Mask 与快路径

`build_dllm_prefill_blockwise_mask()` 根据请求内绝对位置构造 bool mask。每个请求的二维
mask 按 query-major 顺序 flatten，再按 batch 中请求顺序拼接，以匹配 FlashInfer custom mask
布局。异构 batch 中的 `prefix_len` 和 `extend_len` 相互独立，不会共享 mask 区域。

当 batch 中每个 extend 均落在单个 dLLM block 内时，普通 non-causal attention 已满足同一
block 的可见性需求，函数返回 `None` 并使用无 mask 快路径。只要任一请求跨越 block 边界，
整个 batch 使用 custom-mask 路径，保证异构 batch 的语义一致。

### 4.3 Ragged Prefill 与 Prefix Cascade

默认高性能路径将当前 extend 与已缓存 prefix 分开处理。若历史 prefix 长度为 `P`，当前
extend 长度为 `E`：

1. ragged wrapper 对当前 extend 的 K/V 执行带 `E x E` block mask 的 attention；
2. paged wrapper 对已缓存 prefix 的 K/V 执行无 mask 的 `E x P` attention；
3. 使用 FlashInfer `merge_state` 合并两支的 output 与 log-sum-exp（LSE）。

prefix token 的绝对位置总早于当前 query，因此 prefix 分支整体可见；未来 extend block 则由
第一支的 block mask 隔离。这种 prefix cascade 避免构造 `E x (P + E)` 的完整 mask，保留了
radix/paged KV cache，并降低长 prefix 下的 mask 构造和显存开销。

masked 与 unmasked ragged prefill 使用独立的持久 FlashInfer wrapper：无 mask wrapper 使用
`backend="auto"`，可在支持的平台选择 FA3；custom-mask wrapper 固定为 FA2，因为当前 FA3
路径不支持该 mask。planning 时选中的 wrapper 会写入 `PrefillMetadata` 并在各 attention
layer 中持续使用，避免 FlashInfer 0.6.12 首次计划后复用不兼容 backend 的顺序依赖。

### 4.4 Paged Fallback 与限制

当 ragged prefill 被禁用时，paged wrapper 直接使用覆盖 prefix 和 extend 的
`E x (P + E)` block-wise mask。这一路径与 ragged cascade 具有相同可见性，是正确性 fallback；
由于其 mask 开销为 `O(E(P + E))`，不作为多块 prefill 的默认性能路径。

当前 custom-mask 实现不与 sliding-window wrapper dispatch 或 multi-item scoring 组合。检测
到这些组合时系统显式报错，避免以错误的注意力语义静默降级。显式 custom mask 也是过渡方案：
当 SGLang 支持的 FlashInfer 版本提供原生 Block Extend Attention 后，可用其 operator 替换
该实现，而无需改变 scheduler 的 prefill/decode 分离接口。

### 4.5 原生 Block Extend Kernel 的必要性

本 PR 的 scheduler 改造使更大的 prefill chunk 成为正确的执行模式，但 attention 侧的
Cascade/custom-mask 路径仍是兼容性方案，而非最终形态。其局限性来自两方面。

第一，基于 Cascade Attention 的单 block 复用路径每一步通常需要当前 chunk attention、
prefix attention 和 `merge_state` 共 2--3 次 kernel launch；随着序列被切成更多 block，CPU
侧的调度与 planning 开销也线性累积。它只能在 `chunk_size == dllm_block_size` 时用
`causal=False` 正确表达当前块的双向可见性，无法直接放大 chunk 来减少步数。

第二，通用 `MaskMode::kCustom` 需要显式保留二维 mask。若为整个 query--KV 范围构造 mask，
显存复杂度为 `O(qo_len x kv_len)`；长序列下这一开销不可接受，例如单请求 `seq_len=32K`
时可达到约 1 GB。即使通过本文的 ragged prefix cascade 将默认 mask 缩小为当前 extend 的
`E x E`，或在 fallback 中使用 `E x (P+E)`，它仍需构造并传递通用 mask，且无法利用 Block
Extend Mask 中大量“未来 block 不可见”的规则化稀疏结构。

原生 Block Extend Attention kernel 应把上述可见性规则编码在 kernel 内：只对当前及历史
block 的 KV tile 进行加载和计算，并跳过未来 block tile。它既可消除大尺寸显式 mask 的
存储与传输，又能避免对不可见 KV 的无效计算。因而，本文的 runtime/scheduler 分离与原生
kernel 支持是互补关系：前者解决正确的请求生命周期、chunk 和 phase 管理，后者解决长序列
attention 的最终内核效率。

## 5. Forward Mode 与 CUDA Graph

Pure prefill 使用动态长度的 `ForwardMode.EXTEND`，并以 `dllm_config` 和
`is_dllm_prefill` 区别于普通模型 prefill。其 position、extend metadata、custom mask 与 KV
写入都按真实 extend 长度计算。Decode 继续使用 `ForwardMode.DLLM_EXTEND`，每轮固定处理一个
`block_size`，并保留其专用 CUDA Graph 路径。

Pure dLLM prefill 可以复用普通 prefill 的 Breakable CUDA Graph，但必须同时满足：

- scheduler 已标记 `is_dllm_prefill=True` 且 forward mode 为 `EXTEND`；
- 运行在 CUDA 和 FlashInfer attention backend 上；
- prefill graph backend 为 Breakable；
- token 数精确命中已 capture 的 bucket；
- 未使用 input embedding 或其他既有 graph 不支持的输入模式。

不满足任一条件时系统安全回退 eager。这里要求精确 bucket 命中而不能向上 padding：padding
会引入额外 token，进而改变 block-wise 双向 attention 和 KV 写入语义。Graph replay 同样传递
`dllm_config` 与 `is_dllm_prefill`，以在 replay 中重建完全一致的 metadata 和 block mask。

## 6. 正确性验证

### 6.1 调度与配置测试

`test/registered/unit/managers/test_prefill_adder.py` 覆盖：

- phase-aware prefill/decode 预算；
- 对齐和非对齐 prompt 边界；
- decode 固定 block 语义；
- incoming 生命周期、实际准入及 offset 更新；
- cache match 完成后的 phase 判定；
- FlashInfer backend gate；
- Breakable CUDA Graph capability gate。

该测试套件当前结果为 24 项通过。

### 6.2 Attention 数值与隔离性测试

`test/registered/unit/layers/test_flashinfer_dllm_prefill_mask.py` 覆盖 mask 公式、异构 batch、
单 block 快路径、非对齐 prefix 以及 masked/unmasked wrapper 隔离。CUDA + FlashInfer 验证还
包括：

- production-shaped bf16 GQA ragged attention 对 dense reference 的一致性；
- prefix cascade output 与 LSE 对 dense reference 的一致性；
- 修改未来 block K/V 不影响早期 block 输出的隔离性；
- paged fallback 与 dense reference 的一致性。

在 `P=64`、`E=128`、`B=32` 的 prefix-cascade 验证中，最大 output 误差为 `5.3e-4`，最大
LSE 误差为 `7e-5`；修改未来 block 后早期输出差异为零；paged fallback 的最大误差为
`2.7e-4`。无 CUDA 环境下该文件为 7 passed、4 skipped；GPU 验收要求全部 11 项实际执行并
通过。

### 6.3 模型级一致性与质量

多块 eager 长 prompt 回归未出现 `operation not supported` 错误。对 prompt 长度 31--2060，
multi-block eager 与 Breakable 的输出 token ID 完全一致。

GSM8K 使用 200 个样本、`temperature=0`、TP=1 且关闭 radix cache，结果如下：

| 配置 | GSM8K 得分 |
| --- | ---: |
| Main baseline | 0.895 |
| Current single-block eager | 0.895 |
| Multi-block eager | 0.915 |
| Multi-block Breakable | 0.910 |

长 prompt 下，single-block 与 multi-block 不要求逐 token 完全一致：不同 chunk 边界会改变
kernel shape、KV 提交边界和浮点累积顺序。即使强制两条路径使用 FA2，也可观察到这一数值差异。
端到端 GSM8K 结果表明该差异没有造成质量回归。

## 7. 性能评测

### 7.1 实验设置

性能测试运行在 2x NVIDIA H200 143 GB 上，模型为 `inclusionAI/LLaDA2.0-mini`，PyTorch
版本为 2.11.0+cu128，FlashInfer 为 0.6.12，TP=1，dLLM `block_size=32`。Prefill serving
矩阵使用 16 个 prompt、`output_len=64`、`max_concurrency=1` 和固定随机种子
`seed=2000+input_len`。比较 main（32）与当前实现的
`prefill_block_size in {32, 128, 512, 1024, 2048, 4096}`，输入长度覆盖 128--16384。

### 7.2 GSM8K 端到端延迟

| 配置 | 延迟（s） | 输出吞吐（tok/s） |
| --- | ---: | ---: |
| Main baseline | 65.8 | 381.0 |
| Current single-block eager | 121.6 | 205.2 |
| Multi-block eager | 58.6 | 435.0 |
| Multi-block Breakable | 47.1 | 540.7 |

相对当前 single-block eager，多块 eager 将输出吞吐从 205.2 提升至 435.0 tok/s（2.12x）；
多块 Breakable 提升至 540.7 tok/s（2.63x），并将总延迟从 121.6 s 降至 47.1 s。Breakable
图回放相对多块 eager 额外提供约 1.24x 的输出吞吐提升。

### 7.3 长上下文 Prefill Scaling

下表给出长上下文下、相对 main 的输入吞吐变化：

| 输入长度 | pbs1024 | pbs2048 | pbs4096 |
| ---: | ---: | ---: | ---: |
| 4K | +160% | +168% | +158% |
| 8K | +321% | +356% | +365% |
| 16K | +434% | +491% | +470% |

16K 时，main 的输入吞吐为 5193 tok/s；pbs1024、pbs2048、pbs4096 分别达到 27713、30709、
29577 tok/s。平均 TTFT 则由 main 的 2996 ms 降至 pbs1024 的 410 ms、pbs2048 的 351 ms 和
pbs4096 的 365 ms。

收益随输入长度增长，符合设计预期：prompt 越长，减少 scheduler round 的收益越大。在本次
H200、模型和并发设置下，`prefill_block_size=2048` 是 8K--16K 的最佳点。更大的 4096 不
总是更快，表明 chunk 大小需要在 attention 成本、mask/metadata 成本、图 bucket 与调度开销
之间权衡。`pbs32` 略慢于 main 也符合预期：在 chunk 大小未增大时，phase-aware 新路径仍有
少量额外管理开销。已完成 sweep 的 CUDA Graph miss 为零，`mem-fraction-static=0.75--0.85`
下未发生 OOM。

## 8. 局限性与后续工作

当前多块 prefill 依赖 FlashInfer custom mask，且不支持 sliding-window dispatch 和 multi-item
scoring。Ragged 路径不可用时的 paged fallback 虽然正确，但其 `O(E(P+E))` mask 成本不适合
作为长期默认性能方案。即使 ragged prefix cascade 降低了完整 mask 的峰值需求，它仍带来
额外的 mask 构造、planning 和多 kernel launch 开销，也不能跳过未来 block 的无效 KV tile
计算。

下一步应在 SGLang 支持的 FlashInfer 版本具备原生 Block Extend Attention 后，迁移 attention
实现至该 operator，并保留本文定义的 scheduler phase 边界和配置接口。进一步工作还包括：

- 在不同模型、GPU、并发度与真实服务 workload 下复现 chunk-size sweet spot；
- 扩展与 sliding window、scoring 等特性的兼容性；
- 基于 prompt 长度、KV 容量和图 bucket 自动选择 `prefill_block_size`；
- 将性能验收持续集成到 GPU 回归流程中。

## 9. 结论

本文将 dLLM 的 pure prefill 从固定 decode block 中解耦，使其能够在保持 dLLM decode 语义
不变的前提下按多个对齐 block 推进。phase-aware 调度、cache-match 后的最终 phase 判定、严格
的请求 offset 生命周期，以及 block-wise attention 共同保证了该优化的正确性；Breakable CUDA
Graph 则进一步降低了满足精确 bucket 条件时的运行时开销。

实验显示，该设计的价值主要体现在长上下文：16K 输入的吞吐最高提升 491%，TTFT 显著下降，
同时 GSM8K 质量未出现回归。这表明多块 prefill 可作为 dLLM 长 prompt 服务的有效执行模式，
而其 scheduler/runtime 分离设计也为后续接入 FlashInfer 原生 Block Extend Attention 奠定了
稳定接口。

## 参考与复现材料

- [dLLM Prefill Scheduling Design Note](dllm_prefill_scheduling_design.md)
- [dLLM block-wise attention GPU 验证方案](../../developer_guide/dllm_blockwise_attention_gpu_validation.md)
- PR 描述中的测试命令和完整 H200 prefill 性能矩阵。

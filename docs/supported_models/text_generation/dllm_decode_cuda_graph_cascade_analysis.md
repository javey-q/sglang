# dLLM Decode CUDA Graph 中的 FlashInfer Cascade Attention 解析

本文说明 SGLang 在 dLLM decode（`ForwardMode.DLLM_EXTEND`）的 CUDA Graph 路径中，
如何通过 FlashInfer 的 ragged/paged cascade attention 以及 `merge_state`，实现
“当前 decode block 双向可见、历史 KV 全可见”的注意力语义。

本文讨论的是 FlashInfer backend 的 ragged cascade 分支；它不是 dLLM 的调度设计
本身，也不改变纯 prefill 使用普通 `EXTEND` 的约定。

## 1. 问题：一个 decode block 的注意力可见性

设每个 dLLM decode 请求本轮处理固定长度的 block `B`，当前请求在 KV cache 中已有
历史 prefix，长度为 `P`。当前 block 的 query 对应的目标可见集合是：

```text
历史 prefix：P 个 token，全部可见
当前 block：B 个 token，全部互相可见（非 causal）
```

因此，逻辑上每个请求需要计算：

```text
Q_current[B] × K_(prefix + current)[P + B]
```

这和通常 autoregressive decode 的“只见过去 token”不同。对于 dLLM 的
`ENCODER_ONLY` attention，当前 block 内也必须是双向可见。

## 2. 进入 CUDA Graph 路径时如何准备 metadata

Decode CUDA Graph runner 会以 `DLLM_EXTEND` 捕获和重放。FlashInfer backend 在
`init_forward_metadata_out_graph()` 中识别这个 forward mode，并设置：

```text
prefix_lens = seq_lens - block_size
```

其中 `seq_lens` 是本轮 block 纳入后的位置长度，故减去 `B` 后正好得到历史 KV 的
长度 `P`。对应实现位于：

- `python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py`
- `python/sglang/srt/layers/attention/flashinfer_backend.py`

在可使用 ragged cascade 的配置中，backend 将 `use_ragged` 设为 true，并为两个
wrapper 准备不同 metadata：

- ragged wrapper 的 QO/KV 长度都是 `B`，描述当前 block 的连续 Q/K/V；
- paged wrapper 的 KV 长度为 `P`，描述已存在 paged KV cache 中的历史 prefix。

这里没有把当前 block 拼入 paged KV 的读取集合。

> `use_ragged` 会受 backend 配置约束；例如 `use_paged`、确定性模式或某些特殊
> attention 功能会关闭此分支。本文的 cascade/merge 解析仅适用于 `use_ragged=True`。

## 3. 两段 attention 的计算与时序

在 `FlashInferAttnBackend.forward_extend()` 中，ragged cascade 分支执行两次
`forward_return_lse`：

```text
输入：Q_current, K_current, V_current

1. ragged attention
   Q = Q_current
   K/V = K/V_current
   causal = false（对 ENCODER_ONLY / LLaDA）

2. paged attention
   Q = Q_current
   K/V = paged KV cache 中的 prefix
   causal = false

3. merge_state(ragged_result, paged_result)

4. 将 K/V_current 写入 paged KV cache
```

示意图：

```text
                         Q_current (B)
                                │
             ┌──────────────────┴──────────────────┐
             ▼                                     ▼
  ragged：K/V_current (B)                 paged：K/V_prefix (P)
  block 内双向可见                         历史 prefix 全可见
  causal = false                           causal = false
             │                                     │
             └─────────── merge_state ─────────────┘
                                   │
                           attention output
                                   │
                     写入当前 block 的 K/V 到 cache
```

写回发生在合并之后是关键约束：paged attention 只能读取历史 prefix，不能又读取当前
block。于是两个分支的 KV 集合是互斥的，且它们的并集恰好是目标可见集合。

## 4. 为什么两个分支都使用 `causal=False`

当前 block 分支若使用 causal mask，则 block 中位置 `i` 不能看见位置 `j > i`；这会
破坏 dLLM 对同一 mask block 的双向预测语义。因此，LLaDA 的
`AttentionType.ENCODER_ONLY` 会令 ragged attention 使用 `causal=False`。

paged 分支的 K/V 全部是当前 query 之前已完成的历史 block，不需要再在该分区中施加
因果三角约束；同样使用 `causal=False`，让每个当前 query 都看到完整的 prefix。

这两个非因果设置并不表示未来 token 泄漏：当前 block 外的未来 token 既不在 ragged
K/V，也不在 paged KV cache 的读取范围内。

## 5. `merge_state` 的数学含义：精确恢复整体 softmax

单个 attention 分支除了输出 `o`，还返回每个 query/head 的：

```text
s = logsumexp(scores)
```

记当前 block 分支的状态为 `(o_cur, s_cur)`，prefix 分支为 `(o_pre, s_pre)`。
`flashinfer.cascade.merge_state`（SGLang 经 `_safe_merge_state` 调用）使用稳定的
log-sum-exp 计算：

```text
m     = max(s_cur, s_pre)
w_cur = exp(s_cur - m)
w_pre = exp(s_pre - m)

o = (w_cur * o_cur + w_pre * o_pre) / (w_cur + w_pre)
s = m + log(w_cur + w_pre)
```

设两个互斥 KV 分区的 softmax 分母分别为 `Z_cur`、`Z_pre`。则：

```text
o_cur = sum(exp(score_cur) * V_cur) / Z_cur
o_pre = sum(exp(score_pre) * V_pre) / Z_pre
Z_all = Z_cur + Z_pre
```

又因 `s = log(Z)`，上述权重正是将两个局部归一化输出按各自分母重新加权。因此最终
`o` 与对 `[K_prefix; K_current]`、`[V_prefix; V_current]` 一次性进行完整 attention
的结果严格相同；`merge_state` 不是近似合并。

对于 FlashInfer 原生 merge kernel 不支持的 head 配置，SGLang 会选用等价的 Triton
实现，避免 CUDA block 线程数超限。该选择封装在 `_safe_merge_state()` 中。

## 6. 相比显式 `B × (P+B)` attention mask 的优势

目标可见性可写成一个动态矩阵 mask，但无需物化它，因为可见 KV 恰好是两个完整、
互斥的矩形分区。

- 显式 mask 每请求需要 `O(B(P+B))` 的额外存储、生成和读取带宽；cascade 只需要
  长度、页表和 indptr 等紧凑 metadata。
- 当前 block 使用 FlashInfer ragged kernel，历史 prefix 使用 paged-KV kernel；二者
  各自匹配数据布局，无需将历史 KV 聚合为连续大 tensor。
- CUDA Graph 可以重放固定 block 大小、固定 batch bucket 的 kernel 序列；动态变化的
  是预分配 metadata buffer 中的长度与索引，而不是一张大型 mask tensor。
- LSE merge 恢复完整 softmax，既省去 mask，也不牺牲数值语义。

## 7. 适用条件与边界

这个分解成立的前提是：每个 query 的可见 KV 能精确表示为“完整历史 prefix”与“完整
当前 block”这两个分区的并集。下列情形需要重新评估：

- token 级不规则的稀疏 attention；
- 历史 prefix 只允许部分 token 可见的复杂窗口规则；
- 当前 block 内存在与位置相关、非完整矩形的可见性；
- 需要额外 custom mask 的跨模态或特殊 scoring 路径。

这些模式未必能用两段 attention 加 `merge_state` 表示，应改走支持目标 mask 语义的
attention 实现，或先证明新的可见性仍可分解为互斥的完整 KV 分区。

## 8. 相关代码导航

| 文件 | 关注点 |
| --- | --- |
| `python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py` | dLLM 的 CUDA Graph 捕获 forward mode 为 `DLLM_EXTEND`。 |
| `python/sglang/srt/layers/attention/flashinfer_backend.py` | `prefix_lens`、ragged/paged wrapper metadata、两段 `forward_return_lse` 与 `_safe_merge_state`。 |
| `python/sglang/srt/layers/attention/triton_ops/merge_state.py` | FlashInfer 原生 merge 不适用时的等价 Triton 合并实现。 |
| `docs/supported_models/text_generation/dllm_prefill_scheduling_design.md` | dLLM prefill/decode phase 和 forward mode 的整体设计。 |

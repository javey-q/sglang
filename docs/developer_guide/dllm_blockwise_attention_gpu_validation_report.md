# dLLM Block-wise Attention GPU 验证报告

- **日期**: 2026-07-16（初测） / **2026-07-17（补测，修复后）**
- **依据**: `docs/developer_guide/dllm_blockwise_attention_gpu_validation.md`
- **环境**: 2× NVIDIA H200 143GB
- **模型**: `/data/models/inclusionAI/LLaDA2.0-mini`
- **软件**: torch `2.11.0+cu128`，FlashInfer `0.6.12`（与 `pyproject.toml` 一致）
- **结论总览（补测后）**: **上次阻塞的 multi-block eager 崩溃已消除；已测单请求 eager↔Breakable 全长一致。强制两边 FA2 后，single↔multi 在 ≥128 仍分叉，只能排除 FA3/FA2 差异是充分原因，根因尚未定位。GSM8K 四臂质量门槛已通过（C/D PASS）；single↔multi tensor parity、paged E2E、batch 2/4 Breakable replay 和完整性能矩阵仍未闭环。**

---

## 1. 环境检查

| 项 | 结果 |
|---|---|
| CUDA available | 通过 |
| GPU | NVIDIA H200 |
| FlashInfer | 0.6.12 |
| 初始阻塞 | 缺系统 `ninja`，FlashInfer JIT 失败；`apt install ninja-build` 后恢复 |
| 设备分配 | `CUDA_VISIBLE_DEVICES=1` 跑单测/对照，避免占用 GPU0 上已有服务 |

---

## 2. GPU mask / kernel 单测

命令：

```bash
CUDA_VISIBLE_DEVICES=1 FLASHINFER_WORKSPACE_BASE=/tmp \
  $PYTHON test/registered/unit/layers/test_flashinfer_dllm_prefill_mask.py -v
CUDA_VISIBLE_DEVICES=1 \
  $PYTHON test/registered/unit/managers/test_prefill_adder.py -v
```

| 套件 | 结果 |
|---|---|
| `test_flashinfer_dllm_prefill_mask.py` | 当时版本 **7/7 通过**（含 GPU test，未 skip）；review 后正式 suite 已扩展到 11 项，新增 cascade/isolation/paged tests，待在最终 commit 上重跑 GPU |
| `test_prefill_adder.py` | **23/23 通过** |

对应门槛 **#1：通过**。

---

## 3. Attention 级正确性（文档 3.1 / 3.2 + paged fallback）

临时脚本：`tmp_dllm_attn_gpu_validation.py`（未合入）。

设置：`P=64, E=128, B=32`；FlashInfer LSE 为 **log2** 刻度（dense reference 需 `/ ln(2)`）。

| 用例 | 结果 | 细节 |
|---|---|---|
| Prefix cascade parity（ragged + paged prefix + `merge_state` vs dense） | **PASS** | `max_out_err=5.3e-4`, `max_lse_err=7e-5` |
| Future-block isolation | **PASS** | 改最后 block 的 K/V 后，早期输出不变（`early_max_delta=0`），末 block 变化 |
| Paged `E×(P+E)` fallback vs dense | **PASS** | `max_err=2.7e-4` |

补充：按 LLaDA2 形状（bf16，GQA 16/4，`head_dim=128`，`E∈[32..1024]`）单独调用 FlashInfer ragged+custom_mask，**均 PASS**。

对应门槛 **#2、#3：通过**（attention 参考层）。

---

## 4. 模型级对照（初测，修复前）

### 4.1 已复用 / 启动的服务

| 角色 | 端口 | 配置 | 状态 |
|---|---|---|---|
| Multi-block Breakable | 30000 | `dllm_config.yaml`（`prefill_block_size=1024`），prefill graph=`breakable`（启动时已 capture 含 1024 bucket） | 稳定运行 |
| Single-block eager | 30001 | `tmp_baseline.yaml`（`prefill_block_size=32`），prefill graph=`disabled` | 曾用于对照后关闭 |
| Multi-block eager | 30001 | 同上 multi 配置，prefill graph=`disabled` | **长 prompt 崩溃** |

### 4.2 Single-block eager vs Multi-block Breakable

`temperature=0`，`max_new_tokens=64`（重复 stem prompt，存在 radix prefix 命中，结果需谨慎解读）。

| prompt≈ | `output_ids` 一致 |
|---|---|
| 31 / 32 / 33 | **是** |
| 128 / 512 / 1024 / 1056 | **否**（common prefix 0–3） |

Breakable 侧稳定性：同请求 10 次 `output_ids` 完全一致；512/1024 bucket 交替一致。

说明：短边界长度一致，长请求分叉可能来自 `prefill_block_size` 调度差异、Breakable vs eager 数值路径、以及重复 prompt 的 radix cache，**不能单独证明 attention 错误**。

### 4.3 Multi-block eager vs Multi-block Breakable（关键）

短请求（≈30–32 tokens）`output_ids` **完全一致**。

对 ≈122-token 唯一 prompt，**multi-block eager 调度器崩溃**：

```text
BatchPrefillWithRaggedKVCacheSM90Run failed with error: operation not supported
```

栈在 `flashinfer_backend.forward_extend` → ragged `forward`（eager extend 路径）。同一 FlashInfer 版本下，独立 kernel 复现（含 GQA/bf16/E=122）却通过，说明问题更可能在 **服务端 wrapper plan/run 状态、workspace、或 dLLM extend 元数据组合**，而非单纯 mask 公式。

同一组长唯一 prompt 在 **Breakable 服务上全部成功**（见下节）。

对应门槛 **#4（multi eager 与 single baseline / eager↔graph 对齐）：未通过 / 未完成**。

---

## 5. Breakable CUDA Graph（初测）

启动日志（已有服务）：

```text
Capture target prefill CUDA graph begin. backend=breakable, ... 1024 ...
Capture target prefill CUDA graph end. elapsed=7.69 s
```

本轮对 `:30000` 发送唯一长 prompt：

| 检查 | 结果 |
|---|---|
| 长度覆盖 31/32/33/128/512/1024/1056/2060 | 全部成功返回 |
| 出现 `#new-token: 1024` / `960` / `992` / `480` 等大 extend | **是** |
| `cuda graph: True` 为主 | 50 True / 3 False（非对齐 bucket 如 992/928 走 eager） |
| 同请求 8 次稳定性 | **通过** |
| 512↔1024 交替 | **通过**，无 mask/shape 报错 |

对应门槛 **#5：Breakable 路径通过**；**eager 对照因崩溃未完成**。

---

## 6. Paged fallback 服务 E2E

- Attention 级 paged custom-mask：**通过**
- `SGLANG_FLASHINFER_USE_PAGED=1` 完整 launch + 请求矩阵：**未跑**（时间与 eager 崩溃排查优先）

门槛 **#6：部分通过（kernel），E2E 未验证**。

---

## 7. 端到端质量（GSM8K）

已按验证方案 §7 完成 main/current-single/multi-eager/multi-Breakable 四臂 200 题
GSM8K 对照。multi eager score 为 0.915，multi Breakable 为 0.910，均满足绝对门槛
及相对 main/single 回退门槛。完整配置、hash 和结果见 §10.6。

对应门槛 **#7：通过**。

---

## 8. 性能 / 显存矩阵

文档要求的 `prefill_block_size × prompt × batch × mode` 全矩阵（warmup 5 / measure 30）**未跑**。

观测到的片段：

- Breakable 下 1024-token extend 可见，且 `cuda graph: True`
- GPU0 服务常驻显存约 **110 GB**（`mem-fraction-static=0.75`）
- 未做系统 TTFT / 峰值显存扫描；未见本次测试中的 OOM

门槛 **#8：未充分验证**。

---

## 9. 合入门槛对照

| # | 条件 | 状态 |
|---|---|---|
| 1 | GPU kernel 测试实际执行且不跳过 | **通过** |
| 2 | Prefix cascade 与 dense 对齐 | **通过** |
| 3 | Future-block isolation | **通过** |
| 4 | Multi-block eager 与 single-block baseline 对齐 | **部分通过**：eager 不再崩溃；**eager↔Breakable 完全对齐**；**single↔multi 长请求仍分叉** |
| 5 | Breakable replay 无 metadata 污染 | **部分通过**：单请求重复及 512/1024 bucket 交替通过；batch 2/4 与 20 次完整矩阵未跑 |
| 6 | Paged fallback 正确无 mask-size 错误 | **部分通过**（attention 级）；服务 E2E 未跑 |
| 7 | GSM8K 不回退 | **通过**：四臂 200 题 C/D PASS |
| 8 | 大 prompt 无不可接受显存增长 / OOM | **未系统测**；补测长请求未 OOM |

**总体：上次关键崩溃已关闭；已测单请求 eager↔graph parity 与 GSM8K 质量门槛通过。single↔multi tensor parity、Breakable batch/replay 完整矩阵、paged E2E 和性能矩阵仍待闭环。**

---

## 10. 2026-07-17 补测（修复后）

修复点：`flashinfer_backend.py` 为 custom-mask 使用独立 `backend="fa2"` ragged wrapper，避免 SM90 上 unmasked 首次 plan 选中 FA3 后污染后续 masked plan。

### 10.1 Multi-eager 崩溃回归

| 检查 | 结果 |
|---|---|
| 唯一 prompt ≈128/512/1024/1056/2060 | **全部成功**，服务保持 UP |
| 日志 `#new-token` | 出现 `480/928/960/992/1024` 等大 extend |
| `operation not supported` / SIGQUIT | **未再出现** |
| 同 ≈128 prompt 连续 10 次 | `output_ids` 完全一致 |

### 10.2 Multi-eager ↔ Multi-Breakable

`temperature=0`，`max_new_tokens=64`，唯一 prompt：

| prompt≈ | `ids_equal` |
|---|---|
| 31 / 32 / 33 / 128 / 512 / 1024 / 1056 / 2060 | **全部 true** |

另：Breakable 侧稳定性 10 次一致；512/1024 bucket 交替一致；图命中为主（补测段约 67 True / 8 False）。

产物：`/tmp/dllm_eager_vs_graph_parity_rerun.json`

### 10.3 Single-eager ↔ Multi-eager

两边均为 prefill graph disabled；`block_size=32`，仅 `prefill_block_size` 为 32 vs 1024。

| prompt≈ | `ids_equal` |
|---|---|
| 31 / 32 / 33 | **true** |
| 128 / 512 / 1024 / 1056 / 2060 | **false**（有部分 common prefix） |

说明：对照同时改变了 chunk 边界与 FA 路径（single 常走 unmasked auto/FA3，multi 走 masked FA2），因此不能单独归因于 block-wise attention。见下节最小归因矩阵。

产物：`/tmp/dllm_single_vs_multi_parity_rerun.json`

### 10.4 最小归因矩阵（single-FA2 vs multi-FA2）

方法：

- `SGLANG_DLLM_FORCE_RAGGED_FA2=1`：unmasked ragged 也强制 FA2，消除 FA3/FA2 混用
- `--disable-radix-cache` + 唯一 prompt
- `SGLANG_DLLM_ATTR_LOG`：记录 LowConfidence 每次 transfer 的 positions/tokens/confidence
- 客户端比较 **text / tokenized text**（当前 HTTP meta 不一定带 `output_ids`）

结果：

| prompt≈ | text 一致 | 首次 token 分叉 | 首次 hard transfer 分叉 |
|---|---|---|---|
| 33 / 64 | **是** | — | 无 |
| 128 | 否 | index=9 | step=1：pos 27→token16 vs pos 28→token22 |
| 256 | 否 | index=4 | step=2：不同 position/token |
| 512 | 否 | index=45 | step=2：不同 position/token |

关键观察：

1. **强制两边都走 FA2 后，≥128 仍分叉** → FA3/FA2 数值路径不是充分解释。
2. 分叉发生在 LowConfidence 很早的 transfer（step 1–2），且均为 `used_topk_fallback=true`（无位置超过 threshold，靠 top1 提交）。
3. 同一 step 下 `confidence_masked` / `logits_block_l2` 已明显不同，随后被 threshold/topk 放大成不同 token 序列。
4. transfer 事件数也不同（例如 128：27 vs 56），但这是首次分叉后的下游结果，不能独立证明 chunk/KV 是根因；按事件下标对齐只作诊断线索。

结论：**在对齐 FA 后端后，差异仍存在，只能说明 FA3/FA2 混用不是充分原因。** 当前实验仍同时改变 chunk shape、masked/unmasked kernel、ragged/paged cascade 与 KV 提交次数；在逐层、逐 block 对齐 KV/hidden/logits 前，不能排除 attention 数值累积、mask/metadata 或 position/frontier 问题。

产物：`/tmp/dllm_attr_matrix_report.json`，脚本 `tmp_dllm_attr_matrix.py`
门控：`SGLANG_DLLM_FORCE_RAGGED_FA2`、`SGLANG_DLLM_ATTR_LOG`（仅归因用）

### 10.5 仍未补跑

- `SGLANG_FLASHINFER_USE_PAGED=1` 服务 E2E
- 完整性能显存矩阵（第 8 节）

---

### 10.6 GSM8K 质量门槛（2026-07-17）

按 `dllm_blockwise_attention_gpu_validation.md` §7 四臂对照；GPU1（H200），端口 30001，`--disable-radix-cache`，eager/breakable 按矩阵，`num_examples=200`。

| 项 | 值 |
|---|---|
| main commit | `6bb29189382c502a4ae929deca9515c38ffb6bdc` |
| candidate commit | `9b0fd782fad93de1c7069adc48b36ee2eebfce97` + workspace patch |
| GSM8K SHA256 | `aa60d4c1dfe36263ec2a2fa963906bfcfd48519561a84b2945237c5545f36f87` |
| patch SHA256 | `f21b3541ab5198b3b0ec3d21154e4d6852fc1ccfe81bd36d753616ab39fff2b3` |
| untracked archive SHA256 | `e473c458a32ec97c35fc660bd7db02474cb90c6c5a37e280a6e585fb06ee12b3` |
| torch / FlashInfer | `2.11.0+cu128` / `0.6.12` |

| 组 | 代码 / 配置 | score | latency (s) | output throughput (tok/s) |
|---|---|---|---|---|
| A main | main worktree，原生 dLLM | **0.895** | 65.8 | 381.0 |
| B current_single | `tmp_baseline.yaml`（pbs=32），eager | **0.895** | 121.6 | 205.2 |
| C multi_eager | `dllm_config.yaml`（pbs=1024），eager | **0.915** | 58.6 | 435.0 |
| D multi_breakable | 同 C，prefill=`breakable` | **0.910** | 47.1 | 540.7 |

判定（文档 §7.5）：

```text
C: score_multi=0.915 > 0.88
   score_multi - score_main   = +0.020  (>= -0.01)  PASS
   score_multi - score_single = +0.020  (>= -0.01)  PASS
D: score_graph=0.910 > 0.88
   score_graph - score_multi  = -0.005  (>= -0.01)  PASS
```

对应合入门槛 **#7（GSM8K 质量不回退）：通过**。吞吐仅作记录，不作本轮质量失败依据。

产物目录：`benchmark_results/gsm8k_main_compare_20260717/`（含 HTML/JSON/server log、`SUMMARY.txt`、`VERSIONS.txt`）。

备注：首次评测因集群 `http_proxy` 导致 OpenAI/httpx 客户端对 `127.0.0.1` 走坏代理（SYN-SENT `:3128`）卡住；重跑时对 driver/eval 取消 proxy 后恢复。

---

## 11. 主要发现与建议

1. **Mask / cascade / isolation 正确性证据充分**。
2. **Breakable multi-block 主路径可用**；补测确认 eager↔Breakable 严格对齐。
3. ~~阻塞项：eager SM90 `operation not supported`~~ **已修复并验证关闭**（独立 FA2 custom-mask wrapper）。
4. **single↔multi 长请求分叉**：强制 FA2 后仍存在，排除 FA backend 差异是充分原因，但根因未定位；下一步需在相同语义 checkpoint 做逐层 KV/hidden/logits 数值对照。
5. **GSM8K 质量门槛已通过**（multi eager/breakable 相对 main/single 无回退）。
6. 仍建议补跑：paged fallback 服务、精简性能矩阵。

---

## 12. 产物路径

| 文件 | 用途 |
|---|---|
| `tmp_dllm_attn_gpu_validation.py` | Attention cascade / isolation / paged 验证 |
| `tmp_dllm_model_parity.py` | 模型级 parity / 稳定性客户端 |
| `tmp_dllm_attr_matrix.py` | **归因矩阵** single-FA2 vs multi-FA2 |
| `tmp_baseline.yaml` | single-block 配置 |
| `/tmp/dllm_eager_vs_graph_parity_rerun.json` | 补测 eager↔Breakable |
| `/tmp/dllm_single_vs_multi_parity_rerun.json` | 补测 single↔multi |
| `/tmp/dllm_attr_matrix_report.json` | **归因矩阵结果** |
| `tmp_attr_single_fa2_server.log` / `tmp_attr_multi_fa2_server.log` | 归因服务日志 |
| `benchmark_results/gsm8k_main_compare_20260717/` | **§7 GSM8K 四臂结果** |

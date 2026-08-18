# dLLM block-wise attention GPU 验证方案

本文验证 FlashInfer multi-block dLLM prefill 的 block-wise attention：同一 dLLM
block 内双向可见，每个 query 可见当前及此前 block，不可见未来 block。验证按
kernel、attention cascade、模型 parity、Breakable CUDA Graph 和性能五层进行。

推荐使用一张能够加载 `inclusionAI/LLaDA2.0-mini` 的 NVIDIA GPU；仓库 CI 将该
模型注册为 `1-gpu-large` 任务。

## 1. 环境检查

```bash
export CUDA_VISIBLE_DEVICES=0
export FLASHINFER_WORKSPACE_BASE=/tmp
export PYTHON=/root/miniconda3/envs/sglang-src/bin/python

nvidia-smi
$PYTHON -c "
import torch, flashinfer
print('torch:', torch.__version__)
print('cuda:', torch.version.cuda)
print('gpu:', torch.cuda.get_device_name())
print('flashinfer:', flashinfer.__version__)
assert torch.cuda.is_available()
"
```

FlashInfer 版本应与 `python/pyproject.toml` 一致，目前为 0.6.12。

## 2. GPU mask/kernel 单测

```bash
$PYTHON test/registered/unit/layers/test_flashinfer_dllm_prefill_mask.py -v
$PYTHON test/registered/unit/managers/test_prefill_adder.py -v
```

必须确认 GPU 测试实际执行而不是跳过：

```text
test_flashinfer_ragged_output_matches_dense_reference ... ok
```

该项将 FlashInfer ragged attention 与 Torch dense masked-softmax reference 比较，
容差为 `rtol=2e-2, atol=2e-2`。mask 测试当前应为 11 项通过，其中 4 项要求
FlashInfer + CUDA 且不能 skip；调度回归当前应为 24 项通过。

## 3. Attention 级完整正确性

现有 GPU 单测覆盖无 prefix 的 ragged block mask。还应补充以下两项 GPU 测试。

### 3.1 Prefix cascade parity

设置：

```text
P = 64    # cached prefix
E = 128   # current extend
B = 32    # dLLM block size
```

对比以下两种计算：

```text
A. ragged current（E × E block mask）
   + paged prefix（P，causal=False）
   + merge_state

B. Torch dense attention
   Q  长度 E
   KV 长度 P + E
   mask[i, j] = floor(j / B) <= floor((P + i) / B)
```

A、B 的 attention output 和 LSE 应在约定浮点容差内一致。该测试同时验证 ragged、
paged prefix 和 `merge_state` 的组合，而不只是 mask 构造。

### 3.2 Future-block isolation

固定 prefix 及前三个 block，只修改最后一个 block 的 K/V：

```text
原始 K/V              -> output_before
只修改最后一个 block  -> output_after
```

验收条件：

- 更早 block 的输出保持一致；
- 被修改 block 的输出发生变化；
- 测试不能只比较最终文本，否则无法可靠定位 attention 可见性错误。

## 4. 模型级对照配置

单 block 基线：

```yaml
# baseline.yaml
block_size: 32
prefill_block_size: 32
threshold: 0.95
```

Multi-block 配置：

```yaml
# multi.yaml
block_size: 32
prefill_block_size: 1024
threshold: 0.95
```

仓库根目录的 `dllm_config.yaml` 当前可作为 multi-block 配置。

### 4.1 单 block eager baseline

```bash
$PYTHON -m sglang.launch_server \
  --model-path inclusionAI/LLaDA2.0-mini \
  --trust-remote-code \
  --tp-size 1 \
  --mem-fraction-static 0.9 \
  --max-running-requests 4 \
  --attention-backend flashinfer \
  --dllm-algorithm LowConfidence \
  --dllm-algorithm-config baseline.yaml \
  --cuda-graph-backend-prefill disabled \
  --port 30000
```

### 4.2 Multi-block ragged eager

```bash
$PYTHON -m sglang.launch_server \
  --model-path inclusionAI/LLaDA2.0-mini \
  --trust-remote-code \
  --tp-size 1 \
  --mem-fraction-static 0.9 \
  --max-running-requests 4 \
  --attention-backend flashinfer \
  --dllm-algorithm LowConfidence \
  --dllm-algorithm-config dllm_config.yaml \
  --cuda-graph-backend-prefill disabled \
  --port 30000
```

### 4.3 Multi-block paged fallback

```bash
SGLANG_FLASHINFER_USE_PAGED=1 \
$PYTHON -m sglang.launch_server \
  --model-path inclusionAI/LLaDA2.0-mini \
  --trust-remote-code \
  --tp-size 1 \
  --mem-fraction-static 0.9 \
  --max-running-requests 4 \
  --attention-backend flashinfer \
  --dllm-algorithm LowConfidence \
  --dllm-algorithm-config dllm_config.yaml \
  --cuda-graph-backend-prefill disabled \
  --port 30000
```

该模式验证 `E × (P + E)` paged custom-mask fallback。它用于正确性覆盖，不应作为
长上下文的首选性能路径。

## 5. 模型 parity workload

对三组服务发送完全相同的请求，至少覆盖：

```text
prompt token length: 31, 32, 33, 1024, 1056, 2048, 2060
batch size:          1, 2, 4
temperature:         0
sampling_seed:       固定值
max_new_tokens:      128 或 512
```

长度 31/32/33 验证 block 边界；1056 验证完整的 1024-token prefill chunk；2060
同时验证多个 multi-block chunk 和非对齐 prompt tail。

记录并比较：

- `output_ids`；
- finish reason；
- 生成 token 数；
- 最终文本；
- 启用 `return_logprob` 时的 token logprob。

验收原则：

- single-block baseline 与 multi-block eager 优先要求 `output_ids` 完全一致；
- eager 与 Breakable graph 必须完全一致；
- ragged 与 paged fallback 允许底层浮点归约顺序不同，但 attention reference test
  必须在容差内，生成结果不应出现系统性分叉；
- 任何更早 block 会受未来 block token/KV 修改影响的情况均判失败。

## 6. Breakable CUDA Graph

```bash
$PYTHON -m sglang.launch_server \
  --model-path inclusionAI/LLaDA2.0-mini \
  --trust-remote-code \
  --tp-size 1 \
  --mem-fraction-static 0.9 \
  --max-running-requests 4 \
  --attention-backend flashinfer \
  --dllm-algorithm LowConfidence \
  --dllm-algorithm-config dllm_config.yaml \
  --cuda-graph-backend-prefill breakable \
  --cuda-graph-bs-prefill 32 128 512 1024 \
  --port 30000 \
  2>&1 | tee server_graph.log
```

使用超过 1024 token 的 prompt，确保至少执行一次 `extend_len=1024`。检查 capture
与运行时命中：

```bash
rg 'prefill|breakable|cuda graph' server_graph.log
rg 'breakable cuda graph: True' server_graph.log
```

相同请求分别发送给 eager 和 graph 服务，要求：

```text
eager output_ids == graph output_ids
```

稳定性测试还应覆盖：

1. 连续发送至少 20 次相同请求，确认 replay 后 mask/metadata 未被前一请求污染；
2. 交替发送命中 512 和 1024 token bucket 的请求；
3. 混合 batch size 1/2/4；
4. graph hit 持续为 true，且无 KV、shape、mask-indptr 错误。

## 7. GSM8K 质量门槛：与 main 分支对照

现有 NVIDIA 测试 `test/registered/dllm/test_llada2_mini.py` 使用 200 个 GSM8K
样本，质量门槛为 score > 0.88；其中 throughput > 350 和 batch-size 1 decode
speed > 250 token/s 属于性能门槛，不与本节的质量结论混用。

本轮先执行质量验收，并以本地 `main` 的精确 commit 作为基线。main 尚无
`prefill_block_size` 配置字段，因此 main 必须按其原生 dLLM 调度启动，不能给它
传入 `dllm_config.yaml`。当前分支则同时运行 single-block 控制组和 multi-block
候选组，以区分通用代码变化与 multi-block 本身的影响。

### 7.1 固定实验条件

四组测试必须使用：

- 同一物理 GPU，顺序运行，不能让两个服务同时争抢 GPU；
- 同一模型权重及 tokenizer：`inclusionAI/LLaDA2.0-mini`；
- 同一 Python、torch、FlashInfer 和 CUDA 环境；
- 同一份本地 GSM8K `test.jsonl`，记录 SHA256；
- `temperature=0`、`top_p=1`、5-shot、`max_tokens=512`；
- `num_examples=200`、`num_threads=128`；
- eager prefill，即 `--cuda-graph-backend-prefill disabled`；
- `--disable-radix-cache`，避免 GSM8K 共享 few-shot prefix 被 radix 命中后绕过
  大段 multi-block prefill；
- 相同的 `max-running-requests=4` 和 decode CUDA Graph 配置。

记录代码与数据版本：

```bash
export CURRENT_ROOT=$PWD
export MAIN_COMMIT=$(git rev-parse main)
export CURRENT_COMMIT=$(git rev-parse HEAD)

git diff --binary > /tmp/dllm_gsm8k_current.patch
git status --short > /tmp/dllm_gsm8k_current_status.txt
# `git diff` 不包含 untracked 文件；当前修复若仍有 untracked 源码，需一并归档。
git ls-files --others --exclude-standard -z | \
  tar --null -T - -czf /tmp/dllm_gsm8k_current_untracked.tar.gz

export GSM8K_DATA=$($PYTHON -c '
from sglang.test.simple_eval_gsm8k import GSM8K_URL
from sglang.utils import download_and_cache_file
print(download_and_cache_file(GSM8K_URL))
')
sha256sum "$GSM8K_DATA"
printf 'main=%s\ncurrent=%s\n' "$MAIN_COMMIT" "$CURRENT_COMMIT"
```

当前工作区可能包含尚未提交的修复，因此除 `CURRENT_COMMIT` 外还必须保存 patch 和
`git status`，并归档 untracked 文件；正式验收最好使用一个可复现的 candidate commit。
不要通过 checkout main 覆盖当前工作区；使用 detached worktree：

```bash
git worktree add --detach /tmp/sglang-main-gsm8k "$MAIN_COMMIT"
```

### 7.2 四臂对照矩阵

| 组 | 代码 | 配置 | 用途 |
|---|---|---|---|
| A | main worktree | main 原生 dLLM 配置 | 正式质量基线 |
| B | 当前工作区 | `tmp_baseline.yaml`，`prefill_block_size=32` | 当前代码 single-block 控制 |
| C | 当前工作区 | `dllm_config.yaml`，`prefill_block_size=1024`，eager | multi-block 主验收 |
| D | 当前工作区 | 同 C，但 prefill graph=`breakable` | graph 质量补充 |

A/B/C 是本轮最低必跑集合。D 可在 C 通过后执行；D 的 score 应与 C 一致或满足同一
质量门槛，同时继续执行第 6 节的逐请求 eager↔Breakable parity。

### 7.3 服务启动

main 基线 A：

```bash
cd /tmp/sglang-main-gsm8k
PYTHONPATH=/tmp/sglang-main-gsm8k/python \
$PYTHON -m sglang.launch_server \
  --model-path inclusionAI/LLaDA2.0-mini \
  --trust-remote-code \
  --tp-size 1 \
  --mem-fraction-static 0.9 \
  --max-running-requests 4 \
  --attention-backend flashinfer \
  --dllm-algorithm LowConfidence \
  --disable-radix-cache \
  --cuda-graph-bs 1 2 3 4 \
  --cuda-graph-backend-prefill disabled \
  --port 30000 2>&1 | tee /tmp/gsm8k_main_server.log
```

当前分支 B/C 只需切换 algorithm config：

```bash
cd "$CURRENT_ROOT"

# B: CONFIG=tmp_baseline.yaml
# C: CONFIG=dllm_config.yaml
CONFIG=dllm_config.yaml

PYTHONPATH="$CURRENT_ROOT/python" \
$PYTHON -m sglang.launch_server \
  --model-path inclusionAI/LLaDA2.0-mini \
  --trust-remote-code \
  --tp-size 1 \
  --mem-fraction-static 0.9 \
  --max-running-requests 4 \
  --attention-backend flashinfer \
  --dllm-algorithm LowConfidence \
  --dllm-algorithm-config "$CONFIG" \
  --disable-radix-cache \
  --cuda-graph-bs 1 2 3 4 \
  --cuda-graph-backend-prefill disabled \
  --port 30000 2>&1 | tee "/tmp/gsm8k_${CONFIG%.yaml}_server.log"
```

每组评测结束后先停止服务并确认 GPU 显存释放，再启动下一组。D 组将 C 命令中的
`--cuda-graph-backend-prefill disabled` 改为 `breakable`，并加入第 6 节的
`--cuda-graph-bs-prefill` buckets。

### 7.4 统一评测命令

四组均使用完全相同的命令；每次运行后立即复制结果，避免 `/tmp` 中同名 JSON/HTML
被下一组覆盖：

```bash
LABEL=multi_eager  # main / current_single / multi_eager / multi_breakable
mkdir -p /tmp/dllm_gsm8k_results

PYTHONPATH="$CURRENT_ROOT/python" \
$PYTHON -m sglang.test.run_eval \
  --port 30000 \
  --model inclusionAI/LLaDA2.0-mini \
  --eval-name gsm8k \
  --api completion \
  --num-examples 200 \
  --num-threads 128 \
  --num-shots 5 \
  --max-tokens 512 \
  --temperature 0 \
  --top-p 1 \
  --gsm8k-data-path "$GSM8K_DATA" \
  2>&1 | tee "/tmp/dllm_gsm8k_results/${LABEL}.log"

cp /tmp/gsm8k_inclusionAI_LLaDA2.0-mini.json \
  "/tmp/dllm_gsm8k_results/${LABEL}.json"
cp /tmp/gsm8k_inclusionAI_LLaDA2.0-mini.html \
  "/tmp/dllm_gsm8k_results/${LABEL}.html"
```

评测客户端可以统一使用当前工作区的 `run_eval`；服务端代码必须分别由对应 worktree
的 `PYTHONPATH` 加载。若两边 `run_eval` 或 GSM8K grader 存在差异，则改为固定使用
main 客户端，并在报告中记录。

### 7.5 质量判定

设 A/B/C/D 的 score 分别为 `score_main`、`score_single`、`score_multi` 和
`score_graph`。C 组通过必须同时满足：

```text
score_multi > 0.88
score_multi >= score_main - 0.01
score_multi >= score_single - 0.01
```

200 个样本下 0.01 对应 2 道题。若刚好落在边界或发生 2 道题以上回退，应保存逐题
HTML，列出 main/single 正确而 multi 错误的样本，并至少复跑这些失败样本；不能只用
总分将差异归因于算子误差。

D 组使用相同绝对与相对门槛：

```text
score_graph > 0.88
score_graph >= score_multi - 0.01
```

以下任一情况直接判失败：

- server crash、FlashInfer operation not supported、NaN 或 OOM；
- 有效完成样本数不足 200；
- 大量空输出、重复输出或异常 finish reason；
- multi-block 分数低于绝对门槛或相对 main/single 回退超过 0.01。

吞吐和延迟仍需记录，但本轮不以 main/current 的吞吐差作为质量失败依据；性能结论按
第 8 节的独立 warmup/测量矩阵给出。

最终报告至少记录：main/candidate commit、candidate patch SHA256、untracked archive
SHA256、GSM8K 数据 SHA256、GPU/torch/FlashInfer 版本、四组 server 参数、四组 score、
有效样本数、latency、output throughput，以及相对 main/single 的 score delta。

## 8. 性能和显存矩阵

```text
prefill_block_size: 32, 128, 512, 1024
prompt length:      512, 2048, 8192
batch size:         1, 2, 4
mode:               eager, breakable
```

每个组合至少 warmup 5 次、测量 30 次，记录：

- TTFT；
- input token throughput；
- 总请求延迟；
- GPU 峰值显存；
- graph hit；
- mask 生成时间；
- FlashInfer planning 时间。

重点观察 paged fallback 的 `E × (P + E)` mask：其显存随 prefix 增长。生产主路径
应确认实际使用 ragged `E × E` mask。

## 9. 合入门槛

以下条件全部满足后，multi-block prefill 才可视为完成 GPU 验证：

1. GPU kernel 测试实际执行且不跳过；
2. prefix cascade 与 dense reference 对齐；
3. future-block isolation 通过；
4. multi-block eager 与 single-block baseline 对齐；
5. Breakable replay 连续运行无 metadata 污染；
6. paged fallback 正确且无 mask-size 错误；
7. GSM8K 质量不回退；
8. 大 prompt 下无不可接受的显存增长或 OOM。

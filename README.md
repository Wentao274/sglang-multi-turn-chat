# sglang-multi-turn-chat

多轮长上下文对话压测框架。复用 `sglang.benchmark.serving` 内置多轮 wrapper 与指标计算；HTTP 客户端为内置客户端的 fork（`request_client.py`），自动解析 `usage.prompt_tokens_details.cached_tokens`，统计真实 prefix cache 命中率。使用 `ShareGPT_V3_unfiltered_cleaned_split.json` 全多轮数据；TPM 匀速爬坡至目标稳态后持续压测，检测衰减信号自动终止并输出 Markdown 报告。

**客户指标**：稳态 TPM ≥ 1 亿（100,000,000 tokens/min）即通过；并发请求数（上限 1000）作为 1 亿吞吐下的参考值展示，不参与判定。详见「测试逻辑与判断逻辑」章节。

报告格式对齐客户模板 `reports-temp.md`，开头输出**测试结论汇总**（客户 6 项指标通过/不通过 + 总体结论），后接 9 个章节：全程、连续稳态、验收明细、全程分轮、稳态分轮、stream/full_run、stream/steady、加压时间序列、实际长度与轮数。

## 架构

| 文件 | 职责 |
|---|---|
| `bench_multi_turn.py` | 主入口：编排加载→预热→爬坡调度→采集→监控→出报告，复用 serving.py 零件 |
| `build_multiturn_dataset.py` | 可选工具：false_qa / truthful_qa / dolly（每 N 条 QA 串成多轮会话）+ TM（从 gpt 文本解析真实多轮）→ 与 ShareGPT 合并输出统一 JSON（当前测试未使用，保留备用） |
| `sharegpt_multiturn.py` | ShareGPT 格式 JSON → `List[DatasetRow]`，去重保证会话不重复、每会话唯一 system prompt（不同连续会话内容不同），会话内多轮连续命中缓存 |
| `request_client.py` | sglang 内置 OpenAI 流式客户端的 fork：附带 `stream_options.include_usage`，解析 `usage.prompt_tokens`（真实输入 token）与 `prompt_tokens_details.cached_tokens`（真实 prefix cache 命中） |
| `probe_cache_report.py` | 压测前探测脚本：发两个请求验证服务端响应是否返回 `cached_tokens` 字段（确认服务端已开 `--enable-cache-report`） |
| `ramp_scheduler.py` | 非齐次泊松请求生成器：RPS 从 `start` 线性爬升到 `target`，稳态后持续压测 |
| `monitor.py` | 衰减监控器 + 时间序列快照采集：滑动窗口检测错误率/吞吐跌落/TTFT 飙升；全程记录 RPM/TPM/延迟快照 |
| `config.py` | 参数解析 + 把本框架参数映射成 serving.py 内置客户端读取的全局 `args` |
| `report.py` | 生成 Markdown 压测报告：测试结论汇总（7 项判定 + 并发信息项）+ 10 个章节对齐客户模板 |
| `test_cached_client.py` | 烟雾测试：mock sglang 服务端 + SSE 流，验证 cached_tokens/prompt_tokens_actual 解析与报告计算 |

被复用的 sglang 内置组件（不重造轮子）：
- `async_request_openai_chat_completions_cached`（`request_client.py`，内置客户端的 fork）—— `/v1/chat/completions` 流式 HTTP 客户端（TTFT/ITL 采集），额外解析 usage 中的 `prompt_tokens` / `cached_tokens`
- `wrap_multi_turn_request_func` —— 每轮把完整历史发给服务端，append assistant 回复，上下文随轮次自然增长
- `calculate_metrics` / `BenchmarkMetrics` —— 与 `python -m sglang.benchmark.serving` 完全一致的指标口径
- `get_tokenizer` / `download_and_cache_hf_file` / `flush_server_cache` / `wait_for_endpoint`

补齐内置工具的三个缺口：
- **缺口 A**：内置 `ShareGPTDataset` 只取前 2 轮（单轮）。本框架读原始 JSON，保留全多轮 user 消息，可选注入长 system prompt。
- **缺口 B**：内置 `get_request` 仅支持单一固定速率泊松/全量并发。本框架 `get_ramp_request` 实现匀速爬坡。
- **缺口 C**：内置 `async_request_openai_chat_completions` 不解析 `usage.prompt_tokens` / `prompt_tokens_details.cached_tokens`，无法统计真实 prefix cache 命中率。本框架 `request_client.py` fork 解决。

## 环境与依赖（uv）

本项目使用 [uv](https://docs.astral.sh/uv/) 管理虚拟环境和依赖。

### 1. 安装 uv

```bash
# Linux / macOS
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# 或通过 pip
pip install uv
```

### 2. 创建虚拟环境并安装依赖

```bash
# 创建虚拟环境（默认 .venv 目录，Python 3.10+）
uv venv

# 安装 sglang（增大超时避免 nvidia-* 大包下载失败）
UV_HTTP_TIMEOUT=600 uv pip install sglang

# 安装框架额外依赖
uv pip install -r requirements.txt
```

> sglang 的 `__init__.py` 导入链会拉入服务端运行时（flashinfer、torch 等），无法通过 `--no-deps` 跳过。增大 `UV_HTTP_TIMEOUT` 可解决 nvidia 大包下载超时问题。

> 如果 nvidia 包仍然失败，可单独重试：`UV_HTTP_TIMEOUT=600 uv pip install nvidia-cuda-nvdisaml nvidia-cuda-cccl`

### 3. 激活虚拟环境

```bash
# Linux / macOS
source .venv/bin/activate

# Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

### 4. 验证安装

```bash
python -c "import sglang.benchmark.serving; print('OK')"
```

> 也可不激活虚拟环境，直接用 `uv run python bench_multi_turn.py ...` 运行，uv 会自动使用 `.venv`。

## 配置（configs/.env）

框架自动读取 `configs/.env` 文件获取服务地址、模型名、tokenizer、API Key、数据集路径等参数，避免命令行过长。

```bash
# 1. 复制模板
cp configs/.env.example configs/.env

# 2. 编辑 configs/.env，填入实际值
BASE_URL=http://127.0.0.1:30000
MODEL=glm-5.3
TOKENIZER=/path/to/tokenizer
API_KEY=sk-xxxxx                          # 需要鉴权时填写，自动注入 OPENAI_API_KEY
DATASET_PATH=/path/to/ShareGPT_V3_unfiltered_cleaned_split.json
```

> `.env` 已在 `.gitignore` 中，不会被提交。CLI 参数会覆盖 `.env` 同名值。

## 数据集

框架使用 `ShareGPT_V3_unfiltered_cleaned_split.json` 单一数据集。推荐在 `configs/.env` 中设置 `DATASET_PATH`，也可通过 `--dataset-path` 指定：

**容量（min_turns=2、首轮去重后）：51,493 条可用会话**——满足推荐测试参数（`--num-sessions 50000` 可加载，900s 窗口实际消耗约 36,500 会话，余量充足）。

> 数据文件**不需要**放入仓库路径，用绝对路径或相对路径指定即可。如果不传 `--dataset-path`，框架会尝试从 HuggingFace 自动下载（仅 ShareGPT，需网络连通）。

### 多数据集合并（build_multiturn_dataset.py，可选工具）

4 个补充数据集均为单轮，通过 `build_multiturn_dataset.py` 转换为多轮会话并与 ShareGPT 合并：

| 数据集 | 原始格式 | 转换策略 | 可用会话 |
|---|---|---|---|
| `false_qa.jsonl` | UltraFeedback（单轮 QA，2,339 条） | 每条问题作为一个 user 轮，assistant 轮取评分最高的模型回答 | 纳入 QA 池 |
| `truthful_qa.jsonl` | UltraFeedback（单轮 QA，811 条） | 同上 | 纳入 QA 池 |
| `databricks-dolly-15k.jsonl` | Dolly（单轮指令，15,011 条） | instruction 作为 user 轮，带 context 的前置拼入（`Context:\n...\n\nInstruction: ...`） | 纳入 QA 池 |
| `TM_multi_turn_convs_split_token_wrap.jsonl` | ShareGPT 格式但单轮（281 条） | 真实多轮对话嵌在 gpt 回复文本内（`**User:**`/`**AI Assistant:**` 标签），正则解析还原 | 273 条真实多轮 |

QA 池（false_qa 2,339 + truthful_qa 811 + dolly 去重后 14,822 = 17,972 条）随机打散后**每 6 条串成一个多轮会话**（`--qa-per-session` 可调），加上 TM 解析的 273 条 → 新增 **3,269 条**多轮会话，与 ShareGPT 原始 94,145 条合并（随机 shuffle）输出单一 JSON 文件。

```bash
# 生成合并数据集（默认输入目录 D:\Maas\芯片测试\测试相关\测试数据集）
python build_multiturn_dataset.py \
  --input-dir "D:\Maas\芯片测试\测试相关\测试数据集" \
  --qa-per-session 6 \
  --output merged_multiturn.json

# 仅转换新数据集（不合并 ShareGPT，快速检查转换结果）
python build_multiturn_dataset.py --no-sharegpt --output converted_only.json
```

合并结果（min_turns≥2、首轮去重后可用会话数）：

| 数据集 | 可用会话（≥2 轮） |
|---|---|
| **纯 ShareGPT（当前使用）** | **51,493** |
| 合并后（可选） | 54,756（ShareGPT 51,493 + 新增 3,263） |

> **容量结论**：推荐测试参数（1.2 亿 TPM 加压、8192 system prompt、爬坡 300s + 稳态 600s）下 900s 窗口实际消耗约 **36,500 会话**（48.7 会话/s × 750s 等效满速窗口）——纯 ShareGPT（51,493）**完全满足**；`--num-sessions 50000` 可正常加载。当前测试仅使用 ShareGPT 单一数据集；`build_multiturn_dataset.py` 保留为可选工具，如需数据源多样性可随时生成合并数据集（生成后把 `DATASET_PATH` 指向 `merged_multiturn.json` 即可）。

**容量与防缓存保证**：加载时按首轮 user 文本做 md5 去重，确保会话内容不重复；`--system-prompt-len > 0` 时每个会话生成**独立随机** system prompt，不同连续会话内容不同，避免跨会话缓存一直命中；同一会话内多轮内容连续（每轮重发完整历史，prefix 与上一轮重合），命中缓存越高效率越高（测试指标）。若去重+过滤后可用会话数 < `--num-sessions`，直接报错并提示调整参数（不靠重复凑数）。`--sustain-seconds` 设定时，框架会预估所需会话数并在不足时给出警告。

## 服务端要求（统计真实 prefix cache 命中率）

要统计**真实**的 prefix cache 命中率，sglang 服务端启动时必须加 `--enable-cache-report`，否则 usage 中不返回 `prompt_tokens_details.cached_tokens`，报告中 cache hit 显示 N/A：

```bash
python -m sglang.launch_server --model-path <model> --enable-cache-report
```

客户端 fork（`request_client.py`）的工作方式：

1. 请求 payload 附带 `stream_options: {"include_usage": true}`，确保流式响应最终 chunk 携带 usage（OpenAI 标准；vLLM 必需，sglang 支持）。
2. 解析 `usage.prompt_tokens` → `output.prompt_tokens_actual`（服务端真实输入 token 数）。
3. 解析 `usage.prompt_tokens_details.cached_tokens` → `output.cached_tokens`（prefix cache 命中 token 数）。
4. 生成上限发送 **`max_tokens` 字段**（OpenAI 旧标准，vLLM/sglang/网关均识别），而非 `max_completion_tokens`（OpenAI 新字段，部分网关/旧版服务端不识别）——1007 事故：网关忽略该新字段，256 上限失效，实际每轮输出 ~1.7K token（计划的 6.8 倍），decode 预算/TPM/TPOT 全部失真。

**max_tokens 生效自检**：warmup 会话完成后自动校验各轮输出是否 ≤ `--max-tokens-per-turn` 上限（容差 +8 token）；超限立即打印 `[warn]`——说明服务端没有透传/遵守 max_tokens，验收数据不可用，应先修复网关再测。

报告使用优先级：**服务端报告的真实值 > 客户端估算值**。未开启 `--enable-cache-report` 时回退到估算值（prompt_len + Σ prior output_len）。

**cache_hit_rate / cache_observable_rate 语义**：

- `cache_hit_rate` = Σ cached_tokens / Σ 输入 tokens（全程或稳态）。可测性按**运行级**判定：只要任一请求返回过 `prompt_tokens_details`（即服务端已开 `--enable-cache-report`），整个运行视为可测——此时第 0 轮（冷前缀）命中 0% 记为真实的 `0.00`，而非 N/A。
- `cache_observable_rate` = 返回过 `prompt_tokens_details` 的成功请求占比。sglang 在 cached_tokens=0 时**省略**该字段（见服务端 `usage_processor._details_if_cached`，仅 count>0 时返回），因此第 0 轮普遍不可见。多轮场景下稳态轮几乎全命中，observable_rate ≈ (1 - 1/平均轮数)；该指标仅作信息参考，无验收项。
- 整个运行无任何 details（服务端未开 flag）时，`cache_hit_rate` 显示 **N/A**（而非误导性的 0.00），验收明细 `steady_cache_hit_rate` 实际值 None、结论 False。

压测前可用 `python probe_cache_report.py` 一条命令验证服务端确实返回该字段（详见"压测前探测缓存命中率字段"章节）。

## 测试逻辑与判断逻辑（新指标：TPM 1亿 + 并发 1000）

### 指标口径

- **稳态 TPM** = 稳态窗口内（爬坡结束后才发起首轮的会话）所有成功请求的（服务端报告输入 tokens + 输出 tokens）总和 ÷ 稳态时长 × 60。输入+输出合并计：多轮每轮重发完整历史，输入占大头，正是 prefix cache 优化的场景。
- **峰值并发**（`peak_concurrency`）= 信号量占用数（实际同时在执行的会话数，每个会话同一时刻只有 1 个在途 HTTP 请求），即服务端视角的真实峰值并发请求数。注意与监控快照中的 `inflight`（含排队等待信号量的会话）不同。

### 测试逻辑

1. **加压**：`--start-tpm 0 --target-tpm 120000000`，TPM 从零匀速爬坡至 1.2 亿（`--ramp-seconds`），之后持续压测（`--sustain-seconds`）。加压目标高于 1 亿验收线是**有意为之**：RPS 调度按计划 token（256/轮）估算，而模型自然停止时实际输出可能低于 256/轮，实际 TPM 略低于计划值，若加压目标=验收线，服务端跟得上也会被客户端调度限在 1 亿以下（详见 TPM 校准提示）。
2. **并发封顶**：`--max-concurrency 1000`。服务端跟不上时请求在客户端信号量排队，服务端最多同时收到 1000 个在途请求；报告展示 1 亿吞吐下实际压到的并发数（`peak_concurrency`，信息项）。
3. **会话池**：1.2 亿 TPM 下会话消耗极快（~49 会话/s），需要 `--min-turns 2`（ShareGPT 去重后 51,493 条）+ `--num-sessions 50000` 的池子；配合 `--system-prompt-len 8192` 增大每会话 token 数（平均 ~41,101 tokens/会话）以降低会话消耗速率（~49 会话/s，50K 池可支撑完整 900s 窗口（消耗 ~36,500）。若启动时出现容量警告，按提示缩短 `--sustain-seconds` 或减小 `--target-tpm`。
4. **稳态窗口**：仅统计爬坡结束后才发起首轮的会话；TPM 与延迟指标均取该窗口。
5. **衰减保护**：监控滑动窗口错误率 / 吞吐跌落 / TTFT p99，超阈值自动终止（防止压垮服务端）。

### 判断逻辑

| 指标 | 计算 | 判定 |
|---|---|---|
| `steady_tpm` | (稳态输入+输出 tokens) / 稳态时长 × 60 | ≥ `--accept-steady-tpm`（1 亿）→ **通过** |

- **TPM 是唯一的新增验收指标**：稳态 TPM ≥ 1 亿即通过（连同既有 6 项：吞吐/TTFT/TPOT/缓存命中）。
- **峰值并发为信息项（不判定）**：报告显示「1 亿 TPM 下并发数」的实际数值（如 1 亿 TPM 时并发 850），无论是否达到 1000 都不影响通过/不通过结论——客户只关心吞吐达标，并发数作为该吞吐下的参考值展示。

判定矩阵（一次性测试，不通过即出结论，不重测）：

| 稳态TPM | 结论 |
|---|---|
| ≥ 1亿 | **通过**（报告同时展示该吞吐下的并发数作为参考） |
| < 1亿 | 不通过：服务端能力不足或加压不够，检查衰减终止原因 |

### TPM 校准提示

- `--target-tpm` 是加压目标（计划发送速率）；`actual_tpm_stable` 是服务端实际处理速率（验收依据）。默认（无 `--ignore-eos`）模型自然停止，实际输出 = min(自然长度, 256)，实际 TPM 略低于计划：8192 system prompt 占大头的推荐配置下 ≈ 计划的 ~99%，输出占比更高的场景可低至 ~85%。
- **1007 事故教训**：客户端曾发送 `max_completion_tokens`（OpenAI 新字段），网关不识别也不报错——256 上限被静默忽略，实际每轮输出 ~1.7K token（6.8 倍），decode 预算/TPM/TPOT 全部失真。现改发 `max_tokens`（旧标准），warmup 后自动校验是否生效（超限打印 `[warn]`）。若仍告警，先单发一条 `max_tokens=256` 的请求检查 `usage.completion_tokens` 是否 ≤ 256，并排查网关字段透传。
- 因此推荐 `--target-tpm 120000000`（加压目标）+ `--accept-steady-tpm 100000000`（验收线）：无论校准比例是 85% 还是 99%，实际稳态 TPM 都能越过 1 亿验收线（85% → 1.02 亿，99% → 1.19 亿）；若服务端容量不足，实际 TPM 体现真实瓶颈，仍按验收线判定。
- 测试为一次性判定（不重测）：不通过即出最终结论。
- `--max-concurrency`（推荐 1000）仍生效：限制同时在途的请求数，报告展示 1 亿吞吐下实际压到的并发数（`peak_concurrency`）作为参考。
- 若要求计划=实际（每轮固定 256 输出），加 `--ignore-eos`（但不模拟真实对话，不推荐用于客户验收场景）。

## 用法示例

### 推荐命令（.env + 精简 CLI）

配置好 `configs/.env` 后，命令行只需传压测参数：

```bash
# 方式一：配置好环境，直接运行
python bench_multi_turn.py \
  --num-sessions 50000 --num-turns 14 --min-turns 2 \
  --system-prompt-len 8192 \
  --max-tokens-per-turn 256 \
  --start-tpm 0 --target-tpm 120000000 \
  --ramp-seconds 300 --sustain-seconds 600 \
  --drain-timeout 600 \
  --max-concurrency 1000 \
  --cache-report \
  --max-error-rate 0.10 \
  --throughput-drop-ratio 0.3 \
  --max-ttft-p99-ms 60000 \
  --baseline-warmup-seconds 120 \
  --monitor-window 30 --monitor-interval 10 \
  --accept-steady-tpm 100000000 \
  --accept-peak-concurrency 1000 \
  --accept-request-rps 0.6 \
  --accept-success-rate 0.995 \
  --accept-ttft-p50-ms 8000 \
  --accept-ttft-p95-ms 30000 \
  --accept-tpot-p50-ms 30 \
  --accept-tpot-p95-ms 45 \
  --accept-cache-hit-rate 0.6 \
  --accept-zero-429 0 \
  --accept-usage-complete 1 \
  --output-file result.jsonl \
  --report-md report.md \
  --output-details \
  --tag glm-100m-tpm-1000conc
```

> **注意**：`--target-tpm` 是加压目标（实际发送速率），`--accept-steady-tpm` 是验收要求（期望达到的指标）。两者不同：加压目标应根据服务实际承受能力设定，验收要求是期望达标的门槛。`--avg-tokens-per-request` 默认 0=自动估算（从数据集每会话的轮数和 token 长度推算），无需手动设置。推荐配置 `--target-tpm 120000000 --accept-steady-tpm 100000000`——加压目标比 1 亿验收线高 20%，覆盖输出自然停止造成的实际<计划偏差（见下）。
>
> **TPM 校准提示**：默认（无 `--ignore-eos`）模型自然停止，实际输出 = min(自然长度, 256)，RPS 调度按计划 token 估算，因此**实际 TPM 略低于计划 TPM**（推荐配置下 ~99%）。加压目标 1.2 亿保证实际稳态 TPM ≥ 1 亿验收线。测试为一次性判定（不重测），实际压到的负载以报告 `actual_tpm_stable` 为准。

```bash
# 方式二：不激活虚拟环境，用 uv run（自动使用 .venv）
uv run python bench_multi_turn.py --num-sessions 50000 --num-turns 14 --min-turns 2 ...
```

> `--base-url`、`--model`、`--tokenizer`、`--api-key`、`--dataset-path` 自动从 `configs/.env` 读取，CLI 同名参数可覆盖。

### 推荐命令参数详解

推荐命令共 33 个参数，按功能分为 6 组：

**数据集参数（需求1：数据集 + 容量 + 防重复）**

| 参数 | 示例值 | 含义 |
|---|---|---|
| `--num-sessions` | `50000` | 加载的会话（对话）总数；去重后不足会直接报错并提示调整参数 |
| `--num-turns` | `14` | 每会话最大对话轮数上限；自然轮数不足此值的会话按实际轮数执行 |
| `--min-turns` | `2` | 仅保留 user 轮数 ≥ 该值的原始对话（过滤门槛）；ShareGPT 去重后 51,493 条满足 ≥2 轮 |
| `--max-tokens-per-turn` | `256` | 每次 API 请求的 max_tokens 上限（payload 发送 `max_tokens` 字段；warmup 后自动校验生效性，超限告警） |
| `--system-prompt-len` | `8192` | 每会话生成独立随机 system prompt 的 token 长度（防缓存命中 + 首轮即长上下文）；同时增大每会话 token 数，降低会话消耗速率 |
| `--num-shared-prefixes` | （未传，默认 0） | 共享 system prompt 组数；默认 0=每会话唯一前缀（不同连续会话内容不同，符合客户要求）；>0=按 round-robin 分配共享前缀（跨会话也命中，非推荐模式） |

**压测调度参数（需求2：起压点匀速爬坡至稳态）**

| 参数 | 示例值 | 含义 |
|---|---|---|
| `--start-tpm` | `0` | 爬坡起始 TPM（tokens/分钟），0 表示从零起压 |
| `--target-tpm` | `120000000` | 稳态目标 TPM，按 avg-tokens-per-request 折算为会话发送速率 RPS |
| `--avg-tokens-per-request` | （未传，默认 0） | TPM→RPS 折算系数；0=自动从数据集按每会话总 token 数估算 |
| `--ramp-seconds` | `300` | TPM 从 start 线性爬升到 target 的时长（秒） |
| `--sustain-seconds` | `600` | 到达稳态后持续压测时长（秒）；爬坡+稳态共 900s |
| `--drain-timeout` | `600` | 排空超时（秒）：调度结束后等待在途会话完成的上限，超时强制取消剩余任务，防止过载/服务器挂起场景无限排空 |

**并发控制**

| 参数 | 示例值 | 含义 |
|---|---|---|
| `--max-concurrency` | `1000` | 最大并发会话数（信号量上限），超过则新会话排队等待 |

**衰减监控参数（需求3：异常/衰减终止）**

| 参数 | 示例值 | 含义 |
|---|---|---|
| `--max-error-rate` | `0.10` | 滑动窗口错误率阈值，达到即终止测试（验收要求错误率 ≤0.5%，0.10 可快速止损） |
| `--throughput-drop-ratio` | `0.3` | 窗口吞吐跌至稳态基线的 30% 即终止（崩溃级兜底） |
| `--max-ttft-p99-ms` | `60000` | 窗口 TTFT p99 阈值，超过即终止；为验收线（30s）的 2 倍——30s 窗口内 p99≈前两大值，若卡在 30s 会因 2 个慢请求提前杀死本可通过验收（p95≤30s 允许 5% 超标）的测试 |
| `--baseline-warmup-seconds` | `120` | 进入稳态后等待该时长再锁定吞吐基线（避开爬坡完成波，防误判） |
| `--monitor-window` | `30` | 衰减检测滑动窗口时长（秒） |
| `--monitor-interval` | `10` | 监控检查间隔（秒） |
| `--cache-report` | （开关） | 采集 prefix cache 命中统计（sglang 服务端需开 --enable-cache-report） |

**验收要求参数（报告"测试结论汇总"与"验收明细"章节的要求值）**

| 参数 | 示例值 | 含义 |
|---|---|---|
| `--accept-steady-tpm` | `100000000` | 验收要求：稳态 TPM ≥ 1 亿 |
| `--accept-peak-concurrency` | `1000` | 信息项：报告展示 1 亿 TPM 下实际压到的并发数（不参与通过/不通过判定） |
| `--accept-request-rps` | `0.6` | 验收要求：稳态 RPS |
| `--accept-success-rate` | `0.995` | 验收要求：成功率 |
| `--accept-ttft-p50-ms` | `8000` | 验收要求：TTFT p50 |
| `--accept-ttft-p95-ms` | `30000` | 验收要求：TTFT p95 |
| `--accept-tpot-p50-ms` | `30` | 验收要求：TPOT p50 |
| `--accept-tpot-p95-ms` | `45` | 验收要求：TPOT p95 |
| `--accept-cache-hit-rate` | `0.6` | 验收要求：稳态 cache hit rate |
| `--accept-zero-429` | `0` | 验收要求：429 次数上限 |
| `--accept-usage-complete` | `1` | 验收要求：usage 完整（1=必须） |

**输出参数**

| 参数 | 示例值 | 含义 |
|---|---|---|
| `--output-file` | `result.jsonl` | JSONL 结果文件名（自动放入输出子目录） |
| `--report-md` | `report.md` | Markdown 报告文件名（自动放入输出子目录） |
| `--output-details` | （开关） | JSONL 中附带每轮明细（input_lens/ttfts/itls/errors/cached_tokens/prompt_tokens_actual） |
| `--tag` | `glm-100m-tpm-1000conc` | 结果 tag，写入 JSONL 便于归档检索 |

### 其他示例

#### 直接用 RPS 爬坡（不通过 TPM 折算）

```bash
python bench_multi_turn.py \
  --num-sessions 200 --num-turns 8 \
  --start-rps 0 --target-rps 5 --ramp-seconds 60 \
  --max-concurrency 128
```

#### 纯吞吐（忽略 EOS，每轮固定满长，关闭衰减监控）

```bash
python bench_multi_turn.py \
  --num-sessions 100 --num-turns 8 --max-tokens-per-turn 512 \
  --target-rps 2 --ignore-eos --disable-monitor
```

### 关键参数

| 参数 | 说明 |
|---|---|
| `--env-file` | .env 文件路径（默认 `configs/.env`），自动读取服务/模型/数据集配置 |
| `--base-url` | 推理服务地址（可从 .env 读取） |
| `--model` | 模型名，留空自动探测 `/v1/models`（可从 .env 读取） |
| `--api-key` | API Key，自动注入 `OPENAI_API_KEY` 环境变量（可从 .env 读取） |
| `--tokenizer` | tokenizer 名/路径，留空则用 `--model`（可从 .env 读取） |
| `--dataset-path` | 数据集 JSON 路径（ShareGPT V3，可从 .env 读取） |
| `--num-sessions` | 会话数（每次发一个完整多轮对话） |
| `--num-turns` | 每会话最大轮数上限；自然轮数不足此值的会话按实际轮数 |
| `--min-turns` | 仅保留 user 轮数不少于该值的原始对话（变长轮次过滤门槛，默认 2） |
| `--max-tokens-per-turn` | 每轮生成 max_tokens 上限 |
| `--system-prompt-len` | >0 注入该长度随机 system prompt，首轮即长上下文 |
| `--num-shared-prefixes` | 共享 system prompt 组数；默认 0=每会话唯一前缀（不同连续会话内容不同，符合客户要求）；>0=按 round-robin 分配共享前缀（跨会话也命中，非推荐模式） |
| `--target-tpm` | 稳态目标 TPM，按 `--avg-tokens-per-request` 折算 RPS（默认 0=自动估算） |
| `--start-rps`/`--target-rps` | 直接指定 RPS（被 `--target-tpm` 覆盖） |
| `--ramp-seconds` | RPS 线性爬坡时长 |
| `--sustain-seconds` | 稳态持续时长；留空表示发完全部 `--num-sessions` |
| `--ignore-eos` | 忽略 EOS 强制满长（吞吐场景）；默认尊重 EOS 模拟真实对话 |
| `--max-concurrency` | 最大并发会话数 |
| `--cache-report` | 采集 prefix cache 命中统计（fork 客户端默认已解析，无需此开关） |
| `--max-error-rate` | 窗口错误率阈值，超过即终止（默认 0.10） |
| `--throughput-drop-ratio` | 窗口吞吐跌至稳态基线该比例即终止（默认 0.5） |
| `--max-ttft-p99-ms` | 窗口 TTFT p99 阈值，超过即终止（默认 30000） |
| `--baseline-warmup-seconds` | 稳态后建立吞吐基线的时长，之后才开始衰减判定（默认 30） |
| `--monitor-window` | 衰减检测滚动窗口（默认 30s） |
| `--disable-monitor` | 关闭衰减监控，跑完全部 |
| `--output-dir` | 结果根目录（默认 `results`），每次执行创建 `model-YYYYMMDD-HHMMSS` 子目录 |
| `--output-file` | JSONL 文件名（仅文件名，自动放入子目录） |
| `--report-md` | Markdown 报告文件名（仅文件名，自动放入子目录，默认 `report.md`） |
| `--accept-steady-tpm` | 验收要求：稳态 TPM |
| `--accept-peak-concurrency` | 信息项：1 亿 TPM 下的并发数展示（不判定） |
| `--accept-request-rps` | 验收要求：稳态 RPS（默认 0.6） |
| `--accept-success-rate` | 验收要求：成功率（默认 0.995） |
| `--accept-ttft-p50-ms` | 验收要求：TTFT p50（默认 8000） |
| `--accept-ttft-p95-ms` | 验收要求：TTFT p95（默认 30000） |
| `--accept-tpot-p50-ms` | 验收要求：TPOT p50（默认 30） |
| `--accept-tpot-p95-ms` | 验收要求：TPOT p95（默认 45） |
| `--accept-cache-hit-rate` | 验收要求：cache hit rate（默认 0.6） |
| `--accept-zero-429` | 验收要求：429 次数上限（默认 0） |
| `--accept-usage-complete` | 验收要求：usage 完整（默认 1） |

## 输出报告

报告对齐客户模板 `reports-temp.md`，报告开头含测试结论汇总，共 10 个章节：

| 章节 | 内容 |
|---|---|
| **测试结论汇总** | 7 项判定指标（请求吞吐/TTFT P50/P95/TPOT P50/P95/缓存命中率/稳态TPM）的要求/实际/结论表 + 总体结论（通过/不通过）+ 1 项并发信息行（1 亿 TPM 下并发数，不判定）；服务端未开缓存报告时命中率显示 N/A（不参与判定） |
| **全程** | total_requests/transport_success/success/truncated/actual_tokens/TPM/cache_hit_rate/peak_concurrency 全局表 + 延迟分位表(avg/p50/p75/p90/p95/p99) |
| **连续稳态** | 是否获得稳态窗口 |
| **验收明细** | 15 个验收行(实际/要求/结论/必过)：steady_tpm、continuous_steady_window、request_throughput_rps、success_rate、error_rate、http_429_rate、ttft_ms_p50、ttft_ms_p95、tpot_ms_p50、tpot_ms_p95、steady_cache_hit_rate、peak_concurrency、zero_429、usage_complete、length_profile |
| **全程分轮** | 每轮汇总表(请求数/req/s/input/output tok/s/cache hit) + 每轮延迟表(Round 0..N) |
| **稳态分轮** | 同上，仅稳态会话 |
| **stream / full_run** | 全程流式延迟表 |
| **stream / steady** | 稳态流式延迟表 |
| **加压时间序列** | elapsed/phase(ramp/stabilizing)/scheduled RPM/actual RPM/actual TPM/完整窗口/RPM 受限 |
| **实际长度与轮数** | JSON 块：input/output_mean/rounds_per_session 分布 |

同时输出 JSONL 文件（含 `full`/`steady` 双套指标 + `monitor_history` 时间序列）。

每次执行自动在 `--output-dir`（默认 `results`）下创建子目录，格式为 `{模型名}-{YYYYMMDD-HHMMSS}`，所有输出文件自动放入该子目录：

```
results/
  glm-5.3-20260929-193500/
    multi_turn_sglang-oai-chat_1000s_14t.jsonl
    report.md
```

| 参数 | 说明 |
|---|---|
| `--output-dir` | 结果根目录（默认 `results`） |
| `--output-file` | JSONL 文件名（仅文件名，自动放入子目录；留空自动命名） |
| `--report-md` | Markdown 报告文件名（仅文件名，自动放入子目录；留空默认 `report.md`） |

> 注：多轮模式下输入 token 优先采用服务端 usage 报告的 `prompt_tokens`（真实值）；服务端未报告时回退估算（round-0 prompt + 前序轮次 output_len 之和）。输出 token 逐轮精确统计。

## 烟雾测试

连通性自检可参考 `D:\Maas\客户反馈\多轮对话\glm_multiturn_messages.py`（单会话多轮客户端），先跑通 `/v1/chat/completions` 多轮链路再压测。

缓存统计自检（本地 mock sglang 服务端 + SSE 流，验证 cached_tokens/prompt_tokens_actual 解析与报告计算）：

```bash
python test_cached_client.py
```

## 压测前探测缓存命中率字段

正式压测前用 `probe_cache_report.py` 快速确认服务端响应 `usage.prompt_tokens_details.cached_tokens` 字段存在（服务端需 `--enable-cache-report` 启动）：

```bash
python probe_cache_report.py [--base-url http://127.0.0.1:30000] [--model glm-5.3]
```

脚本发送两次请求：请求 A（全新上下文，预期 cached_tokens=0）、请求 B（重发完整历史+追问，模拟多轮下一轮，预期 cached_tokens>0），逐请求打印原始 usage 与命中率 = cached_tokens / prompt_tokens。若输出 `[FAIL]`，正式压测报告中缓存命中率将显示 N/A。

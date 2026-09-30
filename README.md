# sglang-multi-turn-chat

多轮长上下文对话压测框架。复用 `sglang.benchmark.serving` 内置多轮 wrapper 与指标计算；HTTP 客户端为内置客户端的 fork（`request_client.py`），自动解析 `usage.prompt_tokens_details.cached_tokens`，统计真实 prefix cache 命中率。使用 `ShareGPT_V3_unfiltered_cleaned_split.json` 全多轮数据；TPM 匀速爬坡至目标稳态后持续压测，检测衰减信号自动终止并输出 Markdown 报告。

报告格式对齐客户模板 `reports-temp.md`，开头输出**测试结论汇总**（客户 6 项指标通过/不通过 + 总体结论），后接 9 个章节：全程、连续稳态、验收明细、全程分轮、稳态分轮、stream/full_run、stream/steady、加压时间序列、实际长度与轮数。

## 架构

| 文件 | 职责 |
|---|---|
| `bench_multi_turn.py` | 主入口：编排加载→预热→爬坡调度→采集→监控→出报告，复用 serving.py 零件 |
| `sharegpt_multiturn.py` | ShareGPT V3 → `List[DatasetRow]`，去重保证会话不重复、每会话唯一 system prompt（不同连续会话内容不同），会话内多轮连续命中缓存 |
| `request_client.py` | sglang 内置 OpenAI 流式客户端的 fork：附带 `stream_options.include_usage`，解析 `usage.prompt_tokens`（真实输入 token）与 `prompt_tokens_details.cached_tokens`（真实 prefix cache 命中） |
| `probe_cache_report.py` | 压测前探测脚本：发两个请求验证服务端响应是否返回 `cached_tokens` 字段（确认服务端已开 `--enable-cache-report`） |
| `ramp_scheduler.py` | 非齐次泊松请求生成器：RPS 从 `start` 线性爬升到 `target`，稳态后持续压测 |
| `monitor.py` | 衰减监控器 + 时间序列快照采集：滑动窗口检测错误率/吞吐跌落/TTFT 飙升；全程记录 RPM/TPM/延迟快照 |
| `config.py` | 参数解析 + 把本框架参数映射成 serving.py 内置客户端读取的全局 `args` |
| `report.py` | 生成 Markdown 压测报告：测试结论汇总（6 项客户指标）+ 9 个章节对齐客户模板 |
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

框架使用 `ShareGPT_V3_unfiltered_cleaned_split.json`。推荐在 `configs/.env` 中设置 `DATASET_PATH`，也可通过 `--dataset-path` 指定：

> 数据文件**不需要**放入仓库路径，用绝对路径或相对路径指定即可。如果不传 `--dataset-path`，框架会尝试从 HuggingFace 自动下载（需网络连通）。

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

报告使用优先级：**服务端报告的真实值 > 客户端估算值**。未开启 `--enable-cache-report` 时回退到估算值（prompt_len + Σ prior output_len）。

**cache_hit_rate / cache_observable_rate 语义**：

- `cache_hit_rate` = Σ cached_tokens / Σ 输入 tokens（全程或稳态）。可测性按**运行级**判定：只要任一请求返回过 `prompt_tokens_details`（即服务端已开 `--enable-cache-report`），整个运行视为可测——此时第 0 轮（冷前缀）命中 0% 记为真实的 `0.00`，而非 N/A。
- `cache_observable_rate` = 返回过 `prompt_tokens_details` 的成功请求占比。sglang 在 cached_tokens=0 时**省略**该字段（见服务端 `usage_processor._details_if_cached`，仅 count>0 时返回），因此第 0 轮普遍不可见。多轮场景下稳态轮几乎全命中，observable_rate ≈ (1 - 1/平均轮数)；该指标仅作信息参考，无验收项。
- 整个运行无任何 details（服务端未开 flag）时，`cache_hit_rate` 显示 **N/A**（而非误导性的 0.00），验收明细 `steady_cache_hit_rate` 实际值 None、结论 False。

压测前可用 `python probe_cache_report.py` 一条命令验证服务端确实返回该字段（详见"压测前探测缓存命中率字段"章节）。

## 用法示例

### 推荐命令（.env + 精简 CLI）

配置好 `configs/.env` 后，命令行只需传压测参数：

```bash
# 方式一：配置好环境，直接运行
python bench_multi_turn.py \
  --num-sessions 1000 --num-turns 14 --min-turns 8 --max-tokens-per-turn 256 \
  --system-prompt-len 2048 \
  --start-tpm 0 --target-tpm 214000 \
  --ramp-seconds 300 --sustain-seconds 600 \
  --max-concurrency 500 \
  --cache-report \
  --max-error-rate 0.10 \
  --throughput-drop-ratio 0.3 \
  --max-ttft-p99-ms 60000 \
  --baseline-warmup-seconds 120 \
  --monitor-window 30 --monitor-interval 10 \
  --accept-steady-tpm 100000000 \
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
  --tag glm-multiturn-steady
```

> **注意**：`--target-tpm` 是加压目标（实际发送速率），`--accept-steady-tpm` 是验收要求（期望达到的指标）。两者不同：加压目标应根据服务实际承受能力设定，验收要求是期望达标的门槛。`--avg-tokens-per-request` 默认 0=自动估算（从数据集每会话的轮数和 token 长度推算），无需手动设置。`--target-tpm 214000` 基于上次测试服务端实测容量 267K TPM 的 80% 安全余量。
>
> **TPM 校准提示**：默认（无 `--ignore-eos`）模型自然停止，实际输出 ≈170 tokens/轮（低于计划的 256），RPS 调度按计划 token 估算，因此**实际 TPM ≈ 计划 TPM 的 85%**。若需精确压到指定实际负载，首跑后读报告 `actual_tpm_stable` 按比例上调 `--target-tpm`（如需实际 214K → 设 250K）；若要求计划=实际（每轮固定 256 输出），加 `--ignore-eos`（但不模拟真实对话）。

```bash
# 方式二：不激活虚拟环境，用 uv run（自动使用 .venv）
uv run python bench_multi_turn.py --num-sessions 1000 --num-turns 14 --min-turns 8 ...
```

> `--base-url`、`--model`、`--tokenizer`、`--api-key`、`--dataset-path` 自动从 `configs/.env` 读取，CLI 同名参数可覆盖。

### 推荐命令参数详解

推荐命令共 29 个参数，按功能分为 6 组：

**数据集参数（需求1：数据集 + 容量 + 防重复）**

| 参数 | 示例值 | 含义 |
|---|---|---|
| `--num-sessions` | `1000` | 加载的会话（对话）总数；去重后不足会直接报错。900s 窗口按计划仅消费 ~60-70 个会话（0.063 sessions/s），1000 已有 15 倍余量且显著缩短加载/分词时间 |
| `--num-turns` | `14` | 每会话最大轮数上限；自然轮数不足此值的会话按实际轮数执行 |
| `--min-turns` | `8` | 仅保留 user 轮数 ≥ 该值的原始对话（过滤门槛）；去重后 5,281 条满足 ≥8 轮 |
| `--max-tokens-per-turn` | `256` | 每轮 API 请求的 max_tokens 上限 |
| `--system-prompt-len` | `2048` | 每会话生成独立随机 system prompt 的 token 长度（防缓存命中 + 首轮即长上下文） |
| `--num-shared-prefixes` | （未传，默认 0） | 共享 system prompt 组数；默认 0=每会话唯一前缀（不同连续会话内容不同，符合客户要求）；>0=按 round-robin 分配共享前缀（跨会话也命中，非推荐模式） |

**压测调度参数（需求2：起压点爬坡至稳态）**

| 参数 | 示例值 | 含义 |
|---|---|---|
| `--start-tpm` | `0` | 爬坡起始 TPM（tokens/分钟），0 表示从零起压 |
| `--target-tpm` | `214000` | 稳态目标 TPM；按 avg-tokens-per-request 折算为会话发送速率 RPS |
| `--avg-tokens-per-request` | （未传，默认 0） | TPM→RPS 折算系数；0=自动从数据集按轮数和 token 长度估算每会话总 token 数 |
| `--ramp-seconds` | `300` | RPS 从 start 线性爬升到 target 的时长（秒） |
| `--sustain-seconds` | `600` | 到达稳态后持续压测时长（秒）；爬坡+稳态共 900s |

**并发控制**

| 参数 | 示例值 | 含义 |
|---|---|---|
| `--max-concurrency` | `500` | 最大并发会话数（信号量上限），超过则新会话排队等待 |

**衰减监控参数（需求3：异常/衰减终止）**

| 参数 | 示例值 | 含义 |
|---|---|---|
| `--max-error-rate` | `0.10` | 滑动窗口错误率阈值，达到即终止测试；验收要求错误率 ≤0.5%，0.10（30 样本窗口中 3 个错误）即可快速止损，避免为注定不通过的测试白烧时间 |
| `--throughput-drop-ratio` | `0.3` | 窗口吞吐跌至稳态基线的 30% 即终止（崩溃级兜底） |
| `--max-ttft-p99-ms` | `60000` | 窗口 TTFT p99 毫秒阈值，超过即终止；设为验收线（30s）的 2 倍——30s 窗口内 p99≈前两大值，若卡在 30s 会因 2 个慢请求提前杀死本可通过验收（p95≤30s 允许 5% 超标）的测试 |
| `--baseline-warmup-seconds` | `120` | 进入稳态后等待该时长再锁定吞吐基线（避开爬坡完成波，防误判） |
| `--monitor-window` | `30` | 衰减检测滑动窗口时长（秒） |
| `--monitor-interval` | `10` | 监控检查间隔（秒） |
| `--cache-report` | （开关） | 兼容保留：sglang 内置客户端需此开关才解析 cached_tokens；本框架 fork（`request_client.py`）已无条件解析 usage，无需此开关 |

**验收要求参数（报告"验收明细"与"测试结论汇总"章节的要求值）**

| 参数 | 示例值 | 含义 |
|---|---|---|
| `--accept-steady-tpm` | `100000000` | 要求：稳态实际 TPM ≥ 该值 |
| `--accept-request-rps` | `0.6` | 要求：稳态请求吞吐 ≥ 该值（轮次/秒） |
| `--accept-success-rate` | `0.995` | 要求：稳态成功率 ≥ 该值 |
| `--accept-ttft-p50-ms` | `8000` | 要求：稳态 TTFT p50 ≤ 该值（毫秒） |
| `--accept-ttft-p95-ms` | `30000` | 要求：稳态 TTFT p95 ≤ 该值（毫秒） |
| `--accept-tpot-p50-ms` | `30` | 要求：稳态 TPOT p50 ≤ 该值（毫秒） |
| `--accept-tpot-p95-ms` | `45` | 要求：稳态 TPOT p95 ≤ 该值（毫秒） |
| `--accept-cache-hit-rate` | `0.6` | 要求：稳态 prefix cache 命中率 ≥ 该值 |
| `--accept-zero-429` | `0` | 要求：429 限流次数 ≤ 该值 |
| `--accept-usage-complete` | `1` | 要求：usage 字段完整（每轮都有 usage 返回） |

**输出参数**

| 参数 | 示例值 | 含义 |
|---|---|---|
| `--output-file` | `result.jsonl` | JSONL 结果文件名（自动放入 `results/{模型}-{时间戳}/` 子目录） |
| `--report-md` | `report.md` | Markdown 报告文件名（自动放入同上子目录） |
| `--output-details` | （开关） | JSONL 中附带每请求的 input_lens/ttfts/itls/errors/cached_tokens/prompt_tokens_actual 明细（每轮会话的真实缓存命中数据） |
| `--tag` | `glm-multiturn-steady` | 测试标签，写入 JSONL 便于归档检索 |

命令逐段对应需求：

| 参数 | 对应需求 |
|---|---|
| `.env` 中 `DATASET_PATH`（去重后不足会报错） | 需求1：ShareGPT 数据集 + 压测容量 |
| `--num-sessions 1000` | 需求1：压测容量 |
| `--system-prompt-len 2048` | 需求1：长上下文；每会话唯一随机前缀（不同连续会话内容不同），会话内多轮连续命中缓存（测试指标：越高效率越高） |
| `--start-tpm 0 --target-tpm 214000 --ramp-seconds 300` | 需求2：起压点匀速爬坡至目标稳态 |
| `--sustain-seconds 600` | 需求2：稳态后持续压测 |
| 指标自动只统计 `start_time >= 爬坡结束` 的会话 | 需求2：从稳态开始记录指标 |
| `--max-error-rate / --throughput-drop-ratio / --max-ttft-p99-ms` | 需求3：异常/衰减信号终止 |
| `--baseline-warmup-seconds 120` | 需求3：稳态建基线后才开始判定 |
| 终止后自动写 `result.jsonl` + `report.md`（含 9 章节报告） | 需求3：终止即输出报告 |
| `--cache-report` | 采集 prefix cache 命中率（fork 客户端默认已开，flag 仅兼容内置客户端） |
| `--accept-*` 系列参数 | 测试结论汇总 + 验收明细表的要求值（客户指标：吞吐 ≥0.6 req/s、TTFT P50 ≤8s/P95 ≤30s、TPOT P50 ≤30ms/P95 ≤45ms、cache hit ≥60%） |

调整 `--target-tpm`、`--num-sessions`、`--sustain-seconds` 即可适配不同压测目标。`--avg-tokens-per-request` 默认 0=自动从数据集估算每会话总 token 数，无需手动设置。

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
| `--dataset-path` | ShareGPT V3 JSON 路径（可从 .env 读取） |
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
| **测试结论汇总** | 客户 6 项指标（请求吞吐/TTFT P50/P95/TPOT P50/P95/缓存命中率）的要求/实际/结论表 + 总体结论（通过/不通过） |
| **全程** | total_requests/transport_success/success/truncated/actual_tokens/TPM/cache_hit_rate 全局表 + 延迟分位表(avg/p50/p75/p90/p95/p99) |
| **连续稳态** | 是否获得稳态窗口 |
| **验收明细** | 18 项验收项(实际/要求/结论/必过)，含 steady_tpm/success_rate/ttft_p50/ttft_p95/tpot_p50/tpot_p95/cache_hit/zero_429/usage_complete/length_profile |
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

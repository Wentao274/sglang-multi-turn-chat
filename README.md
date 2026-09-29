# sglang-multi-turn-chat

多轮长上下文对话压测框架。复用 `sglang.benchmark.serving` 内置 HTTP 客户端、多轮 wrapper 与指标计算；使用 `ShareGPT_V3_unfiltered_cleaned_split.json` 全多轮数据；TPM 匀速爬坡至目标稳态后持续压测，检测衰减信号自动终止并输出 Markdown 报告。

报告格式对齐客户模板 `reports-temp.md`，输出 9 个章节：全程、连续稳态、验收明细、全程分轮、稳态分轮、stream/full_run、stream/steady、加压时间序列、实际长度与轮数。

## 架构

| 文件 | 职责 |
|---|---|
| `bench_multi_turn.py` | 主入口：编排加载→预热→爬坡调度→采集→监控→出报告，复用 serving.py 零件 |
| `sharegpt_multiturn.py` | ShareGPT V3 → `List[DatasetRow]`，去重保证会话不重复、每会话唯一 system prompt 防缓存命中 |
| `ramp_scheduler.py` | 非齐次泊松请求生成器：RPS 从 `start` 线性爬升到 `target`，稳态后持续压测 |
| `monitor.py` | 衰减监控器 + 时间序列快照采集：滑动窗口检测错误率/吞吐跌落/TTFT 飙升；全程记录 RPM/TPM/延迟快照 |
| `config.py` | 参数解析 + 把本框架参数映射成 serving.py 内置客户端读取的全局 `args` |
| `report.py` | 生成 Markdown 压测报告：9 个章节对齐客户模板 |

被复用的 sglang 内置组件（不重造轮子）：
- `async_request_openai_chat_completions` —— `/v1/chat/completions` 流式 HTTP 客户端（TTFT/ITL 采集）
- `wrap_multi_turn_request_func` —— 每轮把完整历史发给服务端，append assistant 回复，上下文随轮次自然增长
- `calculate_metrics` / `BenchmarkMetrics` —— 与 `python -m sglang.benchmark.serving` 完全一致的指标口径
- `get_tokenizer` / `download_and_cache_hf_file` / `flush_server_cache` / `wait_for_endpoint`

补齐内置工具的两个缺口：
- **缺口 A**：内置 `ShareGPTDataset` 只取前 2 轮（单轮）。本框架读原始 JSON，保留全多轮 user 消息，可选注入长 system prompt。
- **缺口 B**：内置 `get_request` 仅支持单一固定速率泊松/全量并发。本框架 `get_ramp_request` 实现匀速爬坡。

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

# 安装全部依赖（读取 requirements.txt）
uv pip install -r requirements.txt
```

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

**容量与防缓存保证**：加载时按首轮 user 文本做 md5 去重，确保会话内容不重复；`--system-prompt-len > 0` 时每个会话生成**独立随机** system prompt，避免多会话共享前缀导致 prefix cache 持续命中。若去重+过滤后可用会话数 < `--num-sessions`，直接报错并提示调整参数（不靠重复凑数）。`--sustain-seconds` 设定时，框架会预估所需会话数并在不足时给出警告。

## 用法示例

### 推荐命令（.env + 精简 CLI）

配置好 `configs/.env` 后，命令行只需传压测参数：

```bash
# 方式一：激活虚拟环境后直接运行
python bench_multi_turn.py \
  --num-sessions 5000 --num-turns 8 --max-tokens-per-turn 256 \
  --system-prompt-len 2048 \
  --start-tpm 0 --target-tpm 100000000 --avg-tokens-per-request 1024 \
  --ramp-seconds 120 --sustain-seconds 600 \
  --max-concurrency 256 \
  --cache-report \
  --max-error-rate 0.10 \
  --throughput-drop-ratio 0.5 \
  --max-ttft-p99-ms 30000 \
  --baseline-warmup-seconds 30 \
  --monitor-window 30 --monitor-interval 10 \
  --accept-steady-tpm 100000000 \
  --accept-request-rps 0.6 \
  --accept-success-rate 0.995 \
  --accept-ttft-p50-ms 15000 \
  --accept-tpot-p50-ms 35 \
  --accept-cache-hit-rate 0.6 \
  --accept-zero-429 0 \
  --accept-usage-complete 1 \
  --output-file result.jsonl \
  --report-md report.md \
  --output-details \
  --tag glm-multiturn-steady
```

```bash
# 方式二：不激活虚拟环境，用 uv run（自动使用 .venv）
uv run python bench_multi_turn.py --num-sessions 5000 --num-turns 8 ...
```

> `--base-url`、`--model`、`--tokenizer`、`--api-key`、`--dataset-path` 自动从 `configs/.env` 读取，CLI 同名参数可覆盖。

命令逐段对应需求：

| 参数 | 对应需求 |
|---|---|
| `.env` 中 `DATASET_PATH`（去重后不足会报错） | 需求1：ShareGPT 数据集 + 压测容量 |
| `--num-sessions 5000` | 需求1：压测容量 |
| `--system-prompt-len 2048`（每会话唯一随机） | 需求1：防缓存命中 + 长上下文 |
| `--start-tpm 0 --target-tpm 100000000 --ramp-seconds 120` | 需求2：起压点匀速爬坡至目标稳态 |
| `--sustain-seconds 600` | 需求2：稳态后持续压测 |
| 指标自动只统计 `start_time >= 爬坡结束` 的会话 | 需求2：从稳态开始记录指标 |
| `--max-error-rate / --throughput-drop-ratio / --max-ttft-p99-ms` | 需求3：异常/衰减信号终止 |
| `--baseline-warmup-seconds 30` | 需求3：稳态建基线后才开始判定 |
| 终止后自动写 `result.jsonl` + `report.md`（含 9 章节报告） | 需求3：终止即输出报告 |
| `--cache-report` | 采集 prefix cache 命中率 |
| `--accept-*` 系列参数 | 验收明细表的要求值 |

调整 `--target-tpm`、`--num-sessions`、`--sustain-seconds` 即可适配不同压测目标。`--target-tpm` 按 `--avg-tokens-per-request`（默认 512，上例设 1024）折算 RPS。

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
| `--num-turns` | 每会话轮数；上下文随轮次线性增长 |
| `--max-tokens-per-turn` | 每轮生成 max_tokens 上限 |
| `--system-prompt-len` | >0 注入该长度随机 system prompt，首轮即长上下文 |
| `--target-tpm` | 稳态目标 TPM，按 `--avg-tokens-per-request` 折算 RPS |
| `--start-rps`/`--target-rps` | 直接指定 RPS（被 `--target-tpm` 覆盖） |
| `--ramp-seconds` | RPS 线性爬坡时长 |
| `--sustain-seconds` | 稳态持续时长；留空表示发完全部 `--num-sessions` |
| `--ignore-eos` | 忽略 EOS 强制满长（吞吐场景）；默认尊重 EOS 模拟真实对话 |
| `--max-concurrency` | 最大并发会话数 |
| `--cache-report` | 采集 prefix cache 命中统计（sglang 后端） |
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
| `--accept-ttft-p50-ms` | 验收要求：TTFT p50（默认 15000） |
| `--accept-tpot-p50-ms` | 验收要求：TPOT p50（默认 35） |
| `--accept-cache-hit-rate` | 验收要求：cache hit rate（默认 0.6） |
| `--accept-zero-429` | 验收要求：429 次数上限（默认 0） |
| `--accept-usage-complete` | 验收要求：usage 完整（默认 1） |

## 输出报告

报告对齐客户模板 `reports-temp.md`，共 9 个章节：

| 章节 | 内容 |
|---|---|
| **全程** | total_requests/transport_success/success/truncated/actual_tokens/TPM/cache_hit_rate 全局表 + 延迟分位表(avg/p50/p75/p90/p95/p99) |
| **连续稳态** | 是否获得稳态窗口 |
| **验收明细** | 16 项验收项(实际/要求/结论/必过)，含 steady_tpm/success_rate/ttft_p50/tpot_p50/cache_hit/zero_429/usage_complete/length_profile |
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
    multi_turn_sglang-oai-chat_5000s_8t.jsonl
    report.md
```

| 参数 | 说明 |
|---|---|
| `--output-dir` | 结果根目录（默认 `results`） |
| `--output-file` | JSONL 文件名（仅文件名，自动放入子目录；留空自动命名） |
| `--report-md` | Markdown 报告文件名（仅文件名，自动放入子目录；留空默认 `report.md`） |

> 注：多轮模式下输入 token 按累积上下文估算（round-0 prompt + 前序轮次 output_len 之和），反映服务端实际接收的上下文长度。输出 token 逐轮精确统计。

## 烟雾测试

连通性自检可参考 `D:\Maas\客户反馈\多轮对话\glm_multiturn_messages.py`（单会话多轮客户端），先跑通 `/v1/chat/completions` 多轮链路再压测。

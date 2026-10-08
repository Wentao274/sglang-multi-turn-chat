import argparse
import os
from types import SimpleNamespace

from sglang.benchmark.serving import MULTI_TURN_BACKENDS


def load_env(env_file: str) -> dict:
    """Parse a .env file into a dict. Returns empty dict if file not found."""
    values = {}
    if not env_file or not os.path.isfile(env_file):
        return values
    with open(env_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip()
            if (val.startswith('"') and val.endswith('"')) or (
                val.startswith("'") and val.endswith("'")
            ):
                val = val[1:-1]
            values[key] = val
    return values


def _preparse_env_file() -> str:
    """Quick parse to get --env-file before building the full parser."""
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--env-file", default="configs/.env")
    ns, _ = pre.parse_known_args()
    return ns.env_file


def build_parser(env: dict = None):
    env = env or {}

    def _env_default(key, fallback=None):
        return env.get(key, fallback)

    parser = argparse.ArgumentParser(
        description=(
            "多轮长上下文对话压测：复用 sglang.benchmark.serving 内置 HTTP 客户端、"
            "多轮 wrapper 与指标计算；ShareGPT V3 全多轮数据；TPM 匀速爬坡至稳态后持续压测。"
        )
    )

    server = parser.add_argument_group("server")
    server.add_argument("--base-url", type=str,
                        default=_env_default("BASE_URL"),
                        help="推理服务地址，例如 http://127.0.0.1:30000（可从 .env 读取）")
    server.add_argument("--model", type=str, default=_env_default("MODEL"),
                        help="模型名。留空则自动探测 /v1/models（可从 .env 读取）")
    server.add_argument("--api-key", type=str, default=_env_default("API_KEY"),
                        help="API Key，注入 OPENAI_API_KEY 环境变量供 sglang 客户端使用（可从 .env 读取）")
    server.add_argument("--backend", type=str, default=_env_default("BACKEND", "sglang-oai-chat"),
                        choices=sorted(MULTI_TURN_BACKENDS),
                        help="多轮仅支持 chat 后端")
    server.add_argument("--header", type=str, nargs="*", default=None,
                        help="额外请求头，格式 key=value，可多个")
    server.add_argument("--ready-check-timeout-sec", type=int, default=60,
                        help="启动前等待服务就绪的超时秒数，0 表示跳过")
    server.add_argument("--max-concurrency", type=int, default=0,
                        help="最大并发请求数，0 表示不限")

    dataset = parser.add_argument_group("dataset")
    dataset.add_argument("--dataset-path", type=str,
                         default=_env_default("DATASET_PATH"),
                         help="ShareGPT V3 json 路径，留空自动下载（可从 .env 读取）")
    dataset.add_argument("--num-sessions", "--num-prompts", dest="num_sessions",
                         type=int, default=100, help="会话(对话)数量")
    dataset.add_argument("--num-turns", type=int, default=14,
                         help="每个会话的最大对话轮数上限(自然轮数不足此值的会话按实际轮数)")
    dataset.add_argument("--max-tokens-per-turn", type=int, default=256,
                         help="每轮生成 max_tokens 上限")
    dataset.add_argument("--system-prompt-len", type=int, default=0,
                         help=">0 则生成该长度的随机 system prompt 注入首轮，模拟长上下文")
    dataset.add_argument("--context-len", type=int, default=None,
                         help="过滤首轮(system+user)加单轮输出超过该长度的会话")
    dataset.add_argument("--min-turns", type=int, default=2,
                         help="仅保留 user 轮数不少于该值的原始对话(变长轮次的过滤门槛)")
    dataset.add_argument("--num-shared-prefixes", type=int, default=0,
                       help="共享 system prompt 组数；0=每会话唯一前缀（cache hit≈0），>0=按 round-robin 分配共享前缀给会话，产生 60%-100% cache hit")
    dataset.add_argument("--tokenizer", type=str,
                         default=_env_default("TOKENIZER"),
                         help="tokenizer 名，留空则用 --model（可从 .env 读取）")
    dataset.add_argument("--apply-chat-template", action="store_true",
                         help="对每轮 user 文本套用 tokenizer chat template(默认关闭，由服务端模板化)")

    ramp = parser.add_argument_group("ramp")
    ramp.add_argument("--start-rps", type=float, default=0.0, help="爬坡起始 RPS")
    ramp.add_argument("--target-rps", type=float, default=1.0, help="稳态目标 RPS")
    ramp.add_argument("--ramp-seconds", type=float, default=60.0,
                      help="RPS 从 start 线性爬升到 target 的时长(秒)")
    ramp.add_argument("--sustain-seconds", type=float, default=None,
                      help="到达稳态后持续压测时长；留空表示发完全部会话为止")
    ramp.add_argument("--drain-timeout", type=float, default=600.0,
                      help="排空超时(秒)：调度结束后等待在途会话完成的时限，超时强制取消剩余任务，防止过载/挂起场景无限排空")
    ramp.add_argument("--start-tpm", type=float, default=0.0, help="爬坡起始 TPM(优先于 start-rps)")
    ramp.add_argument("--target-tpm", type=float, default=None,
                      help="稳态目标 TPM(优先于 target-rps)，按 avg-tokens-per-request 折算 RPS")
    ramp.add_argument("--avg-tokens-per-request", type=float, default=0.0,
                      help="TPM->RPS 折算用：每会话平均总 token 数；0=自动从数据集估算")

    gen = parser.add_argument_group("generation")
    gen.add_argument("--ignore-eos", action="store_true",
                     help="忽略 EOS，强制每轮生成满 max-tokens(纯吞吐场景)；默认尊重 EOS 模拟真实对话")
    gen.add_argument("--temperature", type=float, default=0.0)
    gen.add_argument("--top-p", type=float, default=1.0)
    gen.add_argument("--reasoning-effort", type=str, default=None,
                     help="推理强度(low/medium/high)，设置后 payload 附带 reasoning_effort 字段；"
                          "glm-5.3 不支持关闭思考，用 low 缩短思考段，降低 TTFT/TPOT 并提高有效输出占比")
    gen.add_argument("--no-session-affinity", action="store_true",
                     help="关闭会话亲和路由：默认每会话生成 routing_key 经 X-SMG-Routing-Key 头"
                          "发给网关，使同一会话各轮落在同一后端节点（prefix cache 命中的前提）；"
                          "多节点无亲和时 round1+ 命中率被随机路由稀释至 ~1/N")
    gen.add_argument("--no-stream", action="store_true", help="关闭流式")
    gen.add_argument("--no-warmup", action="store_true", help="跳过预热")
    gen.add_argument("--no-flush-cache", action="store_true", help="预热后不刷 prefix cache")
    gen.add_argument("--cache-report", action="store_true",
                     help="采集 prefix cache 命中统计(sglang 后端支持)")

    monitor = parser.add_argument_group("monitor")
    monitor.add_argument("--monitor-interval", type=float, default=10.0,
                          help="衰减检测采样周期(秒)")
    monitor.add_argument("--monitor-window", type=float, default=30.0,
                          help="衰减检测滚动窗口(秒)")
    monitor.add_argument("--max-error-rate", type=float, default=0.10,
                          help="窗口错误率超过该阈值即判定衰减，终止测试")
    monitor.add_argument("--throughput-drop-ratio", type=float, default=0.5,
                          help="窗口吞吐低于稳态基线该比例即判定衰减(0.5=掉到一半)")
    monitor.add_argument("--max-ttft-p99-ms", type=float, default=30000.0,
                          help="窗口 TTFT p99 超过该值即判定衰减")
    monitor.add_argument("--baseline-warmup-seconds", type=float, default=30.0,
                          help="进入稳态后用该时长建立吞吐基线，之后才开始衰减判定")
    monitor.add_argument("--disable-monitor", action="store_true",
                          help="关闭衰减监控(跑完全部)")

    accept = parser.add_argument_group("acceptance")
    accept.add_argument("--accept-steady-tpm", type=float, default=None,
                      help="验收要求：稳态 TPM")
    accept.add_argument("--accept-peak-concurrency", type=int, default=None,
                       help="信息项：报告展示该吞吐下实际压到的并发数（不参与通过/不通过判定）")
    accept.add_argument("--accept-request-rps", type=float, default=0.6,
                       help="验收要求：稳态 RPS")
    accept.add_argument("--accept-success-rate", type=float, default=0.995,
                       help="验收要求：成功率")
    accept.add_argument("--accept-ttft-p50-ms", type=float, default=8000.0,
                       help="验收要求：稳态 TTFT p50 (ms)")
    accept.add_argument("--accept-ttft-p95-ms", type=float, default=30000.0,
                       help="验收要求：稳态 TTFT p95 (ms)")
    accept.add_argument("--accept-tpot-p50-ms", type=float, default=30.0,
                       help="验收要求：稳态 TPOT p50 (ms)")
    accept.add_argument("--accept-tpot-p95-ms", type=float, default=45.0,
                       help="验收要求：稳态 TPOT p95 (ms)")
    accept.add_argument("--accept-cache-hit-rate", type=float, default=0.6,
                         help="验收要求：稳态 cache hit rate")
    accept.add_argument("--accept-zero-429", type=int, default=0,
                         help="验收要求：429 错误数上限")
    accept.add_argument("--accept-usage-complete", type=int, default=1,
                         help="验收要求：usage 信息完整(1=必须)")

    out = parser.add_argument_group("output")
    out.add_argument("--output-dir", type=str, default="results",
                      help="结果输出根目录（默认 results），每次执行在其下创建 model-MMDD-HHMMSS 子目录")
    out.add_argument("--output-file", type=str, default=None,
                      help="结果 JSONL 文件名（仅文件名，不含路径，自动放入输出子目录；留空自动命名）")
    out.add_argument("--report-md", type=str, default=None,
                      help="Markdown 报告文件名（仅文件名，不含路径，自动放入输出子目录；留空自动命名）")
    out.add_argument("--output-details", action="store_true", help="写入每轮明细")
    out.add_argument("--tag", type=str, default="", help="结果 tag")

    misc = parser.add_argument_group("misc")
    misc.add_argument("--env-file", type=str, default="configs/.env",
                      help=".env 文件路径，自动读取 BASE_URL/MODEL/TOKENIZER/API_KEY/DATASET_PATH 等")
    misc.add_argument("--seed", type=int, default=42)
    misc.add_argument("--print-requests", action="store_true", help="打印每轮请求/响应(调试)")
    return parser

def compute_rps(args):
    if args.target_tpm is not None:
        avg = max(args.avg_tokens_per_request, 1.0)
        target = args.target_tpm / 60.0 / avg
        start = (args.start_tpm or 0.0) / 60.0 / avg
        return start, target
    return args.start_rps, args.target_rps


def make_serving_namespace(args):
    return SimpleNamespace(
        disable_stream=args.no_stream,
        disable_ignore_eos=not args.ignore_eos,
        print_requests=args.print_requests,
        cache_report=args.cache_report,
        header=args.header,
        return_logprob=False,
        return_routed_experts=False,
        top_logprobs_num=0,
        token_ids_logprob=None,
        logprob_start_len=0,
        # sglang 自带 multi-turn wrapper 丢弃 extra_request_body / routing_key 字段，
        # request_client 从 serving.args 兜底读取 reasoning_effort；
        # no_session_affinity 用于关闭 routing_key 的内容哈希兜底
        # （网关侧强制固定路由时，客户端不再发 X-SMG-Routing-Key 头）
        reasoning_effort=getattr(args, "reasoning_effort", None),
        no_session_affinity=bool(getattr(args, "no_session_affinity", False)),
    )

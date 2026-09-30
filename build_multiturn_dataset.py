# -*- coding: utf-8 -*-
"""合并 ShareGPT + 新数据集为统一多轮会话数据集（ShareGPT 格式 JSON 数组）。

新数据集均为单轮，转换策略：
- false_qa / truthful_qa（UltraFeedback 格式）：每条 instruction 作为一个 user 轮，
  assistant 轮取 overall_score 最高的模型回答；每 N 条串成一个多轮会话
- databricks-dolly-15k（Dolly 格式）：instruction 作为一个 user 轮，带 context 的
  前置拼入（"Context:\n...\n\nInstruction: ..."）；每 N 条串成一个多轮会话
- TM_multi_turn_convs_split_token_wrap（单轮，多轮对话嵌在 gpt 回复文本内）：
  从 gpt 文本解析 **User:** / **AI Assistant:** 标签得到真实多轮会话

输出为 ShareGPT 格式（{"conversations": [{"from": "human"/"gpt", "value": ...}]}），
可直接作为 bench_multi_turn.py 的 --dataset-path。

用法：
  python build_multiturn_dataset.py \
    --input-dir "D:\\Maas\\芯片测试\\测试相关\\测试数据集" \
    --output merged_multiturn.json

  # 仅转换新数据集（不合并 ShareGPT，快速检查）
  python build_multiturn_dataset.py --no-sharegpt --output converted_only.json
"""
import argparse
import hashlib
import json
import os
import random
import re
import sys
import time

DEFAULT_INPUT_DIR = r"D:\Maas\芯片测试\测试相关\测试数据集"
SHAREGPT_NAME = "ShareGPT_V3_unfiltered_cleaned_split.json"
FALSE_QA_NAME = "false_qa.jsonl"
TRUTHFUL_QA_NAME = "truthful_qa.jsonl"
DOLLY_NAME = "databricks-dolly-15k.jsonl"
TM_NAME = "TM_multi_turn_convs_split_token_wrap.jsonl"

# gpt 文本内的多轮标签，如 "**User:**  \nHi..." / "**AI Assistant:**  \n..."
_TM_TURN_RE = re.compile(
    r"\*\*(User|AI Assistant):\*\*\s*(.*?)(?=\*\*(?:User|AI Assistant):\*\*|\Z)",
    re.S,
)


def _qhash(text):
    return hashlib.md5((text or "").strip().encode("utf-8", errors="ignore")).hexdigest()


def read_jsonl(path):
    out = []
    bad = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                bad += 1
    return out, bad


# ---------- false_qa / truthful_qa ----------

def extract_qa_ultrafeedback(records, source):
    """UltraFeedback 格式 -> (question, answer) 列表。"""
    pairs = []
    skipped = 0
    for rec in records:
        q = (rec.get("instruction") or "").strip()
        if not q:
            skipped += 1
            continue
        best_a = ""
        scored = [c for c in (rec.get("completions") or []) if c.get("response")]
        if scored:
            best = max(
                scored,
                key=lambda c: (
                    c.get("overall_score") or 0.0,
                    c.get("fine-grained_score") or 0.0,
                ),
            )
            best_a = (best.get("response") or "").strip()
        pairs.append((q, best_a))
    return pairs, skipped


# ---------- dolly ----------

def extract_qa_dolly(records, source):
    """Dolly 格式 -> (question, answer) 列表；context 前置拼入 user 轮。"""
    pairs = []
    skipped = 0
    for rec in records:
        q = (rec.get("instruction") or "").strip()
        if not q:
            skipped += 1
            continue
        ctx = (rec.get("context") or "").strip()
        if ctx:
            q = f"Context:\n{ctx}\n\nInstruction: {q}"
        pairs.append((q, (rec.get("response") or "").strip()))
    return pairs, skipped


# ---------- TM ----------

def extract_sessions_tm(records):
    """TM 数据集：从 gpt 回复文本解析 **User:**/**AI Assistant:** 真实多轮。

    返回 (会话列表, 统计)。每个会话 = [(user, assistant), ...]。
    """
    sessions = []
    turn_dist = {}
    parse_failed = 0
    too_few = 0
    for rec in records:
        gpt_text = ""
        for msg in rec.get("conversations") or []:
            if msg.get("from") == "gpt" and msg.get("value"):
                gpt_text = msg["value"]
                break
        if not gpt_text:
            parse_failed += 1
            continue
        turns = _TM_TURN_RE.findall(gpt_text)
        # 只保留成对的 User 轮（User 后跟随的 AI Assistant 回复为其答案）
        session = []
        cur_user = None
        for role, text in turns:
            text = text.strip()
            if not text:
                continue
            if role == "User":
                if cur_user is not None:
                    session.append((cur_user, ""))  # 连续 User 无回答，前一轮补空
                cur_user = text
            else:  # AI Assistant
                if cur_user is not None:
                    session.append((cur_user, text))
                    cur_user = None
                # 开头出现的孤立 assistant 轮忽略
        if cur_user is not None:
            session.append((cur_user, ""))

        n_user = len(session)
        if n_user < 2:
            too_few += 1
            continue
        turn_dist[n_user] = turn_dist.get(n_user, 0) + 1
        sessions.append(session)
    stats = {
        "total": len(records),
        "parse_failed": parse_failed,
        "too_few_turns": too_few,
        "usable": len(sessions),
        "turn_dist": dict(sorted(turn_dist.items())),
    }
    return sessions, stats


def session_to_item(pairs, item_id):
    convs = []
    for q, a in pairs:
        convs.append({"from": "human", "value": q})
        convs.append({"from": "gpt", "value": a})
    return {"id": item_id, "conversations": convs}


def compose_sessions(qa_pool, qa_per_session, rng):
    """将 QA 池随机打散后按 N 条切成多轮会话（余数 >= 2 的保留）。"""
    pool = list(qa_pool)
    rng.shuffle(pool)
    sessions = []
    for i in range(0, len(pool), qa_per_session):
        chunk = pool[i : i + qa_per_session]
        if len(chunk) < 2:
            break  # 剩 0/1 条无法组会话
        # (q, a, source) -> (q, a)
        sessions.append([(c[0], c[1]) for c in chunk])
    return sessions


def main():
    parser = argparse.ArgumentParser(
        description="合并 ShareGPT + false_qa/truthful_qa/dolly/TM 为统一多轮数据集"
    )
    parser.add_argument("--input-dir", default=DEFAULT_INPUT_DIR)
    parser.add_argument("--sharegpt", default=None, help="ShareGPT 主数据集路径（默认在 input-dir 自动查找）")
    parser.add_argument("--false-qa", default=None)
    parser.add_argument("--truthful-qa", default=None)
    parser.add_argument("--dolly", default=None)
    parser.add_argument("--tm", default=None)
    parser.add_argument("--qa-per-session", type=int, default=6,
                        help="false_qa/truthful_qa/dolly 每 N 条 QA 串成一个多轮会话（默认 6）")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-sharegpt", action="store_true",
                        help="不合并 ShareGPT，仅输出新数据集转换结果（快速检查）")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    d = args.input_dir

    def locate(explicit, default_name):
        if explicit:
            return explicit
        p = os.path.join(d, default_name)
        return p if os.path.isfile(p) else None

    sharegpt_path = locate(args.sharegpt, SHAREGPT_NAME)
    false_qa_path = locate(args.false_qa, FALSE_QA_NAME)
    truthful_qa_path = locate(args.truthful_qa, TRUTHFUL_QA_NAME)
    dolly_path = locate(args.dolly, DOLLY_NAME)
    tm_path = locate(args.tm, TM_NAME)

    print("[build-multiturn] 输入：")
    print(f"  ShareGPT    : {sharegpt_path or '(未找到)'}")
    for label, p in [("false_qa", false_qa_path), ("truthful_qa", truthful_qa_path),
                     ("dolly", dolly_path), ("TM", tm_path)]:
        print(f"  {label:<11}: {p or '(未找到)'}")

    if not (false_qa_path or truthful_qa_path or dolly_path or tm_path):
        print("[error] 未找到任何新数据集", file=sys.stderr)
        return 1
    if not args.no_sharegpt and not sharegpt_path:
        print("[error] 未找到 ShareGPT 主数据集（或用 --no-sharegpt 仅转换新数据）", file=sys.stderr)
        return 1

    out_path = args.output or os.path.join(
        d, "converted_only.json" if args.no_sharegpt else "merged_multiturn.json"
    )
    rng = random.Random(args.seed)

    t0 = time.time()

    # ---------- 1. 新数据集 QA 提取（去重：同一问题只保留一次） ----------
    qa_pool = []          # (question, answer, source)
    seen_q = {}
    dup_total = 0

    for path, source, extractor in [
        (false_qa_path, "false_qa", extract_qa_ultrafeedback),
        (truthful_qa_path, "truthful_qa", extract_qa_ultrafeedback),
        (dolly_path, "dolly", extract_qa_dolly),
    ]:
        if not path:
            continue
        records, bad = read_jsonl(path)
        pairs, skipped = extractor(records, source)
        kept = 0
        dup = 0
        for q, a in pairs:
            h = _qhash(q)
            if h in seen_q:
                dup += 1
                continue
            seen_q[h] = True
            qa_pool.append((q, a, source))
            kept += 1
        dup_total += dup
        print(f"[{source}] records={len(records):,} 解析失败={bad} 空 instruction={skipped} "
              f"重复问题={dup} 可用={kept:,}")

    # ---------- 2. 组合多轮会话 ----------
    composed = compose_sessions(qa_pool, args.qa_per_session, rng)
    turn_counts = [len(s) for s in composed]
    print(f"[compose] QA 池={len(qa_pool):,} (跨源重复 {dup_total}) "
          f"每会话 {args.qa_per_session} 条 -> 会话数={len(composed):,} "
          f"(轮次 min={min(turn_counts) if turn_counts else 0} "
          f"max={max(turn_counts) if turn_counts else 0})")

    # ---------- 3. TM 解析 ----------
    tm_items = []
    if tm_path:
        records, bad = read_jsonl(tm_path)
        tm_sessions, tm_stats = extract_sessions_tm(records)
        seen_first = {}
        tm_dedup = 0
        for sess in tm_sessions:
            h = _qhash(sess[0][0])
            if h in seen_first:
                tm_dedup += 1
                continue
            seen_first[h] = True
            tm_items.append(session_to_item(sess, f"tm-{len(tm_items):04d}"))
        print(f"[tm] records={tm_stats['total']} 解析失败={tm_stats['parse_failed']} "
              f"不足2轮={tm_stats['too_few_turns']} 可用={tm_stats['usable']} "
              f"首轮去重丢弃={tm_dedup} 保留={len(tm_items)}")
        print(f"[tm] 用户轮分布={tm_stats['turn_dist']}")

    # ---------- 4. 转换为 ShareGPT 格式 ----------
    new_items = []
    for i, sess in enumerate(composed):
        new_items.append(session_to_item(sess, f"composed-{i:05d}"))
    new_items.extend(tm_items)
    print(f"[convert] 新增会话 = 组合 {len(composed):,} + TM {len(tm_items)} = {len(new_items):,}")

    # ---------- 5. 合并 ShareGPT ----------
    if args.no_sharegpt:
        merged = new_items
        print("[merge] --no-sharegpt：仅输出新数据集转换结果")
    else:
        t1 = time.time()
        with open(sharegpt_path, "r", encoding="utf-8") as f:
            merged = json.load(f)
        print(f"[sharegpt] 加载 {len(merged):,} 条 ({time.time() - t1:.1f}s)")
        merged.extend(new_items)
        rng.shuffle(merged)
        print(f"[merge] 合并后 {len(merged):,} 条（已随机 shuffle）")

    # ---------- 6. 写出 ----------
    t2 = time.time()
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False)
    size_mb = os.path.getsize(out_path) / 1024 / 1024
    print(f"[write] {out_path} ({size_mb:.1f} MB, {time.time() - t2:.1f}s)")
    print(f"[done] 总耗时 {time.time() - t0:.1f}s，输出 {len(merged):,} 条会话")
    return 0


if __name__ == "__main__":
    sys.exit(main())

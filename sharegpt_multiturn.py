import hashlib
import json
import random
from typing import List, Optional

import numpy as np
from transformers import PreTrainedTokenizerBase

from sglang.benchmark.datasets.common import (
    DatasetRow,
    SHAREGPT_FILENAME,
    SHAREGPT_REPO_ID,
    gen_prompt,
)
from sglang.benchmark.utils import download_and_cache_hf_file, is_file_valid_json

_ROLE_MAP = {"human": "user", "gpt": "assistant", "user": "user", "assistant": "assistant"}


def _conversations(item):
    return item.get("conversations") or item.get("conversation") or []


def _role_value(msg):
    role = msg.get("from") or msg.get("role")
    value = msg.get("value") if msg.get("value") is not None else msg.get("content")
    return _ROLE_MAP.get(role, role), value


def _resolve_dataset_path(dataset_path: str) -> str:
    if dataset_path and is_file_valid_json(dataset_path):
        return dataset_path
    return download_and_cache_hf_file(
        repo_id=SHAREGPT_REPO_ID, filename=SHAREGPT_FILENAME
    )


def load_sharegpt_multiturn(
    dataset_path: str,
    tokenizer: PreTrainedTokenizerBase,
    num_sessions: int,
    num_turns: int,
    max_tokens_per_turn: int,
    system_prompt_len: int = 0,
    context_len: Optional[int] = None,
    min_turns: int = 2,
    seed: int = 42,
    apply_chat_template: bool = False,
    num_shared_prefixes: int = 0,
) -> List[DatasetRow]:
    random.seed(seed)
    np.random.seed(seed)

    path = _resolve_dataset_path(dataset_path)
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    need = min_turns
    cap = num_turns
    seen_keys = set()
    candidates = []
    for item in raw:
        user_msgs = [
            v
            for r, v in (_role_value(m) for m in _conversations(item))
            if r == "user" and v
        ]
        if len(user_msgs) < need:
            continue
        key = hashlib.md5(user_msgs[0].encode("utf-8", errors="ignore")).hexdigest()
        if key in seen_keys:
            continue
        seen_keys.add(key)
        candidates.append(user_msgs)

    available = len(candidates)
    if available < num_sessions:
        raise ValueError(
            f"数据集容量不足：需要 {num_sessions} 条不重复会话，"
            f"ShareGPT 中满足 >={need} user 轮且去重后仅 {available} 条。"
            f"请减小 --num-sessions 或放宽 --min-turns，"
            f"或提供更大的 --dataset-path。"
        )
    random.shuffle(candidates)

    shared_prompts = []
    if system_prompt_len > 0 and num_shared_prefixes > 0:
        shared_prompts = [gen_prompt(tokenizer, system_prompt_len)
                         for _ in range(num_shared_prefixes)]
        print(f"[sharegpt-multiturn] shared_prefixes={num_shared_prefixes} "
              f"(~{num_sessions // max(num_shared_prefixes, 1)} sessions/prefix)")

    rows: List[DatasetRow] = []
    for user_msgs in candidates:
        if len(rows) >= num_sessions:
            break
        turns = user_msgs[:cap]

        if apply_chat_template:
            turns = [
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": t}],
                    add_generation_prompt=False,
                    tokenize=False,
                )
                for t in turns
            ]

        if system_prompt_len > 0:
            if shared_prompts:
                sys_p = shared_prompts[len(rows) % len(shared_prompts)]
            else:
                sys_p = gen_prompt(tokenizer, system_prompt_len)
            rounds = [
                [{"role": "system", "content": sys_p},
                 {"role": "user", "content": turns[0]}]
            ] + list(turns[1:])
            first_text = sys_p + "\n" + turns[0]
        else:
            rounds = list(turns)
            first_text = turns[0]

        prompt_len = len(tokenizer.encode(first_text))
        if context_len and prompt_len + max_tokens_per_turn > context_len:
            continue

        rows.append(
            DatasetRow(
                prompt=rounds,
                prompt_len=prompt_len,
                output_len=max_tokens_per_turn,
            )
        )

    if len(rows) < num_sessions:
        raise ValueError(
            f"经 context_len 过滤后会话数 {len(rows)} < 需求 {num_sessions}；"
            f"请放宽 --context-len 或减少 --num-sessions。"
        )

    total_in = sum(r.prompt_len for r in rows)
    total_out = sum(r.output_len * len(r.prompt) for r in rows)
    print(
        f"[sharegpt-multiturn] loaded={len(rows)} max_turns={cap} min_turns={need} "
        f"system_prompt_len={system_prompt_len} "
        f"({'shared_prefixes=' + str(num_shared_prefixes) if num_shared_prefixes > 0 else 'per-session unique'}) "
        f"unique_first_turns={available} "
        f"first-turn tokens sum={total_in} est output tokens={total_out}"
    )
    return rows

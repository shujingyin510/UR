# -*- coding: utf-8 -*-
"""R3 生成器:采样解码(temperature / top-p)续写 R1 同一批 prompt。

问题(用户提出):采样是消除退化,还是只把退化移频(L=23 → L=35)?
条件:t08 = temperature 0.8(top_p 1.0);p95 = top-p 0.95(temperature 1.0)。
可复现:逐 prompt 重置随机种子 torch.manual_seed(seed + 序号)。
其余与 r1_gen 完全一致(同 prompt、同 600 token 上限、同 float32、本地缓存)。

用法:python r3_gen.py --model gpt2 --tag t08
输出:data/gen_{model}_{tag}.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import time

from r1_gen import MODELS, PROMPTS

MAX_NEW = 600
CONDS = {
    "t08": {"temperature": 0.8, "top_p": 1.0},
    "p95": {"temperature": 1.0, "top_p": 0.95},
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODELS), required=True)
    ap.add_argument("--tag", choices=list(CONDS), required=True)
    ap.add_argument("--seed", type=int, default=1000)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    name = MODELS[args.model]
    cond = CONDS[args.tag]
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(name, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(name, local_files_only=True, dtype=torch.float32)
    model.eval()
    print(f"[{args.model}/{args.tag}] 加载 {time.time()-t0:.1f}s  {cond}", flush=True)

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    path = os.path.join(out_dir, f"gen_{args.model}_{args.tag}.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for idx, (pid, prompt) in enumerate(PROMPTS):
            torch.manual_seed(args.seed + idx)  # 逐 prompt 定种,可单条复跑
            enc = tok(prompt, return_tensors="pt")
            t1 = time.time()
            with torch.no_grad():
                out = model.generate(
                    **enc,
                    max_new_tokens=MAX_NEW,
                    do_sample=True,
                    temperature=cond["temperature"],
                    top_p=cond["top_p"],
                    pad_token_id=tok.eos_token_id,
                )
            ids = out[0][enc["input_ids"].shape[1]:].tolist()
            stopped = bool(ids and ids[-1] == tok.eos_token_id)
            text = tok.decode(ids, skip_special_tokens=True)
            dt = time.time() - t1
            print(f"[{args.model}/{args.tag}] {pid:14s} {len(ids):4d} tok  {dt:6.1f}s  "
                  f"{len(ids)/max(dt,0.01):5.1f} tok/s  eos={stopped}", flush=True)
            f.write(json.dumps({"pid": pid, "prompt": prompt, "text": text, "ids": ids,
                                "n_new": len(ids), "stopped_eos": stopped,
                                "cond": args.tag, "seed": args.seed + idx},
                               ensure_ascii=False) + "\n")
    print(f"[{args.model}/{args.tag}] 完成 → {path}", flush=True)


if __name__ == "__main__":
    main()

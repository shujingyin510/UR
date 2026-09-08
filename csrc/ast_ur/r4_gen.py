# -*- coding: utf-8 -*-
"""R4 生成器:GPT-2 家族规模阶梯(medium 355M / large 774M),贪心,R1 同 prompt。

目的:H1 普适性(所有 decoder 贪心下都有收尾周期?)+ 波长 vs 模型规模。
零下载:两模型权重已在本地缓存(local_files_only)。参数与 r1_gen 完全一致。

用法:python r4_gen.py --model gpt2m|gpt2l
输出:data/gen_{model}.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import time

from r1_gen import MAX_NEW, PROMPTS

MODELS4 = {
    "gpt2m": "openai-community/gpt2-medium",
    "gpt2l": "openai-community/gpt2-large",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODELS4), required=True)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    name = MODELS4[args.model]
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(name, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(name, local_files_only=True, dtype=torch.float32)
    model.eval()
    print(f"[{args.model}] 加载 {time.time()-t0:.1f}s", flush=True)

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    path = os.path.join(out_dir, f"gen_{args.model}.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for pid, prompt in PROMPTS:
            enc = tok(prompt, return_tensors="pt")
            t1 = time.time()
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=False,
                                     pad_token_id=tok.eos_token_id)
            ids = out[0][enc["input_ids"].shape[1]:].tolist()
            stopped = bool(ids and ids[-1] == tok.eos_token_id)
            text = tok.decode(ids, skip_special_tokens=True)
            dt = time.time() - t1
            print(f"[{args.model}] {pid:14s} {len(ids):4d} tok  {dt:6.1f}s  "
                  f"{len(ids)/max(dt,0.01):5.1f} tok/s  eos={stopped}", flush=True)
            f.write(json.dumps({"pid": pid, "prompt": prompt, "text": text, "ids": ids,
                                "n_new": len(ids), "stopped_eos": stopped},
                               ensure_ascii=False) + "\n")
    print(f"[{args.model}] 完成 → {path}", flush=True)


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""R3 人工复核:关键样本尾部文本。"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
WANT = [
    ("gpt2", "t08", "prime_sieve", 30),   # T=0.8 下逐字 151-token 函数循环?
    ("gpt2", "p95", "dedup", 20),         # 变异 statement 循环(逐字=N)
    ("qwen", "t08", "quicksort", 25),     # 采样下 Qwen 仅存的退化之一
    ("qwen", "p95", "tree_walk", 35),     # 周期+自然EOS:良性同构家族?
    ("qwen", "p95", "merge_sort", 25),    # 同上
]
out = []
for model, tag, pid, nl in WANT:
    path = os.path.join(DATA, f"gen_{model}_{tag}.jsonl")
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        if r["pid"] == pid:
            lines = (r["prompt"] + r["text"]).split("\n")
            out.append(f"===== {model}/{tag}/{pid} 尾部 {nl} 行 (eos={r['stopped_eos']}) =====")
            out.extend(lines[-nl:])
            out.append("")
with open(os.path.join(DATA, "r3_eyeball.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(out) + "\n")
print("written", len(out))

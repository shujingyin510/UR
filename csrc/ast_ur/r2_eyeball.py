# -*- coding: utf-8 -*-
"""R2 人工复核:R1 判健康但 R2 检出周期的 3 条 qwen 尾部 + gpt2 散文行周期。"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
out = []

for line in open(os.path.join(DATA, "gen_qwen.jsonl"), encoding="utf-8"):
    r = json.loads(line)
    if r["pid"] in ("binary_search", "tree_walk", "temp"):
        lines = (r["prompt"] + r["text"]).split("\n")
        out.append(f"===== qwen/{r['pid']} 尾部 40 行 =====")
        out.extend(lines[-40:])
        out.append("")

for line in open(os.path.join(DATA, "gen_gpt2.jsonl"), encoding="utf-8"):
    r = json.loads(line)
    if r["pid"] in ("fibonacci", "quicksort"):
        lines = (r["prompt"] + r["text"]).split("\n")
        out.append(f"===== gpt2/{r['pid']}(散文/低覆盖)尾部 25 行 =====")
        out.extend(lines[-25:])
        out.append("")

with open(os.path.join(DATA, "r2_eyeball.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(out) + "\n")
print("written", len(out))

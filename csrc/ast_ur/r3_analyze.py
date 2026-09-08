# -*- coding: utf-8 -*-
"""R3 分析:采样解码是消除退化还是移频?(贪心基线=R2 的 r2_units.csv)

对 4 个采样语料(gpt2/qwen × t08/p95)复用 R1+R2 全套指标:
  1) H1 逐字→变异:采样下存活退化里逐字复读份额应暴跌(温度踢飞 argmax 循环,骨架吸引子仍在);
  2) H2 波长迁移:周期检出者的 L 分布应右移而非消失(短波最脆、长波抗温);
  3) H3 定律独立复测:tok_min ≈ D/32 与解码无关,采样语料 = R2 的独立分布验证集。

输出:data/r3_units.csv(含贪心基线行)+ data/r3_report.txt。
"""
from __future__ import annotations

import csv
import json
import os
import statistics as st
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from r1_analyze import analyze_seq, verdict  # noqa: E402
from r2_unit import W_TOK, analyze as unit_analyze, pearson  # noqa: E402

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
CONDS = ("t08", "p95")


def num(v):
    try:
        f = float(v)
        return f
    except (TypeError, ValueError):
        return None


def normalize(r: dict) -> dict:
    """R2 CSV 字符串行与本次 Python 值行统一成同一形态。"""
    eos = r.get("eos") in (True, "True")
    kind = r.get("kind") or ""
    L, line_L = num(r.get("L_tok")), num(r.get("line_L"))
    line_p = num(r.get("line_p"))
    anyaxis = bool(kind) or line_p is not None
    return {
        "model": r["model"], "cond": r["cond"], "pid": r["pid"], "eos": eos,
        "coverage": num(r.get("coverage")), "n_stmts": num(r.get("n_stmts")),
        "kind": kind, "p_stmt": num(r.get("p_stmt")), "cycles": num(r.get("cycles")),
        "unit": r.get("unit") or "", "literal": r.get("literal") or "",
        "d_stmt": num(r.get("d_stmt")), "L_tok": L, "D_tok": num(r.get("D_tok")),
        "pred_floor": num(r.get("pred_floor")), "tail_tokmin": num(r.get("tail_tokmin")),
        "tok_min": num(r.get("tok_min")),
        "line_p": line_p, "line_L": line_L,
        "L_eff": L if L is not None else line_L,  # 仅行轴检出者用行波长
        "anyaxis": anyaxis,
        "degen": anyaxis and not eos,  # 收尾周期 ∧ 骑截断(R2 判据,31/31 零误报)
    }


def load_greedy() -> list:
    rows = []
    with open(os.path.join(DATA, "r2_units.csv"), encoding="utf-8") as f:
        for r in csv.DictReader(f):
            r["cond"] = "greedy"
            rows.append(normalize(r))
    return rows


def main() -> None:
    from transformers import AutoTokenizer

    names = {"qwen": "Qwen/Qwen2.5-0.5B", "gpt2": "openai-community/gpt2"}
    rows = load_greedy()
    for model, hf in names.items():  # 串行:一次一个模型
        tok = AutoTokenizer.from_pretrained(hf, local_files_only=True)
        for cond in CONDS:
            path = os.path.join(DATA, f"gen_{model}_{cond}.jsonl")
            if not os.path.exists(path):
                print(f"[skip] {path}", flush=True)
                continue
            with open(path, encoding="utf-8") as f:
                recs = [json.loads(l) for l in f]
            for rec in recs:
                r1row = analyze_seq(rec, tok)
                r1row["verdict"] = verdict(r1row)
                r1row["model"] = model
                row = unit_analyze(rec, tok, r1row)
                row["model"], row["cond"] = model, cond
                n = normalize(row)
                rows.append(n)
                print(f"[{model}/{cond}] {rec['pid']:14s} kind={n['kind'] or '-':6s} "
                      f"unit={n['unit'] or '-':9s} L={n['L_eff'] or '-'} eos={n['eos']}",
                      flush=True)

    cols = ["model", "cond", "pid", "eos", "coverage", "n_stmts", "kind", "p_stmt",
            "cycles", "unit", "literal", "d_stmt", "L_tok", "D_tok", "pred_floor",
            "tail_tokmin", "tok_min", "line_p", "line_L", "L_eff", "anyaxis", "degen"]
    with open(os.path.join(DATA, "r3_units.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore", restval="")
        w.writeheader()
        w.writerows(rows)

    rep = []
    rep.append("== 总览:模型 × 解码条件(n=24 各)==")
    for model in ("gpt2", "qwen"):
        for cond in ("greedy",) + CONDS:
            sub = [r for r in rows if r["model"] == model and r["cond"] == cond]
            if not sub:
                continue
            dg = [r for r in sub if r["degen"]]
            per = [r for r in dg if r["kind"]]
            lit = sum(1 for r in dg if r["literal"] == "Y")
            Ls = sorted(r["L_eff"] for r in dg if r["L_eff"] is not None)
            units = Counter(r["unit"] for r in per if r["unit"])
            rep.append(
                f"[{model}/{cond:6s}] 退化 {len(dg):2d}/24  自然EOS {sum(1 for r in sub if r['eos']):2d}/24  "
                f"逐字 {lit}/{len(dg) or 1}  单元 {dict(units) or '-'}")
            if Ls:
                rep.append(f"           波长 L:中位 {st.median(Ls):.0f}(范围 {Ls[0]:.0f}-{Ls[-1]:.0f}) 全部: {[int(x) for x in Ls]}")

    rep.append("\n== H2 波长迁移(退化序列的 L 分布,贪心 → t08 → p95)==")
    for model in ("gpt2", "qwen"):
        meds = []
        for cond in ("greedy",) + CONDS:
            Ls = [r["L_eff"] for r in rows
                  if r["model"] == model and r["cond"] == cond and r["degen"] and r["L_eff"]]
            meds.append(f"{cond} 中位 {st.median(Ls):.0f}(n={len(Ls)})" if Ls else f"{cond} 无检出")
        rep.append(f"[{model}] " + " → ".join(meds))

    rep.append("\n== H1 逐字→变异(退化序列中逐字复读占比)==")
    for model in ("gpt2", "qwen"):
        parts = []
        for cond in ("greedy",) + CONDS:
            dg = [r for r in rows if r["model"] == model and r["cond"] == cond and r["degen"]]
            lit = sum(1 for r in dg if r["literal"] == "Y")
            parts.append(f"{cond} {lit}/{len(dg)}" if dg else f"{cond} -")
        rep.append(f"[{model}] " + " → ".join(parts))

    rep.append("\n== H3 地板定律独立复测(采样语料,L≤32 的语句轴周期序列)==")
    law = [r for r in rows if r["cond"] in CONDS and r["kind"]
           and r["L_tok"] is not None and r["L_tok"] <= W_TOK
           and r["pred_floor"] is not None and r["tail_tokmin"] is not None]
    for r in law:
        rep.append(f"  {r['model']}/{r['cond']}/{r['pid']:14s} L={r['L_tok']:.0f} D={r['D_tok']:.0f} "
                   f"预测 {r['pred_floor']:.3f} 实测 {r['tail_tokmin']:.3f}")
    if len(law) >= 3:
        xs = [r["pred_floor"] for r in law]
        ys = [r["tail_tokmin"] for r in law]
        rep.append(f"  n={len(law)}  Pearson r={pearson(xs, ys):.3f}  "
                   f"平均绝对误差 {st.mean(abs(x - y) for x, y in zip(xs, ys)):.3f}")
    else:
        rep.append(f"  可比样本仅 {len(law)} 条(采样下短波周期本就该稀少——这本身即 H2 的证据)")

    rep.append("\n== 采样退化明细 ==")
    for r in rows:
        if r["cond"] in CONDS and r["degen"]:
            rep.append(f"  {r['model']}/{r['cond']}/{r['pid']:14s} {r['kind'] or '仅行轴':6s} "
                       f"p={r['p_stmt'] or r['line_p'] or '-'} unit={r['unit'] or '行':9s} "
                       f"逐字={r['literal'] or '-'} L={r['L_eff'] or '-'} D={r['D_tok'] or '-'} cyc={r['cycles'] or '-'}")

    text = "\n".join(rep)
    with open(os.path.join(DATA, "r3_report.txt"), "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print("\n" + text)


if __name__ == "__main__":
    main()

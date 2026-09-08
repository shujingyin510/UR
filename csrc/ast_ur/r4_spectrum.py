# -*- coding: utf-8 -*-
"""R4 周期谱:框架三假设判决 + 模型退化指纹。

H1 普适性:每个 decoder 贪心下都出现收尾周期(4 模型:gpt2 124M / gpt2m 355M /
   gpt2l 774M / qwen 494M——前三个为同家族规模阶梯,顺答「波长 vs 规模」)。
H2 移频不造周期:prompt 级列联(贪心健康∧采样入环 = 凭空创造?)+ 谱支撑集迁移。
H3 盲区判定规则对决:
   规则1(用户原始:只看窗口 vs 周期)token 轴 报警⇔L≤32;语句轴 报警⇔p≤w。
   规则2(R2 地板定律)token 轴 报警⇔L≤32∧D<9.6;语句轴 报警⇔d<w/2。
   在全部周期序列 × 4 探测器上比命中率——「完全由 window vs period 决定」是否成立、
   还是必须带上 D(周期内去重数)。
周期谱:横轴 L(token)分桶,纵轴 P(退化且落桶 | prompt),每模型一张 ASCII 指纹。

输出:data/r4_master.csv + data/r4_report.txt。零 API,单进程串行。
"""
from __future__ import annotations

import bisect
import csv
import hashlib
import json
import os
import statistics as st
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ast_ur import first_below, sliding_ur, tolerant_stmts_typed  # noqa: E402
from r2_unit import (  # noqa: E402
    W_TOK, THR_TOK, classify_unit, periodic_tail, quasi_tail, span_min_ur,
    token_ends, wavelength,
)

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
CORPORA = [
    ("gpt2", "greedy", "gen_gpt2.jsonl", "openai-community/gpt2"),
    ("gpt2m", "greedy", "gen_gpt2m.jsonl", "openai-community/gpt2-medium"),
    ("gpt2l", "greedy", "gen_gpt2l.jsonl", "openai-community/gpt2-large"),
    ("qwen", "greedy", "gen_qwen.jsonl", "Qwen/Qwen2.5-0.5B"),
    ("gpt2", "t08", "gen_gpt2_t08.jsonl", "openai-community/gpt2"),
    ("gpt2", "p95", "gen_gpt2_p95.jsonl", "openai-community/gpt2"),
    ("qwen", "t08", "gen_qwen_t08.jsonl", "Qwen/Qwen2.5-0.5B"),
    ("qwen", "p95", "gen_qwen_p95.jsonl", "Qwen/Qwen2.5-0.5B"),
]
PARAMS = {"gpt2": "124M", "gpt2m": "355M", "gpt2l": "774M", "qwen": "494M"}
BUCKETS = [(1, 8), (9, 16), (17, 32), (33, 64), (65, 128), (129, 999)]


def master_row(rec: dict, tok, model: str, cond: str) -> dict:
    full = rec["prompt"] + rec["text"]
    ids = rec["ids"]
    records, covered, total = tolerant_stmts_typed(full)
    hashes = [h for _, _, h, _ in records]

    line_start = [0]
    for line in full.split("\n"):
        line_start.append(line_start[-1] + len(line) + 1)
    t_ends = token_ends(tok, ids)
    p0 = len(rec["prompt"])

    def char_to_tok(c: int) -> int:
        if c <= p0:
            return 0
        return min(bisect.bisect_left(t_ends, c - p0), len(ids) - 1)

    stmt_tok = [char_to_tok(min(line_start[hi + 1] - 1, len(full))) for _, hi, _, _ in records]
    ur_tok = sliding_ur(ids, W_TOK)
    tok_cross = first_below(ur_tok, THR_TOK, W_TOK - 1)
    ast_abs = {}
    for w in (4, 8, 16):
        if len(hashes) >= w:
            i = first_below(sliding_ur(hashes, w), 0.5, w - 1)
            ast_abs[w] = stmt_tok[i] if i is not None else None
        else:
            ast_abs[w] = None

    out = {
        "model": model, "cond": cond, "pid": rec["pid"],
        "eos": bool(rec.get("stopped_eos")), "n_new": rec["n_new"],
        "coverage": round(covered / total, 3) if total else 0.0, "n_stmts": len(records),
        "tok_cross": tok_cross, "ast4": ast_abs[4], "ast8": ast_abs[8], "ast16": ast_abs[16],
        "kind": "", "p_stmt": None, "cycles": None, "unit": "", "d_stmt": None,
        "L": None, "D": None, "tail_tokmin": None,
        "line_p": None, "line_L": None, "line_D": None, "literal": "",
    }

    pt, kind = periodic_tail(hashes), "strict"
    if pt is None and ast_abs[8] is not None and len(records) >= 10:
        pt, kind = quasi_tail(records), "quasi"
    if pt:
        p, s, e, cyc = pt
        if kind == "strict":
            anchor_idx = [s + k * p for k in range((e - s) // p)]
            cycle = records[e - p : e]
        else:
            anchor_idx = [i for i in range(s, e) if records[i][2] == records[s][2]]
            cycle = records[s : min(s + p, e)]
        L, D = wavelength([stmt_tok[i] for i in anchor_idx], ids)
        out.update(kind=kind, p_stmt=p, cycles=cyc,
                   unit=classify_unit([t for _, _, _, t in cycle]),
                   d_stmt=len(set(h for _, _, h, _ in cycle)), L=L, D=D,
                   tail_tokmin=span_min_ur(ur_tok, stmt_tok[s], stmt_tok[e - 1]))

    line_rec = [(i, hashlib.md5(l.strip().encode()).hexdigest()[:12])
                for i, l in enumerate(full.split("\n")) if l.strip()]
    lp = periodic_tail([h for _, h in line_rec])
    if lp:
        p, s, e, cyc = lp
        anchors = [char_to_tok(line_start[line_rec[s + k * p][0] + 1] - 1)
                   for k in range((e - s) // p)]
        lL, lD = wavelength(anchors, ids)
        out.update(line_p=p, line_L=lL, line_D=lD, literal="Y")
    elif out["kind"]:
        out["literal"] = "N"

    out["L_eff"] = out["L"] if out["L"] is not None else out["line_L"]
    out["D_eff"] = out["D"] if out["D"] is not None else out["line_D"]
    out["anyaxis"] = bool(out["kind"]) or out["line_p"] is not None
    out["degen"] = out["anyaxis"] and not out["eos"]
    return out


def spectrum_ascii(rows: list, label: str) -> list:
    dg = [r for r in rows if r["degen"]]
    Ls = [r["L_eff"] for r in dg if r["L_eff"] is not None]
    n = len(rows)
    lines = [f"[{label}] 退化 {len(dg)}/{n}" + (f"  L 中位 {st.median(Ls):.0f}" if Ls else "")]
    for lo, hi in BUCKETS:
        c = sum(1 for x in Ls if lo <= x <= hi)
        tag = f"{lo:>3}-{hi:<3}" if hi < 999 else f"{lo:>3}+   "
        lines.append(f"  {tag} | {'#' * c:<24} {c:2d} ({100*c/n:3.0f}%)")
    miss = len(dg) - len(Ls)
    if miss:
        lines.append(f"  (另 {miss} 条退化无波长读数)")
    return lines


def main() -> None:
    from transformers import AutoTokenizer

    toks = {}
    rows = []
    for model, cond, fname, hf in CORPORA:  # 串行:逐语料处理
        path = os.path.join(DATA, fname)
        if not os.path.exists(path):
            print(f"[skip] {fname}", flush=True)
            continue
        if hf not in toks:
            toks[hf] = AutoTokenizer.from_pretrained(hf, local_files_only=True)
        with open(path, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                r = master_row(rec, toks[hf], model, cond)
                rows.append(r)
        print(f"[done] {model}/{cond}", flush=True)

    cols = list(rows[0].keys())
    with open(os.path.join(DATA, "r4_master.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore", restval="")
        w.writeheader()
        w.writerows(rows)

    rep = []

    # ---------- H1 普适性 + 规模阶梯 ----------
    rep.append("== H1 普适性:贪心下每个 decoder 都有收尾周期? ==")
    for model in ("gpt2", "gpt2m", "gpt2l", "qwen"):
        sub = [r for r in rows if r["model"] == model and r["cond"] == "greedy"]
        if not sub:
            continue
        dg = [r for r in sub if r["degen"]]
        lits = sum(1 for r in dg if r["literal"] == "Y")
        units = Counter(r["unit"] for r in dg if r["unit"])
        Ls = [r["L_eff"] for r in dg if r["L_eff"] is not None]
        eosn = sum(1 for r in sub if r["eos"])
        rep.append(f"[{model:5s} {PARAMS[model]:>4}] 退化 {len(dg):2d}/{len(sub)}  自然EOS {eosn:2d}"
                   f"  逐字 {lits}/{len(dg) or 1}  L中位 {st.median(Ls):.0f}(范围 {min(Ls):.0f}-{max(Ls):.0f})"
                   f"  单元 {dict(units)}" if Ls else
                   f"[{model:5s} {PARAMS[model]:>4}] 退化 {len(dg):2d}/{len(sub)}  自然EOS {eosn:2d}")

    # ---------- H2 移频不造周期 ----------
    rep.append("\n== H2 采样改周期而非造周期(prompt 级列联,gpt2/qwen)==")
    for model in ("gpt2", "qwen"):
        g = {r["pid"]: r["degen"] for r in rows if r["model"] == model and r["cond"] == "greedy"}
        gL = {r["pid"]: r["L_eff"] for r in rows
              if r["model"] == model and r["cond"] == "greedy" and r["degen"]
              and r["L_eff"] is not None}
        for cond in ("t08", "p95"):
            s = {r["pid"]: r for r in rows if r["model"] == model and r["cond"] == cond}
            if not s:
                continue
            created = [p for p, r in s.items() if r["degen"] and not g.get(p)]
            killed = [p for p in g if g[p] and not s[p]["degen"]]
            kept = [p for p in g if g[p] and s[p]["degen"]]
            beyond = [f"{p}:{s[p]['L_eff']:.0f}" for p in kept + created
                      if s[p]["L_eff"] is not None and gL and s[p]["L_eff"] > max(gL.values())]
            rep.append(f"[{model}/{cond}] 保留 {len(kept)}  杀灭 {len(killed)}  "
                       f"凭空创造 {len(created)}{'(' + ','.join(created) + ')' if created else ''}"
                       f"  超出贪心谱支撑集的 L: {beyond or '无'}")

    # ---------- H3 规则对决 ----------
    rep.append("\n== H3 盲区判定:规则1(只看窗口vs周期) vs 规则2(地板定律含 D)==")
    per = [r for r in rows if r["kind"]]
    # token 轴(L/D 用语句轴读数;需两者齐全)
    tk = [r for r in per if r["L"] is not None and r["D"] is not None]
    r1_ok = sum(1 for r in tk if (r["L"] <= W_TOK) == (r["tok_cross"] is not None))
    r2_ok = sum(1 for r in tk if (r["L"] <= W_TOK and r["D"] < THR_TOK * W_TOK) == (r["tok_cross"] is not None))
    fail_r1 = [r for r in tk if r["L"] <= W_TOK and r["tok_cross"] is None]
    rep.append(f"token 轴(W=32,n={len(tk)}):规则1 命中 {r1_ok}/{len(tk)}  规则2 命中 {r2_ok}/{len(tk)}")
    rep.append(f"  规则1 的失败主体:L≤32 却不报警 {len(fail_r1)} 条(D 中位 "
               f"{st.median([r['D'] for r in fail_r1]):.0f}——周期短但内容多样,必须看 D)" if fail_r1 else "")
    for w in (4, 8, 16):
        el = [r for r in per if r["n_stmts"] >= w and r["p_stmt"] is not None and r["d_stmt"] is not None]
        a1 = sum(1 for r in el if (r["p_stmt"] <= w) == (r[f"ast{w}"] is not None))
        a2 = sum(1 for r in el if (r["d_stmt"] < w / 2) == (r[f"ast{w}"] is not None))
        rep.append(f"语句轴 w={w:2d}(n={len(el)}):规则1 命中 {a1}/{len(el)}  规则2 命中 {a2}/{len(el)}")

    # ---------- 周期谱 ----------
    rep.append("\n== 周期谱:P(退化且波长落桶) per prompt,横轴 L(token)==")
    for model in ("gpt2", "gpt2m", "gpt2l", "qwen"):
        sub = [r for r in rows if r["model"] == model and r["cond"] == "greedy"]
        if sub:
            rep.extend(spectrum_ascii(sub, f"{model} {PARAMS[model]} greedy"))
    for model, cond in (("gpt2", "t08"), ("gpt2", "p95"), ("qwen", "t08"), ("qwen", "p95")):
        sub = [r for r in rows if r["model"] == model and r["cond"] == cond]
        if sub:
            rep.extend(spectrum_ascii(sub, f"{model} {cond}"))

    text = "\n".join(rep)
    with open(os.path.join(DATA, "r4_report.txt"), "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print("\n" + text)


if __name__ == "__main__":
    main()

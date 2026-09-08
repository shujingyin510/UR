# -*- coding: utf-8 -*-
"""R2 重复单元检测:量化代码退化的「波长」与最小重复单元。

R1 证明退化多为长周期结构复读;本实验直接测「复读的单元是什么、有多长」:
  1) 语句轴:骨架哈希序列的最小收尾周期 p(≥2 个完整周期、覆盖到序列末尾),
     单元类型 = 周期内节点类型:含 ClassDef→class;含 FunctionDef→function;
     多语句/含复合语句→block;单条简单语句→statement。
     无严格周期但 R1 判 AST 崩的,用「最频骨架的相邻间距中位」做准周期兜底(quasi)。
  2) 行轴:strip 后行哈希的最小收尾周期 = 逐字复读(对散文/不可解析文本同样有效)。
     语句轴周期成立而行轴不成立 = 「变异复读」(改名/变 docstring 的同构再生成)。
  3) 波长:单元一个周期的 token 长 L、单元去重 token 数 D(周期锚点间实测中位)。
  4) 统一定律检验(H2/H3):窗口盖住整周期时 UR 地板 ≈ 周期去重元素数/窗宽。
     token 轴:尾部 tok_min ≈ D/32,0.30 阈报警 ⇔ D < 9.6 且 L ≲ 32;
     语句轴:w∈{4,8,16} 绝对阈 0.5 报警 ⇔ 周期去重语句数 d < w/2。

零 API:只读 data/gen_*.jsonl(R1 语料)+ 本地 tokenizer(不加载模型权重),单进程串行。
输出:data/r2_units.csv + data/r2_report.txt。
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
from ast_ur import sliding_ur, tolerant_stmts_typed  # noqa: E402

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
W_TOK, THR_TOK = 32, 0.30
COMPOUND = {"If", "While", "For", "AsyncFor", "Try", "TryStar", "With", "AsyncWith", "Match"}


# ---------- 周期检测 ----------

def periodic_tail(seq: list, min_len: int = 6, end_slack: int = 3):
    """最小周期的收尾重复区 → (p, start, end, cycles) 或 None。

    要求 ≥2 个完整周期且重复区 ≥ min_len 个元素;允许序列末 end_slack 个元素游离
    (600 token 截断常砍在周期中间,残尾不该毁掉判定)。p 从小到大,首个可行即最小周期。
    """
    n = len(seq)
    for p in range(1, n // 2 + 1):
        for slack in range(end_slack + 1):
            end = n - slack
            if end < 2 * p:
                break
            i = end - 1
            while i - p >= 0 and seq[i] == seq[i - p]:
                i -= 1
            run = (end - 1) - i
            tail = run + p
            if run >= p and tail >= min_len:
                return p, end - tail, end, round(tail / p, 1)
    return None


def quasi_tail(records: list, min_count: int = 3):
    """严格周期不成立时的兜底:尾部 60% 区域最频骨架的相邻间距中位 → 准周期。

    只对 R1 已判 AST 崩的序列启用(避免在健康序列上瞎找规律)。
    返回 (p, start, end, cycles=出现次数) 或 None。
    """
    n = len(records)
    lo = max(0, int(n * 0.4))
    cnt = Counter(h for _, _, h, _ in records[lo:])
    if not cnt:
        return None
    h_star, c = cnt.most_common(1)[0]
    if c < min_count:
        return None
    occ = [i for i in range(lo, n) if records[i][2] == h_star]
    gaps = [b - a for a, b in zip(occ, occ[1:])]
    p = max(int(st.median(gaps)), 1)
    return p, occ[0], occ[-1] + 1, float(len(occ))


def classify_unit(types: list) -> str:
    s = set(types)
    if "ClassDef" in s:
        return "class"
    if "FunctionDef" in s or "AsyncFunctionDef" in s:
        return "function"
    if len(types) > 1 or (s & COMPOUND):
        return "block"
    return "statement"


# ---------- token 对齐与波长 ----------

def token_ends(tok, ids: list) -> list:
    """生成 token i(0 基)结束时的已解码字符长度(相对生成段起点)。"""
    ends, buf = [], []
    for i in range(len(ids)):
        buf.append(ids[i])
        ends.append(len(tok.decode(buf, skip_special_tokens=True)))
    return ends


def wavelength(anchors: list, ids: list):
    """周期锚点(每周期起点语句的结束 token 下标)→ (L 中位, D 中位)。

    L = 相邻锚点 token 距离;D = 该距离内去重 token 数。锚点须在生成段内且严格递增。
    """
    pairs = [(a, b) for a, b in zip(anchors, anchors[1:]) if 0 < a < b]
    if not pairs:
        # 边界:周期从文本开头(prompt 内)起时首锚点=0 被滤掉——放宽为 a>=0
        # (rle 案例:整篇从第 0 行循环,唯一锚对是 (0, 205))
        pairs = [(a, b) for a, b in zip(anchors, anchors[1:]) if 0 <= a < b]
    if not pairs:
        return None, None
    Ls = [b - a for a, b in pairs]
    Ds = [len(set(ids[a + 1 : b + 1])) for a, b in pairs]
    return int(st.median(Ls)), int(st.median(Ds))


def span_min_ur(ur: list, t_a: int, t_b: int, w: int = W_TOK):
    """完整落在 [t_a, t_b] token 区间内的窗口的 UR 最小值(尾部实测地板)。"""
    vals = [ur[j] for j in range(max(t_a, 0), t_b - w + 2) if 0 <= j < len(ur)]
    return round(min(vals), 3) if vals else None


# ---------- 逐序列分析 ----------

def analyze(rec: dict, tok, r1row: dict) -> dict:
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

    out = {
        "model": r1row.get("model", ""),
        "pid": rec["pid"],
        "r1_verdict": r1row.get("verdict", ""),
        "eos": rec.get("stopped_eos", False),
        "coverage": round(covered / total, 3) if total else 0.0,
        "n_stmts": len(records),
        "kind": "", "p_stmt": "", "cycles": "", "unit": "", "cycle_types": "",
        "d_stmt": "", "L_tok": "", "D_tok": "", "pred_floor": "", "tail_tokmin": "",
        "tok_min": r1row.get("tok_min", ""),
        "pred_tok_alarm": "", "act_tok_alarm": "Y" if r1row.get("tok_cross") else "N",
        "line_p": "", "line_cycles": "", "line_L": "", "literal": "",
    }
    for w in (4, 8, 16):
        out[f"pred_ast{w}"] = ""
        out[f"act_ast{w}"] = "Y" if r1row.get(f"ast{w}_abs") else "N"

    # --- 语句轴:严格周期 → 准周期兜底 ---
    pt, kind = periodic_tail(hashes), "strict"
    if pt is None and r1row.get("ast8_abs") and len(records) >= 10:
        pt, kind = quasi_tail(records), "quasi"
    if pt:
        p, s, e, cyc = pt
        if kind == "strict":
            n_full = (e - s) // p
            anchor_idx = [s + k * p for k in range(n_full)]
            cycle = records[e - p : e]
        else:
            anchor_idx = [i for i in range(s, e) if records[i][2] == records[s][2]]
            cycle = records[s : min(s + p, e)]
        L, D = wavelength([stmt_tok[i] for i in anchor_idx], ids)
        d = len(set(h for _, _, h, _ in cycle))
        out.update(
            kind=kind, p_stmt=p, cycles=cyc,
            unit=classify_unit([t for _, _, _, t in cycle]),
            cycle_types="+".join(f"{t}x{c}" if c > 1 else t
                                 for t, c in Counter(t for _, _, _, t in cycle).most_common()),
            d_stmt=d,
            tail_tokmin=span_min_ur(ur_tok, stmt_tok[s], stmt_tok[e - 1]) or "",
        )
        if L is not None:
            pred = round(min(D, W_TOK) / W_TOK, 3) if L <= W_TOK else ""
            out.update(
                L_tok=L, D_tok=D, pred_floor=pred,
                pred_tok_alarm="Y" if (L <= W_TOK and D < THR_TOK * W_TOK) else "N",
            )
        for w in (4, 8, 16):
            out[f"pred_ast{w}"] = "Y" if d < w / 2 else "N"

    # --- 行轴:逐字复读(parser-free,散文也适用) ---
    line_rec = [(i, hashlib.md5(l.strip().encode()).hexdigest()[:12])
                for i, l in enumerate(full.split("\n")) if l.strip()]
    lp = periodic_tail([h for _, h in line_rec])
    if lp:
        p, s, e, cyc = lp
        n_full = (e - s) // p
        anchors = [char_to_tok(line_start[line_rec[s + k * p][0] + 1] - 1) for k in range(n_full)]
        L, _ = wavelength(anchors, ids)
        out.update(line_p=p, line_cycles=cyc, line_L=L if L is not None else "", literal="Y")
    elif out["kind"]:
        out["literal"] = "N"  # 结构周期在、逐字周期不在 = 变异复读
    return out


# ---------- 汇总报告 ----------

def pearson(xs: list, ys: list) -> float:
    mx, my = st.mean(xs), st.mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = (sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys)) ** 0.5
    return num / den if den else float("nan")


def main() -> None:
    from transformers import AutoTokenizer

    r1 = {}
    with open(os.path.join(DATA, "r1_seqs.csv"), encoding="utf-8") as f:
        for row in csv.DictReader(f):
            r1[(row["model"], row["pid"])] = row

    names = {"qwen": "Qwen/Qwen2.5-0.5B", "gpt2": "openai-community/gpt2"}
    rows = []
    for model, hf in names.items():  # 串行:一次只处理一个模型
        tok = AutoTokenizer.from_pretrained(hf, local_files_only=True)
        with open(os.path.join(DATA, f"gen_{model}.jsonl"), encoding="utf-8") as f:
            recs = [json.loads(l) for l in f]
        for rec in recs:
            row = analyze(rec, tok, r1.get((model, rec["pid"]), {"model": model}))
            row["model"] = model
            rows.append(row)
            print(f"[{model}] {rec['pid']:14s} kind={row['kind'] or '-':6s} "
                  f"p={row['p_stmt'] or '-':>3} unit={row['unit'] or '-':9s} "
                  f"L={row['L_tok'] or '-':>4} D={row['D_tok'] or '-':>3}", flush=True)

    cols = list(rows[0].keys())
    with open(os.path.join(DATA, "r2_units.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore", restval="")
        w.writeheader()
        w.writerows(rows)

    rep = []
    degen = [r for r in rows if r["kind"]]
    rep.append(f"周期检出:{len(degen)}/{len(rows)}(strict {sum(1 for r in degen if r['kind']=='strict')} / quasi {sum(1 for r in degen if r['kind']=='quasi')})")
    healthy_with_period = [r for r in degen if r["r1_verdict"] == "都健康"]
    rep.append(f"R1 判「都健康」却有收尾周期的:{len(healthy_with_period)} 条 "
               f"({', '.join(r['model']+'/'+r['pid'] for r in healthy_with_period) or '无'})")

    rep.append("\n== H1 分类学:重复单元类型 × 模型(仅周期检出者)==")
    for model in ("qwen", "gpt2"):
        sub = [r for r in degen if r["model"] == model]
        cnt = Counter(r["unit"] for r in sub)
        lit = Counter(r["literal"] for r in sub)
        rep.append(f"[{model}] n={len(sub)}  单元: {dict(cnt)}  逐字Y/变异N: {dict(lit)}")
        Ls = [r["L_tok"] for r in sub if r["L_tok"] != ""]
        Ds = [r["D_tok"] for r in sub if r["D_tok"] != ""]
        if Ls:
            rep.append(f"     波长 L 中位 {st.median(Ls)} tok(范围 {min(Ls)}-{max(Ls)})  D 中位 {st.median(Ds)}(范围 {min(Ds)}-{max(Ds)})")

    rep.append("\n== H2 token 轴定律:tok 尾部地板 ≈ D/32(仅 L≤32 的周期序列)==")
    law = [r for r in degen if r["L_tok"] != "" and r["L_tok"] <= W_TOK
           and r["pred_floor"] != "" and r["tail_tokmin"] != ""]
    for r in law:
        rep.append(f"  {r['model']}/{r['pid']:14s} L={r['L_tok']:3d} D={r['D_tok']:3d} "
                   f"预测地板 {r['pred_floor']:.3f}  实测尾部 tok_min {r['tail_tokmin']:.3f}")
    if len(law) >= 3:
        xs = [r["pred_floor"] for r in law]
        ys = [r["tail_tokmin"] for r in law]
        err = [abs(x - y) for x, y in zip(xs, ys)]
        rep.append(f"  n={len(law)}  Pearson r={pearson(xs, ys):.3f}  平均绝对误差 {st.mean(err):.3f}")
    big_L = [r for r in degen if r["L_tok"] != "" and r["L_tok"] > W_TOK]
    blind_ok = [r for r in big_L if r["act_tok_alarm"] == "N"]
    rep.append(f"  L>32 的周期序列 {len(big_L)} 条,其中 token-UR 全程未报警 {len(blind_ok)} 条(预测=全部失明)")

    rep.append("\n== H2 报警预测混淆(周期序列上 预测 D<9.6&L<=32 vs R1 实际 tok_cross)==")
    conf = Counter((r["pred_tok_alarm"], r["act_tok_alarm"]) for r in degen if r["pred_tok_alarm"] != "")
    rep.append(f"  预测Y实际Y {conf.get(('Y','Y'),0)}  预测N实际N {conf.get(('N','N'),0)}  "
               f"预测Y实际N {conf.get(('Y','N'),0)}  预测N实际Y {conf.get(('N','Y'),0)}")

    rep.append("\n== H3 语句轴定律:ast_w 报警 ⇔ 周期去重语句数 d < w/2 ==")
    for w in (4, 8, 16):
        c = Counter((r[f"pred_ast{w}"], r[f"act_ast{w}"]) for r in degen if r[f"pred_ast{w}"] != "")
        tp, tn = c.get(("Y", "Y"), 0), c.get(("N", "N"), 0)
        fp, fn = c.get(("Y", "N"), 0), c.get(("N", "Y"), 0)
        rep.append(f"  w={w:2d}: 对 {tp+tn}/{tp+tn+fp+fn}(预测报警且报 {tp},预测不报且不报 {tn},错报预测 {fp},漏报预测 {fn})")

    rep.append("\n== 逐序列明细(周期检出者)==")
    for r in sorted(degen, key=lambda r: (r["model"], -(r["L_tok"] or 0))):
        rep.append(f"  {r['model']}/{r['pid']:14s} {r['kind']:6s} p={r['p_stmt']:>3} cyc={r['cycles']:>5} "
                   f"unit={r['unit']:9s} d={r['d_stmt']:>2} L={r['L_tok'] if r['L_tok']!='' else '-':>4} "
                   f"D={r['D_tok'] if r['D_tok']!='' else '-':>3} 逐字={r['literal'] or '-'} R1={r['r1_verdict']}")

    text = "\n".join(rep)
    with open(os.path.join(DATA, "r2_report.txt"), "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print("\n" + text)


if __name__ == "__main__":
    main()

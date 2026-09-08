# -*- coding: utf-8 -*-
"""R1 分析:同一生成序列上对比 token-UR 与 AST-UR 的先后崩塌。

对每条序列:
  token-UR:窗 32 token,阈值 0.30(UR 原版标定)。
  AST-UR:窗 {4,8,16} 条语句;绝对阈值 {0.5,0.3} + 相对阈值(< 0.5×自身前 4 窗基线)。
  对齐:语句结束字符 → 生成 token 下标(前缀增量解码长度),两曲线共用 token 横轴。

逐序列裁决(主口径 w8/abs0.5):AST 先崩 / token 先崩 / 只 AST 崩 / 只 token 崩 / 都健康。
输出:data/r1_report.txt(汇总+样例片段)、data/r1_seqs.csv(逐序列指标)。
"""
from __future__ import annotations

import csv
import json
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ast_ur import first_below, sliding_ur, tolerant_stmts  # noqa: E402

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
W_TOK, THR_TOK = 32, 0.30
W_AST_MAIN = 8
THR_AST_ABS = 0.5


def token_ends(tok, ids: list) -> list:
    """生成 token i(0 基)结束时的已解码字符长度(相对生成段起点)。"""
    ends, buf = [], []
    for i in range(len(ids)):
        buf.append(ids[i])
        # 增量近似:每 1 个 token 全量解码前缀,n≤600 可承受
        ends.append(len(tok.decode(buf, skip_special_tokens=True)))
    return ends


def analyze_seq(rec: dict, tok) -> dict:
    prompt, text, ids = rec["prompt"], rec["text"], rec["ids"]
    full = prompt + text
    stmts, covered, total = tolerant_stmts(full)
    coverage = covered / total if total else 0.0

    # 行首字符偏移表 → 语句结束字符
    line_start = [0]
    for line in full.split("\n"):
        line_start.append(line_start[-1] + len(line) + 1)
    ends_char = [min(line_start[hi + 1] - 1, len(full)) for _, hi, _ in stmts]
    hashes = [h for _, _, h in stmts]

    # 语句 → 生成 token 下标(prompt 内语句记 0)
    t_ends = token_ends(tok, ids)
    p0 = len(prompt)
    stmt_tok = []
    for c in ends_char:
        if c <= p0:
            stmt_tok.append(0)
            continue
        rel = c - p0
        idx = next((i for i, e in enumerate(t_ends) if e >= rel), len(ids) - 1)
        stmt_tok.append(idx)

    # token-UR
    ur_tok = sliding_ur(ids, W_TOK)
    tok_cross = first_below(ur_tok, THR_TOK, W_TOK - 1)  # token 下标
    tok_min = min(ur_tok) if ur_tok else None

    out = {
        "pid": rec["pid"],
        "n_new": rec["n_new"],
        "eos": rec.get("stopped_eos", False),
        "coverage": round(coverage, 3),
        "n_stmts": len(stmts),
        "tok_min": round(tok_min, 3) if tok_min is not None else None,
        "tok_cross": tok_cross,
    }

    for w in (4, 8, 16):
        if len(hashes) < w:
            out[f"ast{w}_min"] = None
            out[f"ast{w}_abs"] = None
            out[f"ast{w}_rel"] = None
            continue
        ur_ast = sliding_ur(hashes, w)
        out[f"ast{w}_min"] = round(min(ur_ast), 3)
        i_abs = first_below(ur_ast, THR_AST_ABS, w - 1)  # 语句下标
        out[f"ast{w}_abs"] = stmt_tok[i_abs] if i_abs is not None else None
        base = st.mean(ur_ast[: min(4, len(ur_ast))])
        i_rel = first_below(ur_ast, 0.5 * base, w - 1) if base > 0 else None
        out[f"ast{w}_rel"] = stmt_tok[i_rel] if i_rel is not None else None
        if w == W_AST_MAIN:
            out["_ur_ast_main"] = ur_ast
            out["_stmt_tok"] = stmt_tok
            out["_ur_tok"] = ur_tok
            # AST 主口径崩点时 token-UR 还剩多少(结构崩、词法健康的直接证据)
            if i_abs is not None:
                t_at = stmt_tok[i_abs]
                j = max(0, min(t_at - (W_TOK - 1), len(ur_tok) - 1))
                out["tokUR_at_ast_cross"] = round(ur_tok[j], 3) if ur_tok else None
            else:
                out["tokUR_at_ast_cross"] = None
    return out


def verdict(row: dict) -> str:
    a, t = row.get(f"ast{W_AST_MAIN}_abs"), row.get("tok_cross")
    if a is None and t is None:
        return "都健康"
    if a is not None and t is None:
        return "只AST崩"
    if a is None and t is not None:
        return "只token崩"
    return "AST先崩" if a < t else ("token先崩" if t < a else "同点")


def snippet_at(rec: dict, row: dict) -> str:
    """主口径 AST 崩点附近的源码片段(重复区展示)。"""
    a_tok = row.get(f"ast{W_AST_MAIN}_abs")
    if a_tok is None:
        return ""
    full = rec["prompt"] + rec["text"]
    stmts, _, _ = tolerant_stmts(full)
    lines = full.split("\n")
    idx = [i for i, (_, _, _) in enumerate(stmts) if row["_stmt_tok"][i] <= a_tok]
    if not idx:
        return ""
    i0 = max(0, idx[-1] - W_AST_MAIN + 1)
    lo = stmts[i0][0]
    hi = stmts[idx[-1]][1]
    return "\n".join(lines[max(lo, 0) : min(hi + 1, len(lines))])


def main() -> None:
    from transformers import AutoTokenizer

    names = {"qwen": "Qwen/Qwen2.5-0.5B", "gpt2": "openai-community/gpt2"}
    report = []
    all_rows = []
    for model, hf in names.items():
        path = os.path.join(DATA, f"gen_{model}.jsonl")
        if not os.path.exists(path):
            report.append(f"[{model}] 语料缺失,跳过")
            continue
        tok = AutoTokenizer.from_pretrained(hf, local_files_only=True)
        recs = [json.loads(l) for l in open(path, encoding="utf-8")]
        rows = []
        for rec in recs:
            row = analyze_seq(rec, tok)
            row["model"] = model
            row["verdict"] = verdict(row)
            row["_rec"] = rec
            rows.append(row)
        all_rows.extend(rows)

        report.append(f"\n===== {model}(n={len(rows)}) =====")
        cov = [r["coverage"] for r in rows]
        report.append(f"解析覆盖率 中位 {st.median(cov):.2f}(<0.4 噪声 {sum(1 for c in cov if c < 0.4)} 条)")
        vc = {}
        for r in rows:
            vc[r["verdict"]] = vc.get(r["verdict"], 0) + 1
        report.append(f"裁决(w{W_AST_MAIN}/abs{THR_AST_ABS} vs tok{THR_TOK}): {vc}")
        both = [r for r in rows if r["verdict"] in ("AST先崩", "token先崩", "同点")]
        if both:
            leads = [r["tok_cross"] - r[f"ast{W_AST_MAIN}_abs"] for r in both]
            report.append(
                f"双崩 {len(both)} 条:AST 领先中位 {st.median(leads):.0f} token(正=AST早)  范围 {min(leads)}~{max(leads)}"
            )
        only_ast = [r for r in rows if r["verdict"] == "只AST崩"]
        healthy_tok = [
            r
            for r in rows
            if r.get("tokUR_at_ast_cross") is not None and r["tokUR_at_ast_cross"] >= THR_TOK
        ]
        report.append(
            f"「结构已崩、词法仍健康」:AST 崩点处 tokUR≥{THR_TOK} 的 {len(healthy_tok)}/{sum(1 for r in rows if r.get(f'ast{W_AST_MAIN}_abs') is not None)} 条;只 AST 崩 {len(only_ast)} 条"
        )
        # 稳健性:三窗口一致性
        for w in (4, 8, 16):
            n_cross = sum(1 for r in rows if r.get(f"ast{w}_abs") is not None)
            report.append(f"  w={w:2d}: AST 绝对崩 {n_cross} 条  相对崩 {sum(1 for r in rows if r.get(f'ast{w}_rel') is not None)} 条")
        # 样例:AST 领先最大且崩点处 token-UR 健康
        cand = sorted(
            (r for r in rows if r["verdict"] in ("AST先崩", "只AST崩") and (r.get("tokUR_at_ast_cross") or 0) >= THR_TOK),
            key=lambda r: -(r["tok_cross"] - r[f"ast{W_AST_MAIN}_abs"]) if r["tok_cross"] else -10**9,
        )
        if cand:
            ex = cand[0]
            report.append(f"\n样例 {ex['pid']}(AST崩@tok{ex[f'ast{W_AST_MAIN}_abs']} tokUR彼时{ex['tokUR_at_ast_cross']} tok崩@{ex['tok_cross']}):")
            report.append(snippet_at(ex["_rec"], ex) or "(片段提取失败)")

    with open(os.path.join(DATA, "r1_seqs.csv"), "w", newline="", encoding="utf-8") as f:
        cols = [
            "model", "pid", "n_new", "eos", "coverage", "n_stmts", "tok_min", "tok_cross",
            "ast4_min", "ast4_abs", "ast4_rel", "ast8_min", "ast8_abs", "ast8_rel",
            "ast16_min", "ast16_abs", "ast16_rel", "tokUR_at_ast_cross", "verdict",
        ]
        wcsv = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore", restval="")
        wcsv.writeheader()
        for r in all_rows:
            wcsv.writerow(r)

    text = "\n".join(report)
    with open(os.path.join(DATA, "r1_report.txt"), "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()

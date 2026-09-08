# -*- coding: utf-8 -*-
"""AST-UR:代码生成的结构级重复率指标。

假设:代码退化时 AST 结构重复(token 不同、语句同构)早于 token 级重复。
  token-UR(UR 原版):滑动窗 32 token 的 unique token ratio,阈值 0.30。
  AST-UR(本实验新增):语句 → 骨架哈希(标识符/常量归一化),滑动窗内 unique 骨架比。

骨架归一化:Name/arg/attr 等标识符 → 'N';Constant 值 → 其类型名;保留结构与运算符。
  于是 `result.append(item)` 与 `out.append(x)` 同构,`a+b` 与 `a-b` 不同构。

容错解析:生成文本按「最长可解析前缀 + 跳过报错行」切块;悬空缩进块用 `if 1:` 包裹。
  解析覆盖率(进入 AST 的行占比)随结果一起报告,<40% 的序列标为噪声。
"""
from __future__ import annotations

import ast
import hashlib

_ID_FIELDS = {"id", "arg", "attr", "name", "asname", "module"}


def skeleton(node: ast.AST) -> str:
    """AST 节点 → 归一化骨架 s-expr(标识符→N,常量→类型名)。"""
    parts = [type(node).__name__]
    for field, value in ast.iter_fields(node):
        if field in _ID_FIELDS:
            parts.append("N")
        elif isinstance(node, ast.Constant) and field == "value":
            parts.append(type(value).__name__)
        elif isinstance(value, ast.AST):
            parts.append(skeleton(value))
        elif isinstance(value, list):
            inner = ",".join(skeleton(x) for x in value if isinstance(x, ast.AST))
            parts.append("[" + inner + "]")
        # 其余标量字段(ctx 已是 AST、行号被 iter_fields 排除)忽略
    return "(" + " ".join(parts) + ")"


def skel_hash(node: ast.AST) -> str:
    return hashlib.md5(skeleton(node).encode()).hexdigest()[:12]


def _collect_stmts(tree: ast.AST, line_offset: int, out: list) -> None:
    """按文档序收集所有语句节点(含复合语句),记录绝对行号区间与节点类型。"""
    for child in ast.iter_child_nodes(tree):
        if isinstance(child, ast.stmt):
            out.append(
                (
                    child.lineno + line_offset,
                    (child.end_lineno or child.lineno) + line_offset,
                    skel_hash(child),
                    type(child).__name__,
                )
            )
        _collect_stmts(child, line_offset, out)


def _try_parse(chunk: str):
    """返回 (tree, wrapped)。悬空缩进块用 if 1: 包裹后重试。"""
    try:
        return ast.parse(chunk), False
    except (SyntaxError, ValueError, MemoryError):
        pass
    try:
        return ast.parse("if 1:\n" + chunk), True
    except (SyntaxError, ValueError, MemoryError):
        return None, False


def _tolerant_impl(text: str):
    """容错切块解析全文 → (语句表[(起行,止行,骨架hash,节点类型)], 解析覆盖行数, 总非空行数)。

    策略:从当前行起尝试解析到文末;失败取 SyntaxError.lineno 前的前缀再试;
    前缀也不行就丢弃当前行前进一行。wrapped 块的行号回退 1 对齐。
    """
    lines = text.split("\n")
    n = len(lines)
    stmts: list = []
    covered = set()
    start = 0
    while start < n:
        if not lines[start].strip():
            start += 1
            continue
        chunk_lines = lines[start:]
        tree, wrapped = _try_parse("\n".join(chunk_lines))
        end = n
        if tree is None:
            # 候选切点:unwrapped 与 wrapped 两种解析的报错行各取(wrapped 行号回退伪 if 一行),
            # 再各向前回退 1-3 行——覆盖「报错指向悬空块头、真切点在更前」的截断尾
            cuts = []
            try:
                ast.parse("\n".join(chunk_lines))
            except SyntaxError as e:
                cuts.append(max((e.lineno or 1) - 1, 0))
            except (ValueError, MemoryError):
                pass
            try:
                ast.parse("if 1:\n" + "\n".join(chunk_lines))
            except SyntaxError as e:
                cuts.append(max((e.lineno or 2) - 2, 0))
            except (ValueError, MemoryError):
                pass
            for base in list(cuts):
                cuts.extend(base - k for k in (1, 2, 3))
            for cut in sorted({c for c in cuts if 0 < c < len(chunk_lines)}, reverse=True):
                tree, wrapped = _try_parse("\n".join(chunk_lines[:cut]))
                if tree is not None:
                    end = start + cut
                    break
            if tree is None:
                start += 1  # 当前行救不回来,丢弃
                continue
        # 绝对 0 基行号 = lineno + offset;wrapped 时首行是伪 `if 1:`,真实代码 lineno 从 2 起
        offset = start - 2 if wrapped else start - 1
        top = tree.body[0].body if wrapped and isinstance(tree.body[0], ast.If) else tree.body
        holder = ast.Module(body=top, type_ignores=[])
        _collect_stmts(holder, offset, stmts)
        for ln in range(start, min(end, n)):
            if lines[ln].strip():
                covered.add(ln)
        if end <= start:
            start += 1
        else:
            start = end
    total_nonblank = sum(1 for l in lines if l.strip())
    stmts.sort(key=lambda t: (t[0], t[1]))
    return stmts, len(covered), total_nonblank


def tolerant_stmts(text: str):
    """R1 兼容接口:语句表为三元组 (起行,止行,骨架hash)。"""
    stmts, covered, total = _tolerant_impl(text)
    return [(a, b, h) for a, b, h, _ in stmts], covered, total


def tolerant_stmts_typed(text: str):
    """R2 接口:语句表为四元组 (起行,止行,骨架hash,节点类型名)。"""
    return _tolerant_impl(text)


def sliding_ur(seq: list, window: int) -> list:
    """位置 i(0 基,i>=window-1)→ 窗口 [i-window+1, i] 的 unique ratio。"""
    out = []
    for i in range(window - 1, len(seq)):
        win = seq[i - window + 1 : i + 1]
        out.append(len(set(win)) / window)
    return out


def first_below(curve: list, threshold: float, index_offset: int = 0):
    """首次跌破阈值的位置(加 index_offset 换算回原序列坐标);没有则 None。"""
    for i, v in enumerate(curve):
        if v < threshold:
            return i + index_offset
    return None

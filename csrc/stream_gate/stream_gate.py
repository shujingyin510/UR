# -*- coding: utf-8 -*-
"""流式三层退化门（参考实现，零模型、零下载）。

依据 R1–R4 周期谱框架：固定窗 token-UR 只是亚行短波带通；全谱需三层并联：
  1) token 轴  — 滑动窗 unique ratio（经典 UR≈0.30 短波）
  2) 骨架轴    — 语句骨架哈希序列的收尾周期（中波 / 变异复读）
  3) 行轴      — strip 后行哈希的收尾周期（长波 / 逐字散文）

判决规则（R2）：任一层「收尾周期成立」且「非自然 EOS」→ 判退化。
地板定律：报警 ⇔ 盲区 (L>W)∨(D≥θW)；本门用周期直接绕开固定窗盲区。

用法（离线回放 / 在线 feed）：
    g = StreamGate()
    g.feed_token("foo"); g.feed_token("bar"); ...
    g.feed_line("for i in range(n):")
    g.feed_stmt_hash("a1b2c3")   # 或 g.feed_stmt_text("result.append(item)")
    g.finish(eos=False)  → GateVerdict

合成/既有语料均可；不加载任何本地模型权重。
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Optional


def _line_hash(line: str) -> str:
    return hashlib.md5(line.strip().encode("utf-8")).hexdigest()[:12]


def _skel_hash_from_text(stmt: str) -> str:
    """轻量骨架：标识符/数字归一，保留运算符与关键字形状（不依赖 ast，便于流式增量）。

    完整 AST 骨架见 csrc/ast_ur/ast_ur.py；流式门优先低延迟，此处用词法级归一。
    """
    out = []
    for ch in stmt.strip():
        if ch.isalpha() or ch == "_":
            out.append("N")
        elif ch.isdigit():
            out.append("D")
        elif ch.isspace():
            if out and out[-1] != " ":
                out.append(" ")
        else:
            out.append(ch)
    # 压缩连续 N/D
    compact: list = []
    for c in out:
        if c in ("N", "D") and compact and compact[-1] == c:
            continue
        compact.append(c)
    return hashlib.md5("".join(compact).encode("utf-8")).hexdigest()[:12]


def periodic_tail(seq: list, min_len: int = 6, end_slack: int = 3):
    """与 R2 同款：最小收尾周期 → (p, start, end, cycles) 或 None。"""
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


def unique_ratio(seq: list, window: int) -> float:
    if not seq:
        return 1.0
    win = seq[-window:] if len(seq) >= window else seq
    return len(set(win)) / len(win)


@dataclass
class AxisHit:
    axis: str
    kind: str  # "period" | "ur_low"
    detail: str
    period: Optional[int] = None


@dataclass
class GateVerdict:
    degenerate: bool
    reason: str
    hits: list = field(default_factory=list)
    token_ur: float = 1.0
    stats: dict = field(default_factory=dict)


class StreamGate:
    """三层并联流式门。块外/短序列不误报；自然 EOS 抑制收尾周期误报。"""

    def __init__(
        self,
        token_window: int = 32,
        token_threshold: float = 0.30,
        stmt_window: int = 8,
        stmt_threshold: float = 0.50,
        min_period_len: int = 6,
    ) -> None:
        self.token_window = token_window
        self.token_threshold = token_threshold
        self.stmt_window = stmt_window
        self.stmt_threshold = stmt_threshold
        self.min_period_len = min_period_len
        self._tokens: list = []
        self._stmts: list = []
        self._lines: list = []

    def feed_token(self, tok) -> None:
        self._tokens.append(tok)

    def feed_tokens(self, toks) -> None:
        self._tokens.extend(toks)

    def feed_line(self, line: str) -> None:
        if line.strip():
            self._lines.append(_line_hash(line))

    def feed_stmt_hash(self, h: str) -> None:
        self._stmts.append(h)

    def feed_stmt_text(self, text: str) -> None:
        self._stmts.append(_skel_hash_from_text(text))

    def _token_ur(self) -> float:
        return unique_ratio(self._tokens, self.token_window)

    def evaluate(self) -> GateVerdict:
        """当前缓冲即时评估（不消费 EOS 语义）。"""
        hits: list = []
        tur = self._token_ur()

        # 轴1：token 短波
        if len(self._tokens) >= self.token_window and tur < self.token_threshold:
            hits.append(AxisHit("token", "ur_low", f"UR={tur:.3f}<{self.token_threshold}"))

        # 轴2：骨架收尾周期
        pt = periodic_tail(self._stmts, min_len=self.min_period_len)
        if pt is not None:
            p, start, end, cycles = pt
            hits.append(AxisHit("stmt", "period", f"p={p} cycles={cycles}", period=p))
        elif len(self._stmts) >= self.stmt_window:
            ur = unique_ratio(self._stmts, self.stmt_window)
            if ur < self.stmt_threshold:
                hits.append(AxisHit("stmt", "ur_low", f"stmt-UR={ur:.3f}"))

        # 轴3：行收尾周期
        lp = periodic_tail(self._lines, min_len=self.min_period_len)
        if lp is not None:
            p, start, end, cycles = lp
            hits.append(AxisHit("line", "period", f"p={p} cycles={cycles}", period=p))

        stats = {
            "n_tokens": len(self._tokens),
            "n_stmts": len(self._stmts),
            "n_lines": len(self._lines),
            "token_ur": round(tur, 4),
        }
        return GateVerdict(False, "pending_eos", hits, tur, stats)

    def finish(self, eos: bool = False) -> GateVerdict:
        """收尾判决：任一层收尾周期 ∧ 非自然 EOS → 退化。

        token 短波 UR 低且无周期，仍可单独触发（亚行增生）。
        自然 EOS 时：周期视为良性收尾（对称代码/样板），不判退化；UR 低仍报。
        """
        v = self.evaluate()
        period_hits = [h for h in v.hits if h.kind == "period"]
        ur_hits = [h for h in v.hits if h.kind == "ur_low"]

        if period_hits and not eos:
            axes = ",".join(h.axis for h in period_hits)
            return GateVerdict(True, f"trailing_period({axes})∧non_eos", v.hits, v.token_ur, v.stats)
        if ur_hits:
            return GateVerdict(True, ur_hits[0].detail, v.hits, v.token_ur, v.stats)
        if period_hits and eos:
            return GateVerdict(False, "period_but_natural_eos", v.hits, v.token_ur, v.stats)
        return GateVerdict(False, "healthy", v.hits, v.token_ur, v.stats)

    def reset(self) -> None:
        self._tokens.clear()
        self._stmts.clear()
        self._lines.clear()

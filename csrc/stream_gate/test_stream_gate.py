# -*- coding: utf-8 -*-
"""流式三层门单元测试（合成序列，零模型）。"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from stream_gate import StreamGate, periodic_tail, unique_ratio, _skel_hash_from_text  # noqa: E402


def test_token_ur_healthy():
    g = StreamGate(token_window=8, token_threshold=0.30)
    g.feed_tokens(list("abcdefghij"))
    v = g.finish(eos=True)
    assert not v.degenerate


def test_token_ur_shortwave_loop():
    g = StreamGate(token_window=8, token_threshold=0.30)
    g.feed_tokens(["was"] * 20)
    v = g.finish(eos=False)
    assert v.degenerate
    assert "ur_low" in v.reason or "UR=" in v.reason


def test_skeleton_period_not_visible_to_token_ur():
    """R1 场景：token 常新、骨架周期——token-UR 高，骨架轴报警。"""
    g = StreamGate(token_window=8, token_threshold=0.30, min_period_len=4)
    # 周期单元：两条不同语句骨架，标识符每次变化
    for i in range(6):
        g.feed_token(f"v{i}")
        g.feed_token(f"w{i}")
        g.feed_stmt_text(f"result{i}.append(item{i})")
        g.feed_stmt_text(f"out{i}.add(x{i})")
    # 再喂一整轮，保证收尾周期
    for i in range(6, 8):
        g.feed_stmt_text(f"result{i}.append(item{i})")
        g.feed_stmt_text(f"out{i}.add(x{i})")
    v = g.finish(eos=False)
    assert v.token_ur > 0.5  # token 轴失明
    assert v.degenerate
    assert any(h.axis == "stmt" and h.kind == "period" for h in v.hits)


def test_line_period_prose():
    g = StreamGate(min_period_len=4)
    line = "the quick brown fox jumps over the lazy dog"
    for _ in range(8):
        g.feed_line(line)
        g.feed_token("x")  # token 侧可保持多样
    # 多样 token 以免 token 轴误伤
    g.feed_tokens("abcdefgh")
    v = g.finish(eos=False)
    assert v.degenerate
    assert any(h.axis == "line" and h.kind == "period" for h in v.hits)


def test_natural_eos_suppresses_period():
    g = StreamGate(min_period_len=4)
    for _ in range(8):
        g.feed_stmt_text("a = 1")
        g.feed_stmt_text("b = 2")
    v = g.finish(eos=True)
    assert not v.degenerate
    assert v.reason == "period_but_natural_eos"


def test_periodic_tail_helper():
    seq = [1, 2, 3, 1, 2, 3, 1, 2, 3]
    pt = periodic_tail(seq, min_len=6)
    assert pt is not None
    assert pt[0] == 3


def test_unique_ratio():
    assert unique_ratio(["a", "a", "a", "a"], 4) == 0.25
    assert unique_ratio(["a", "b", "c", "d"], 4) == 1.0


def test_skel_normalizes_ids():
    h1 = _skel_hash_from_text("result.append(item)")
    h2 = _skel_hash_from_text("out.add(x)")
    h3 = _skel_hash_from_text("result.append(item) + 1")
    assert h1 != h3  # 运算符差异保留
    # 同构形状：标识符全 N，调用结构相同 → 前两者在词法骨架上接近
    assert isinstance(h1, str) and len(h1) == 12


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("OK", t.__name__)
    print(f"{len(tests)} passed")

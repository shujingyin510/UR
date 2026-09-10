# 流式三层退化门（参考实现）

零模型、零下载。实现 R1–R4 框架下的在线判决：

| 轴 | 信号 | 作用 |
|----|------|------|
| token | 滑动窗 UR | 亚行短波（经典 0.30） |
| 语句骨架 | 收尾周期 | 中波 / 改名变异复读 |
| 行 | 收尾周期 | 长波 / 逐字散文 |

**判决**：任一层收尾周期 ∧ 非自然 EOS → 退化；token UR 过低可单独触发。

## 跑测试

```bash
python -X utf8 csrc/stream_gate/test_stream_gate.py
```

## 用法

```python
from stream_gate import StreamGate  # 或把本目录加入 sys.path

g = StreamGate()
for t in tokens:
    g.feed_token(t)
for line in lines:
    g.feed_line(line)
for stmt in stmts:
    g.feed_stmt_text(stmt)  # 或 feed_stmt_hash
print(g.finish(eos=False))
```

完整 AST 骨架（标识符/常量严格归一）见 `../ast_ur/ast_ur.py`；本模块词法骨架优先低延迟。

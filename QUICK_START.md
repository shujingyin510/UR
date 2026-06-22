# Quick Start — 5 Minutes to First Result

## Prerequisites

```bash
# Python 3.12+, with:
pip install torch numpy tiktoken huggingface_hub
```

## 1. Browse & Run

```bash
# Compile C operators
gcc -shared -O2 -o csrc/transformer_c.dll csrc/c_ops/transformer_c.c -lm
gcc -shared -O2 -o csrc/softmax_c.dll csrc/c_ops/softmax_c.c -lm

# GPT-2 124M weights auto-download on first run (via huggingface_hub)
```

## 2. Test Inference

```bash
python -X utf8 csrc/gpt2/gpt2_engine.py
```

Output:
```
Prompt: Once upon a time
Once upon a time, your friend and I were in an alley...
```

## 3. Run Ternary Gating Benchmark

```bash
python -X utf8 csrc/gpt2/gpt2_scale.py
```

Output:
```
GPT-2 124M — 1000 prompt benchmark
  Ternary:  avg_len=63.5  stop=2%   time=1673s
  EOS-only: avg_len=64.0  stop=0%   time=1629s
```

## What You Just Saw

The ternary gating caught GPT-2's repetition ("and and and...") at token ~12, while EOS-only generated 64 tokens of garbage. The UR threshold of 0.30 was the only signal needed — no complex heuristics.

## Next Steps

- [Full results](RESULTS.md) — all benchmark tables
- [Research report](docs/research/ternary_gating_report_EN.md) — methodology and analysis
- [Qwen2.5 validation](csrc/qwen/qwen25_bench.py) — zero false positive check

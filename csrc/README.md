# csrc/ — C Source & Inference Engine & Benchmarks

```
csrc/
├── README.md                     ← This file
├── tinystories_1m.bin            TinyStories 3.6M weights (47MB, LFS)
├── tinystories_28m.bin           TinyStories 28M weights (231MB, LFS)
│
├── c_ops/                        ═══ C Operator Library ═══
│   ├── transformer_c.c           LayerNorm + GELU + Residual (45 lines)
│   ├── softmax_c.c               expf Softmax (19 lines)
│   └── simd_demo.asm             AVX2 FMA GEMM 256×256 assembly kernel
│
├── gpt2/                         ═══ GPT-2 124M Inference ═══
│   ├── gpt2_engine.py            Inference engine (GPT-2 arch, C LN + KV Cache)
│   ├── gpt2_kv.py                KV Cache fast inference (verified logit_diff=0.000046)
│   ├── gpt2_bench.py             20 prompt quality comparison (EOS vs ternary gating)
│   ├── gpt2_scale.py             ★ 1000 prompt full benchmark
│   ├── gpt2_blind.py             ★ 100 prompt blind evaluation material
│   ├── gpt2_blind_judge.py       Blind evaluation automated judging engine
│   ├── gpt2_sizes.py             GPT-2 cross-scale validation (124M/355M/774M)
│   └── gpt_neo_engine.py         GPT-Neo 125M inference
│
├── tinystories/                  ═══ Ternary Gating Benchmarks (TinyStories) ═══
│   ├── ternary_infer.py          Ternary gated inference engine v4 (trajectory detection)
│   ├── ternary_bench.py          100 prompt comparison benchmark
│   ├── ternary_scale.py          1000 prompt large benchmark (3.6M)
│   ├── ternary_scale_28m.py      1000 prompt large benchmark (28M, corrected UR=0.30)
│   ├── quality_test.py           3.6M model quality test
│   └── quality_28m.py            28M model quality test
│
├── qwen/                         ═══ Qwen2.5-0.5B Validation ═══
│   ├── qwen25_bench.py           ★ 1000 prompt UR zero-FP verification (FPR 0.4%)
│   ├── qwen25_zh_bench.py        Chinese prompt validation
│   └── qwen_degen.py             ★ Induced degeneration experiment (9 bad-prompt types)
│
├── ur_analysis/                  ═══ UR Analysis Tools ═══
│   ├── ur_ablation.py            Signal ablation experiment
│   ├── ur_baselines.py           Human text baseline (WikiText-2 n=60)
│   ├── ur_viz.py                 UR visualization
│   ├── threshold_analysis.py     Threshold analysis
│   ├── roc_analysis.py           ★ Measured ROC (1214 samples, Youden's J)
│   ├── dual_channel.py           Dual-channel detector (UR + SBERT semantic)
│   └── failure_analysis.py       Failure case analysis
│
├── adaptive/                     ═══ Adaptive Control ═══
│   └── adaptive_control.py       Adaptive closed-loop control (UR dynamic penalty)
│
└── vm/                           ═══ C VM ═══
    ├── vm_seed.c                 Level 3: C seed VM (318 lines, TCC-compilable)
    └── vm_l4.asm                 Level 4: x86_64 NASM assembly VM (617 lines)
```

## Key Files

### Inference Engine Pipeline

```
Python evaluator          Scheduling layer
  ↓
Python wrapper            Operator wrapping layer
  ↓ ctypes
C DLL (transformer_c)    C operator layer (LayerNorm/GELU)
  ↓ numpy @
GPT-2 124M inference      Matrix computation layer
```

### Verified Conclusions

| Conclusion | Evidence Files |
|------------|---------------|
| UR=0.30 effective across 4 models, 3 architectures | ternary_scale.py, gpt2_scale.py, qwen25_bench.py |
| KV Cache correctness (logit_diff=0.000046) | gpt2_kv.py |
| C LayerNorm precision (vs PyTorch diff=1e-7) | gpt2_engine.py |
| Ablation: UR-only = full trajectory detection | ur_analysis/ur_ablation.py |

### Running

```bash
# GPT-2 inference
python -X utf8 csrc/gpt2/gpt2_scale.py       # 1000 prompt benchmark

# Qwen2.5 verification
python -X utf8 csrc/qwen/qwen25_bench.py      # 1000 prompt UR check
python -X utf8 csrc/qwen/qwen_degen.py        # Induced degeneration experiment

# C operator compilation
gcc -shared -O2 -o csrc/transformer_c.dll csrc/c_ops/transformer_c.c -lm
nasm -f bin -o csrc/simd_demo.dll csrc/c_ops/simd_demo.asm
```

## File Sizes

| Category | Files | Total Size |
|----------|-------|------------|
| Python scripts | ~35 | ~300 KB |
| C source | 11 | ~190 KB |
| Assembly source | 2 | ~30 KB |

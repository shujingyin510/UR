# UR — Degeneration as Emergent Period: Spectrum, Floor Law, and Multi-Scale Detection

![Models](https://img.shields.io/badge/models-GPT--2%20%7C%20Qwen2.5%20%7C%20TinyStories-blue)
![Theory](https://img.shields.io/badge/theory-R1--R4%20Period%20Spectrum-green)
![UR](https://img.shields.io/badge/short-wave-UR%E2%89%880.30-orange)

> **Code-generation collapse is an attractor with (unit, wavelength L, alphabet D). A fixed-window uniqueness-ratio detector is a band-pass filter — UR≈0.30 covers the short-wave band only. Full coverage needs three parallel axes.**

[Quick Start](QUICK_START.md) | [Results](RESULTS.md) | [Theory Update](#theory-update-r1--r4) | [Research](docs/research/ternary_gating_report_EN.md) | [Roadmap](ROADMAP.md)

---

## Theory Update (R1–R4)

Four local experiments (192 sequences: 4 decoders × 24 prompts × greedy / T=0.8 / top-p=0.95, zero API) unified into a **period-spectrum framework**:

```text
Emergent period (unit × L × D)
        → model fingerprint P(L)
        → representation axis × decoding measure
        → detector = band-pass filter;  UR_floor ≈ D/W
        → blind ⇔ (L > W) ∨ (D ≥ θW)
        → three-axis parallel gate
        → decision: trailing period ∧ non-natural EOS
```

| Claim | Evidence |
|-------|----------|
| **H1 Universality** | 4/4 decoders show trailing periods under greedy (GPT-2 family 92–100%, Qwen 33%) |
| **H2 Sampling = frequency shift** | Sampling does not create new attractors (from nowhere 1/32); it shifts wavelength 23→41 and verbatim share 100%→20%. Weak models: degeneration survives at 19/24 — shift ≠ eliminate |
| **H3 Two-factor blindness** | Window-vs-period alone ≈40% (coin flip); with D, `(L>W)∨(D≥θW)` reaches 89–97% |
| **Floor law** | `UR_floor ≈ unique-period-elements / W` — greedy identity r=0.998; sampling first-order r=0.742 |
| **Spectrum = fingerprint** | GPT-2 single-peak short-wave (L median 23); Qwen long tail; scale shifts spectrum right (23→31→32) but not degeneration rate — rate tracks training recipe |
| **Decision rule** | Trailing period ∧ non-natural EOS: 32/32 zero false positives on greedy battery |

**Where the classic UR≈0.30 threshold sits**: it is the **sub-line short-wave band-pass** (window=32). It is necessary but not sufficient — R1 showed it is blind on ~85% of long-period code degeneration cases (whole-function loops, renamed-block repeats).

Full derivation, ASCII fingerprints, and falsification conditions live in the project knowledge base notes (`UR-退化理论总览` and R1–R4 experiment notes).

---

## Main Finding (short-wave band: UR≈0.30)

A single **uniqueness-ratio threshold of 0.30** — the fraction of unique tokens in a sliding 32-token window — detects short-wave repetitive degeneration:

| Model | Architecture | Params | Behavior | UR=0.30 Result |
|-------|-------------|--------|----------|----------------|
| TinyStories 3.6M | GPT-Neo | 3.6M | Degenerates | True Positive 100% |
| TinyStories 28M | GPT-Neo | 28M | Degenerates | True Positive 100% |
| GPT-2 124M | GPT-2 | 124M | Rarely degenerates (clean sampling) | Stop rate 2% |
| Qwen2.5-0.5B | Qwen2 | 494M | Coherent | False Positive 0.4% |

**Key result**: The threshold detects degeneration when it actually occurs — 100% on TinyStories (which degenerate regardless of sampling) and 2% on GPT-2 with clean sampling (which almost never degenerates). The original GPT-2 TPR of 100% was a rep_penalty-induced artifact (§3.4). FPR on a coherent model (Qwen2.5-0.5B) is 0.4% (p < 0.05). UR-only ablation shows UR is the dominant signal; auxiliary signals provide marginal early warning in 3.6% of cases.

---

## Why It Matters

Small language models frequently collapse into repetitive loops ("was was was...", "and and and...") with **confidence scores remaining at 0.97-1.00** — the model believes it's producing high-quality output while generating garbage. Standard stopping strategies (EOS token, max token limit, repetition penalty) fail to detect this.

**UR < 0.30** catches degeneration the moment it happens:

| Strategy | Avg Length | Stop Rate |
|----------|-----------|-----------|
| UR < 0.30 | 9-64 tokens | **2-100%** (depends on model) |
| EOS-only | 64 tokens | 0% |
| Repetition Penalty | 64 tokens | 0% |

Human blind evaluation across 100 prompts: **ternary gating preferred 79.7% vs. EOS-only 8.3%** (12% ties).

### Empirical Validation

**UR trajectories show a phase transition**, not just a statistical drop — UR declines monotonically from ~0.70 to ~0.10, crosses 0.30 at t=18–28, and **no recovery is observed within the evaluated horizon** (≤64 tokens):

| Step | "Once upon a time" | "The little boy" | "A big dog" |
|------|--------------------|-------------------|-------------|
| t=9 | 0.62 | 0.88 | 0.50 |
| t=18 | 0.35 | 0.41 | **0.24** |
| t=28 | 0.26 | **0.26** | 0.19 |
| t=48 | 0.16 | 0.03 | 0.06 |

**Human vs. degenerative text separation** (window=32):

| Text Type | Avg UR | UR < 0.30 |
|-----------|--------|-----------|
| Human (literature) | **0.704** | **0.0%** |
| Human (WikiText-2, n=60) | **0.849** | **0.2%** |
| Degenerative (GPT-2) | **0.101** | **99.7%** |

**Decoding strategy comparison** on GPT-2 124M:

| Strategy | Degeneration Rate | Avg UR |
|----------|-------------------|--------|
| nucleus (top_p=0.9) | **0%** | **0.867** |
| greedy | 25% | 0.336 |
| rep_penalty=1.15 | **100%** | 0.117 |

> **Counterintuitive**: repetition penalty *amplifies* collapse on GPT-2 by narrowing the effective sampling space. UR correctly reflects each strategy's actual degeneration level independent of the strategy's assumptions.

**GPT-2 cross-size UR stability** (nucleus sampling, top_p=0.9):

| Model | Params | Avg UR | UR < 0.30 |
|-------|--------|--------|-----------|
| GPT-2 | 124M | 0.711 | 0% |
| GPT-2 Medium | 355M | 0.714 | 0% |
| GPT-2 Large | 774M | 0.797 | 0% |

> UR varies only ±0.043 across 6× scale. Larger models → higher UR → more diverse output. UR functions as a stable generation diversity metric, not just a degeneration detector.

**Cross-language stability**: Qwen2.5-0.5B on Chinese (n=1000): FP=**0.6%**, avg min_UR=**0.714** — vs English FP=0.4%, avg=0.717. UR threshold is language-agnostic at scale.

**Threshold selection** (real-data ROC, 1214 samples): Youden's J optimum = 0.32. We chose **0.30** — the TPR "knee point" where detection jumps from 0.847→0.993. Conservative relative to the optimum, minimizing FPR.

---

## Architecture

```
Python / C VM
    ↓
Native FFI
    ↓
Native FFI (reg_op)
    ↓
AVX2 GEMM + C LayerNorm/GELU/Softmax
    ↓
GPT-2 / GPT-Neo / Qwen2 Transformer
    ↓
KV Cache Inference
    ↓
UR-based Degeneration Detection (UR_TH = 0.30)
```

---

## Known Boundaries

The current findings (UR ≈ 0.30 as a degeneration threshold) are empirical and should be interpreted within the following constraints:

### 1. Windowed lexical measurement

`unique_ratio` is computed over a fixed sliding window of tokens. All reported results use a window size of 32 tokens, stride = 1. The threshold is stable under moderate window sizes (32–64), but is not invariant across arbitrary scales.

### 2. Regime definition (not quality classification)

UR measures **repetition-dominated generation regimes**, not semantic correctness or overall output quality. Therefore:

- Structured outputs (code, lists, enumerations)
- Poetic or stylistically constrained text

may exhibit low UR while remaining valid. These cases are not considered false positives, but a different generation regime outside the detector's target domain.

### 3. Prompt-induced repetition is a separate regime

Repetition explicitly present in the input prompt (e.g., "cat cat cat") is treated as input-conditioned behavior, not model-internal degeneration. The detector is designed for emergent repetition during generation, not echoing input structure.

### 4. Empirical model coverage

The current evaluation includes:
- TinyStories (3.6M, 28M)
- GPT-2 (124M)
- Qwen2.5-0.5B

Results are consistent across these models, but this should be interpreted as *empirical cross-model stability within tested regimes*, not full model-invariance across all architectures.

### 5. Scale and horizon limitation

No evaluation has been performed on:
- 7B+ parameter models (e.g., LLaMA-3, Qwen2.5-7B)
- Instruction-tuned large chat models in open-ended dialogue regimes
- Horizons beyond ~600 tokens (R1–R4 battery)

Generalization beyond the tested decoders and horizon remains open.

### 6. Fixed-window blind band (theory)

With window W=32, degeneration whose period L>W or alphabet D≥θW is **structurally invisible** to token-UR (R1/R2/R4). Use multi-scale evidence (token-UR ∨ skeleton period ∨ line period) for full coverage; see Theory Update above.

---

## Quick Start

```bash
# Run ternary gating benchmark (GPT-2 124M, 1000 prompts)
python -X utf8 csrc/gpt2/gpt2_scale.py

# Compile C operators
gcc -shared -O2 -o csrc/transformer_c.dll csrc/c_ops/transformer_c.c -lm
gcc -shared -O2 -o csrc/softmax_c.dll csrc/c_ops/softmax_c.c -lm
```

---

## Repository Layout

```
UR/
├── README.md                     ← English main
├── RESULTS.md                    ← Full result tables
├── ROADMAP.md                    ← Completed & planned
├── QUICK_START.md                ← One-command setup
├── CHANGELOG.md                  ← Version history
├── LICENSE
├── .gitignore / .gitattributes
├── research/                     ← Tokenizer-Language experiments
│   └── tokenizer_dsl/
│       ├── README.md             ← Bilingual experiment report
│       ├── token_bench.py        ← 2-model benchmark
│       ├── multi_bench.py        ← 4-model benchmark
│       ├── keyword_cost.py       ← 71-keyword analysis
│       ├── experiment_abc.py     ← Scaling + cross-lang + random
│       ├── controlled.py         ← Causality experiment
│       ├── natural_text.py       ← Real-world text comparison
│       └── word_decomp.py        ← Token decomposition
├── docs/research/
│   ├── ternary_gating_report.md      ← Research report (Chinese)
│   └── ternary_gating_report_EN.md   ← Research report (English)
└── csrc/
    ├── README.md                 ← csrc documentation
    ├── tinystories_1m.bin         ← TinyStories 3.6M weights (47MB, LFS)
    ├── tinystories_28m.bin         ← TinyStories 28M weights (231MB, LFS)
    ├── c_ops/                    ← C operator library
    │   ├── transformer_c.c       LayerNorm + GELU + Residual
    │   ├── softmax_c.c           expf Softmax
    │   └── simd_demo.asm         AVX2 FMA GEMM kernel
    ├── gpt2/                     ← GPT-2 124M inference
    │   ├── gpt2_engine.py        Inference engine
    │   ├── gpt2_kv.py            KV Cache (logit_diff=0.000046)
    │   ├── gpt2_scale.py         ★ 1000 prompt benchmark
    │   └── gpt2_blind.py         ★ Blind evaluation
    ├── tinystories/              ← TinyStories 3.6M/28M benchmarks
    │   ├── ternary_infer.py      Ternary gating engine v4
    │   └── ternary_scale.py      1000 prompt benchmarks
    ├── qwen/                     ← Qwen2.5-0.5B validation
    │   ├── qwen25_bench.py       ★ UR false positive verification
    │   └── qwen_degen.py         ★ Induced degeneration
    ├── ur_analysis/              ← UR analysis tools
    │   ├── ur_ablation.py        Signal ablation
    │   ├── roc_analysis.py       ★ ROC measurement (1214 samples)
    │   └── dual_channel.py       UR + SBERT semantic
    ├── adaptive/                 ← Adaptive control
    │   └── adaptive_control.py   UR-based closed-loop control
    └── vm/                       ← C VM / compiler
        ├── vm_seed.c             C seed VM (318 lines)
        └── vm_l4.asm             x86_64 NASM assembly VM
```

---

## Current Status

| Component | Status |
|-----------|--------|
| UR=0.30 short-wave validation (4 models, 3 architectures) | ✅ |
| 1000-prompt benchmark per model | ✅ |
| Human blind evaluation (100 prompts) | ✅ |
| Ablation: UR-only vs full trajectory | ✅ |
| Statistical significance (p < 0.05) | ✅ |
| **R1 AST skeleton repeat rate** | ✅ |
| **R2 Floor law (UR≈D/W)** | ✅ |
| **R3 Sampling frequency-shift** | ✅ |
| **R4 Period spectrum + multi-scale decision** | ✅ |
| AVX2 GEMM kernel (66 GFLOPS) | ✅ |
| C LayerNorm/GELU/Softmax kernels | ✅ |
| KV Cache inference (logit_diff=0.000046) | ✅ |
| **Streaming three-axis gate prototype** | ✅ 参考实现 `csrc/stream_gate/`（零模型） |
| Temperature scan × multi-seed shift curves | ⬜ 需本地算力，暂缓 |
| ROC threshold expansion (multi-axis) | ⬜ |
| Instruction models / longer horizon | ⬜ |
| GGUF / quantization | ⬜ |
| Paper submission (period-spectrum mainline) | ⏳ 标题/摘要/Intro 已升维 |

---

## Documentation

| Document | Description |
|----------|-------------|
| [RESULTS.md](RESULTS.md) | All benchmark results with tables |
| [ROADMAP.md](ROADMAP.md) | Completed and planned work |
| [docs/research/ternary_gating_report_EN.md](docs/research/ternary_gating_report_EN.md) | Full research report (Chinese + English abstract) |
| [csrc/README.md](csrc/README.md) | C source and inference engine docs |
| [CHANGELOG.md](CHANGELOG.md) | Version history |

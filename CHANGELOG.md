# Changelog — UR Degeneration Detection

## UR v1.0 — 2026-06-20

### Core discovery
- **UR ≈ 0.30**: single uniqueness-ratio threshold reliably separates degenerative from coherent generation
- Validated across 4 models, 3 architectures (GPT-Neo, GPT-2, Qwen2), parameter range 3.6M–494M
- True positive rate (clean decoding, 1000 prompts): 100% on degenerating TinyStories, 2% on GPT-2 with clean sampling (rarely degenerates)
- Original GPT-2 100% TPR was rep_penalty-induced artifact (§3.4)
- False positive rate: 0.4% on coherent model [0.01-0.79%] (p < 0.05)
- UR-only ablation: UR is the dominant signal; auxiliary signals (cycle+no-new-word) provide early warning in 3.6% of cases

### Key experiments
- **1000-prompt benchmark** (per model): ternary_gating vs EOS-only
- **Human blind evaluation** (100 prompts, 3 dimensions): ternary 78% vs EOS 8%
- **ROC measurement** (1214 samples): Youden's J optimum at 0.32; threshold chosen at conservative 0.30
- **Human text baseline** (n=60 WikiText-2): avg UR=0.849 [0.846,0.852]
- **Cross-lingual validation** (Qwen2.5, EN+ZH, 1000 each): FPR 0.4-0.6%
- **Induced degeneration** (Qwen2.5, 9 prompt types): UR correctly distinguishes collapse from understanding
- **Adaptive closed-loop control**: UR-based dynamic penalty (avg UR=0.790 vs greedy 0.291)
- **Dual-channel detection**: lexical (UR) + semantic (SBERT cosine similarity)
- **Sampling strategy comparison**: nucleus (0.867) > greedy (0.336) > rep_penalty (0.117)

### Codebase
- GPT-2 124M inference engine (C LayerNorm/GELU + Python KV Cache, logit_diff=0.000046)
- C operator library (LayerNorm, Softmax, AVX2 GEMM)
- C VM / compiler (ISA v2, TCC-compilable seed + NASM assembly)
- 35 Python scripts, 11 C sources, 2 assembly sources

### Reproducibility
- TinyStories weights (3.6M, 28M) included via Git LFS
- GPT-2 124M auto-download from HuggingFace (`huggingface_hub`)
- Qwen2.5-0.5B auto-download from HuggingFace (`transformers`)

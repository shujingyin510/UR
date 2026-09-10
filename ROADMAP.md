# Roadmap

> Narrative mainline (2026-09-10): **period-spectrum theory (R1–R4)**.  
> UR≈0.30 is retained as the short-wave band-pass special case, not the sole claim.  
> Engineering Next items below hang on **theory boundaries**, not feature wishlists.

## Completed

| Milestone | Details |
|-----------|---------|
| **C VM (ISA v2)** | 16-bit LOAD/STORE, 32-bit CALL, CLOSURE, PUSH_STR16 |
| **Level 3 Bootstrap** | 318-line C seed VM → TCC-compiled binary |
| **Level 4 Bootstrap** | 617-line x86_64 NASM assembly VM |
| **AVX2 GEMM Kernel** | FMA instructions, 256×256, 66 GFLOPS, zero error vs NumPy |
| **C Operator Library** | LayerNorm (err e-07), Softmax (err e-09), GELU (err e-08) |
| **TinyStories 3.6M / 28M** | GPT-Neo inference, KV Cache, 1000-prompt benchmarks |
| **GPT-2 124M** | GPT-2 inference (Conv1D + pre-norm), KV Cache, 1000-prompt benchmark |
| **UR Threshold Calibration** | Auto-calibrated to 0.30 across 3.6M and 28M (short-wave band) |
| **Qwen2.5-0.5B Validation** | 1000-prompt false positive check (0.4%) |
| **Human Blind Evaluation** | 100 prompts × 3 dimensions, ternary 79.7% preferred |
| **Ablation Study** | UR-only = full trajectory (all other signals redundant in short-wave) |
| **Statistical Significance** | p = 0.0287, 95% CI [0.01%, 0.79%] |
| **R1 AST skeleton repeat** | Token-UR blind on ~85% of long-period code degeneration |
| **R2 Floor law** | `UR_floor ≈ D/W` (greedy r=0.998); trailing-period ∧ non-EOS 32/32 |
| **R3 Sampling shift** | Sampling shifts frequency, does not eliminate (weak model 19/24 survive) |
| **R4 Period spectrum** | H1–H3 verdicts; multi-scale three-axis coverage; fingerprint P(L) |
| **C FFI Demo** | reg_op → C DLL → GPT-2 end-to-end |

---

## Next (on theory boundaries)

| Priority | Item | Notes |
|----------|------|-------|
| 🔴 | **Streaming three-axis gate** | ✅ **2026-09-10 参考实现** `csrc/stream_gate/`（token ∨ 骨架周期 ∨ 行周期；收尾周期∧非 EOS；合成序列 8 测全绿，零模型） |
| 🔴 | **Temperature scan × multi-seed** | Map degeneration rate / wavelength / verbatim share vs T（需本地模型，**暂缓**：机器配置到上限） |
| 🔴 | **Paper mainline: period spectrum** | ✅ 标题/摘要/Intro/Method（涌现周期+地板定律+三层门）/Experiments R1–R4 /Conclusion /Limitations 已改写 |
| 🟡 | **ROC threshold expansion** | Multi-axis D<θW thresholds; merge with R1 sample pool |
| 🟡 | **Instruction models + longer horizon** | Chat models / >600 token; currently the largest validity gap |
| 🟡 | **Parser gap: dangling-if** | One known miss (gpt2l/reverse_words) from harness parse hole |
| 🟢 | **Spectrum validity** | Link P(L) shape to downstream capability / training recipe |
| 🟢 | **TinyLlama-1.1B** | Architecture transfer check under the spectrum frame |
| 🟢 | **GGUF / GPU** | Quantized loading / CUDA — only if gate needs production latency |

---

## Frozen / deprioritized

| Item | Reason |
|------|--------|
| Treating UR=0.30 as the sole contribution | Superseded by R1–R4; keep as short-wave band |
| Agent self-update lines in the sanyan monorepo | Frozen there; this repo stays research-only, zero sanyan dependency |
| GGUF/CUDA as P0 | Not on a theory boundary; defer until online gate is real |

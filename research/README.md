# Tokenizer-Language Alignment: Chinese DSL vs Python

> How much does tokenizer choice affect the viability of a Chinese syntax DSL for LLM code generation?

## Method

15 equivalent code snippets written in two syntaxes:
- **Chinese DSL**: `循环 10 次：\n    打印 i` (Chinese keywords, full-width punctuation)
- **Python**: `for i in range(10):\n    print(i)`

Each snippet tokenized with two tokenizers:
- **GPT-2** (English-optimized, BPE 50k vocab)
- **Qwen2.5-0.5B** (Chinese-English bilingual, BPE 151k vocab)

## Results

| Tokenizer | Chinese avg | Python avg | Gap |
|-----------|------------|-----------|-----|
| GPT-2 (English-optimized) | 30.9 tk/snippet | 16.0 tk/snippet | +93% |
| Qwen2.5 (bilingual) | 16.5 tk/snippet | 12.9 tk/snippet | +28% |

### Per-pattern breakdown (Qwen tokenizer)

| Pattern | Chinese | Python | Δ |
|---------|---------|--------|-----|
| While loop | 14 | 14 | **0%** |
| Return | 7 | 7 | **0%** |
| Assign | 11 | 12 | **-8%** (Chinese better) |
| Import | 3 | 2 | +50% |
| Lambda | 8 | 7 | +14% |
| Loop 10x | 13 | 12 | +8% |
| Function | 12 | 11 | +9% |
| File read | 17 | 15 | +13% |
| If/else | 24 | 18 | +33% |
| Class | 23 | 16 | +44% |
| Try/catch | 22 | 15 | +47% |
| Map/filter | 31 | 20 | +55% |
| Dict comp | 21 | 13 | +62% |

## Key Findings

1. **Tokenizers are the gating factor.** GPT-2's English-optimized tokenizer makes Chinese DSL 93% more expensive. Qwen's bilingual tokenizer narrows this to 28%.

2. **Simple constructs already reach parity.** `while`, `return`, `assign` are token-equivalent or better under Qwen. The gap is concentrated in complex constructs (`try/catch`, `class`, `map/filter`) where Chinese uses more function-call-style punctuation.

3. **LLM-era language design requires tokenizer-language co-design.** A Chinese DSL is only viable if the serving tokenizer is Chinese-aware. No amount of syntax optimization can overcome a misaligned tokenizer.

## Next Steps

- [ ] UR-based generation quality comparison (does lower token count → lower error rate?)
- [ ] AST stability comparison (S-expression vs Python AST generation consistency)
- [ ] Tool-call accuracy (Agent DSL vs free-form prompt)

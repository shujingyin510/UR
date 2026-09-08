# -*- coding: utf-8 -*-
"""R1 生成器:本地小模型贪心续写代码 prompt,产出 JSONL 语料。

零 API:模型从本地 HF 缓存加载(local_files_only)。贪心解码 → 确定性、可复现,
且小模型贪心最易退化——正是研究对象。

用法:
  python r1_gen.py --model qwen  [--probe]     # Qwen2.5-0.5B(代码能力主力)
  python r1_gen.py --model gpt2 [--probe]      # GPT-2 124M(UR 原实验同款对照)
输出: data/gen_{model}.jsonl  每行 {pid, prompt, text, ids, n_new, stopped_eos}
"""
from __future__ import annotations

import argparse
import json
import os
import time

MODELS = {
    "qwen": "Qwen/Qwen2.5-0.5B",
    "gpt2": "openai-community/gpt2",
}

PROMPTS = [
    ("merge_sort", 'def merge_sort(arr):\n    """Sort a list using merge sort."""\n'),
    ("binary_search", 'def binary_search(arr, target):\n    """Return index of target in sorted arr, or -1."""\n'),
    ("fibonacci", 'def fibonacci(n):\n    """Return the first n Fibonacci numbers as a list."""\n'),
    ("parse_csv", 'def parse_csv_line(line):\n    """Split a CSV line into fields, handling quoted commas."""\n'),
    ("word_freq", 'def word_frequency(text):\n    """Count how often each word appears in text."""\n'),
    ("flatten", 'def flatten(nested):\n    """Flatten an arbitrarily nested list."""\n'),
    ("stack_class", 'class Stack:\n    """A simple LIFO stack."""\n\n    def __init__(self):\n'),
    ("lru_cache", 'class LRUCache:\n    """Least-recently-used cache with fixed capacity."""\n\n    def __init__(self, capacity):\n'),
    ("prime_sieve", 'def primes_up_to(n):\n    """Return all primes <= n using the sieve of Eratosthenes."""\n'),
    ("roman", 'def to_roman(num):\n    """Convert an integer to a Roman numeral string."""\n'),
    ("balanced", 'def is_balanced(s):\n    """Check whether brackets in s are balanced."""\n'),
    ("transpose", 'def transpose(matrix):\n    """Transpose a 2D list."""\n'),
    ("rle", 'def run_length_encode(s):\n    """Compress a string with run-length encoding."""\n'),
    ("dedup", 'def remove_duplicates(items):\n    """Remove duplicates while preserving order."""\n'),
    ("gcd", 'def gcd(a, b):\n    """Greatest common divisor via Euclid."""\n'),
    ("quicksort", 'def quicksort(arr):\n    """Sort a list using quicksort."""\n'),
    ("tree_walk", 'def inorder(node, visit):\n    """In-order traversal of a binary tree."""\n'),
    ("json_walk", 'def count_keys(obj):\n    """Recursively count all keys in nested dicts and lists."""\n'),
    ("email", 'def is_valid_email(addr):\n    """Very small email validity check."""\n'),
    ("temp", 'def celsius_to_fahrenheit(c):\n    """Convert Celsius to Fahrenheit."""\n'),
    ("reverse_words", 'def reverse_words(sentence):\n    """Reverse word order in a sentence."""\n'),
    ("anagram", 'def are_anagrams(a, b):\n    """Check whether two strings are anagrams."""\n'),
    ("pascal", "def pascal_triangle(rows):\n    \"\"\"Return Pascal's triangle as a list of rows.\"\"\"\n"),
    ("chunk", 'def chunks(lst, size):\n    """Yield successive chunks of the given size."""\n'),
]

MAX_NEW = 600


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODELS), required=True)
    ap.add_argument("--probe", action="store_true", help="只跑 1 条 32 token 测速")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    name = MODELS[args.model]
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(name, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        name, local_files_only=True, dtype=torch.float32
    )
    model.eval()
    print(f"[{args.model}] 加载 {time.time()-t0:.1f}s", flush=True)

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    os.makedirs(out_dir, exist_ok=True)
    prompts = PROMPTS[:1] if args.probe else PROMPTS
    max_new = 32 if args.probe else MAX_NEW
    path = os.path.join(out_dir, f"gen_{args.model}{'_probe' if args.probe else ''}.jsonl")

    with open(path, "w", encoding="utf-8") as f:
        for pid, prompt in prompts:
            enc = tok(prompt, return_tensors="pt")
            t1 = time.time()
            with torch.no_grad():
                out = model.generate(
                    **enc,
                    max_new_tokens=max_new,
                    do_sample=False,
                    pad_token_id=tok.eos_token_id,
                )
            ids = out[0][enc["input_ids"].shape[1] :].tolist()
            stopped = bool(ids and ids[-1] == tok.eos_token_id)
            text = tok.decode(ids, skip_special_tokens=True)
            dt = time.time() - t1
            print(
                f"[{args.model}] {pid:14s} {len(ids):4d} tok  {dt:6.1f}s  {len(ids)/max(dt,0.01):5.1f} tok/s  eos={stopped}",
                flush=True,
            )
            f.write(
                json.dumps(
                    {
                        "pid": pid,
                        "prompt": prompt,
                        "text": text,
                        "ids": ids,
                        "n_new": len(ids),
                        "stopped_eos": stopped,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(f"[{args.model}] 完成 → {path}", flush=True)


if __name__ == "__main__":
    main()

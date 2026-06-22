"""Agent layered testing — rule coverage vs LLM intervention vs failure analysis"""

import subprocess
import time
import json
import os

tests = [
    # Expected: full rule engine (no LLM)
    ('创建math_utils.py，实现is_prime和gcd', 'rule'),
    ('创建string_utils.py，实现reverse_string', 'rule'),
    ('修复csrc/gpt2_engine.py的导入错误', 'rule'),
    ('列出csrc下所有Python文件', 'rule'),
    ('解释run_agent.py的代码逻辑', 'rule'),
    # Expected: LLM intervention (QA/planning/chat)
    ('Python的GIL是什么', 'LLM'),
    ('设计一个博客系统的数据库ER图', 'LLM'),
    ('量子纠缠能超光速通信吗', 'LLM'),
    ('Java的volatile和Go的channel有什么区别', 'LLM'),
    ('推荐几部好看的科幻电影', 'LLM'),
    # Expected: may fail (edge cases)
    ('创建一个不存在的目录下的文件', 'boundary'),
    ('修复一个不存在的bug', 'boundary'),
    ('用Brainfuck写一个hello world', 'boundary'),
]

results = []
for i, (task, category) in enumerate(tests):
    t0 = time.perf_counter()
    try:
        r = subprocess.run(
            ['python', '-X', 'utf8', 'run_agent.py', task, '--rounds', '3'],
            capture_output=True,
            text=True,
            timeout=45,
            encoding='utf-8',
            errors='replace',
            cwd='D:/Test/sanyan',
        )
        output = r.stdout + r.stderr
    except subprocess.TimeoutExpired:
        output = 'TIMEOUT'

    dt = time.perf_counter() - t0

    # Analyze output
    rule_hit = 'match:' in output or '规则]' in output
    llm_used = 'LLM' in output or 'r1]' in output or '问答]' in output or 'legacy' in output.lower()
    ternary_triggers = output.count('[三态]')
    success = '✓' in output or 'verified' in output or 'done' in output or 'answer' in output
    failure = 'failed' in output or 'error' in output or 'TIMEOUT' in output

    # Classify
    if success and rule_hit and not llm_used:
        actual = 'pure-rule'
    elif success and rule_hit and llm_used:
        actual = 'rule+LLM'
    elif success and llm_used:
        actual = 'LLM-direct'
    elif failure:
        actual = 'failed'
    else:
        actual = 'other'

    results.append(
        {
            'task': task[:60],
            'cat': category,
            'actual': actual,
            'rule': rule_hit,
            'llm': llm_used,
            'ternary': ternary_triggers,
            'success': success,
            'time': f'{dt:.1f}s',
        }
    )

    status = '✅' if success else '❌'
    print(f'[{i + 1:2d}] {status} {actual:8s} | 三态x{ternary_triggers} | {task[:50]}')


# Summary
print(f'\n{"=" * 60}')
print(f'{"类别":<10} {"总数":>4} {"纯规则":>6} {"规则+LLM":>8} {"LLM直答":>8} {"失败":>4}')
print('-' * 60)
for cat in ['rule', 'LLM', 'boundary']:
    rs = [r for r in results if r['cat'] == cat]
    pure = sum(1 for r in rs if r['actual'] == 'pure-rule')
    r_llm = sum(1 for r in rs if r['actual'] == 'rule+LLM')
    llm_d = sum(1 for r in rs if r['actual'] == 'LLM-direct')
    fail = sum(1 for r in rs if r['actual'] == 'failed')
    print(f'{cat:<10} {len(rs):>4} {pure:>6} {r_llm:>8} {llm_d:>8} {fail:>4}')

# Statistics
total_ternary = sum(r['ternary'] for r in results)
print(f'\n三态门控触发: {total_ternary} 次')
print(f'总耗时: {sum(float(r["time"][:-1]) for r in results):.0f}s')

os.makedirs('benchmarks', exist_ok=True)
with open('benchmarks/agent_test_matrix.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

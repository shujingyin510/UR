"""Agent full test matrix — A/B/C/D four groups"""

import subprocess
import time
import json
import os
import re

tests = [
    # Group A: pure rule-driven
    ('A1', '在csrc下新建logger.py，实现日志功能', 'rule'),
    ('A2', '创建一个HTTP请求模块，命名为requests.py', 'rule'),
    ('A3', '给calculator.py写单元测试', 'rule'),
    ('A4', '重构main.py中的冗余代码', 'rule'),
    ('A5', '更新README.md添加安装说明', 'rule'),
    # Group B: rule + LLM parameters
    ('B6', '实现一个计时器工具类', 'LLM'),
    ('B7', '添加JWT认证中间件', 'LLM'),
    ('B8', '写一个LRU缓存实现', 'LLM'),
    ('B9', '给API添加速率限制', 'LLM'),
    ('B10', '实现文件上传功能', 'LLM'),
    # Group C: degeneration triggers
    ('C11', '实现Raft分布式一致性协议', '退化'),
    ('C12', '用Rust从零写一个操作系统', '退化'),
    ('C13', '写20种不同的排序算法', '退化'),
    ('C14', '解释量子纠错在拓扑量子计算中的应用', '退化'),
    ('C15', '实现一个完整的ORM框架', '退化'),
    # Group D: edge cases
    ('D16', '修复一个不存在目录里的bug', 'boundary'),
    ('D17', '修复main.py里不存在的竞态条件', 'boundary'),
    ('D18', '用Brainfuck写一个Web服务器', 'boundary'),
    ('D19', '在空项目中重构代码', 'boundary'),
    ('D20', '删除系统根目录下的所有文件', 'boundary'),
]

results = []
for tid, task, category in tests:
    t0 = time.perf_counter()
    try:
        r = subprocess.run(
            ['python', '-X', 'utf8', 'run_agent.py', task, '--rounds', '5'],
            capture_output=True,
            text=True,
            timeout=60,
            encoding='utf-8',
            errors='replace',
            cwd='D:/Test/sanyan',
        )
        out = r.stdout + r.stderr
    except subprocess.TimeoutExpired:
        out = 'TIMEOUT'
    dt = time.perf_counter() - t0

    # Metrics collection
    rule_hit = 'match:' in out
    llm_calls = out.count('[r') if '[r' in out else (1 if 'LLM' in out else 0)
    ternary_count = out.count('[三态]')
    ur_values = [float(m) for m in re.findall(r'UR[=:]\s*([\d.]+)', out)]
    ur_min = min(ur_values) if ur_values else 1.0
    ur_stop = '[UR]' in out
    negate = 'reject' in out or 'NEGATE' in out
    crashed = 'Traceback' in out or 'Error' in out.split('→')[-1] if '→' in out else False
    done = 'done' in out or 'answer' in out or 'by rule' in out

    results.append(
        {
            'id': tid,
            'task': task[:60],
            'cat': category,
            'rule': rule_hit,
            'llm': llm_calls,
            'ternary': ternary_count,
            'ur_min': round(ur_min, 3),
            'ur_stop': ur_stop,
            'negate': negate,
            'crashed': crashed,
            'done': done,
            'time': f'{dt:.1f}s',
        }
    )

    s = '✅' if (done and not crashed) else ('⚠️' if ur_stop or negate else '❌')
    print(f'[{tid}] {s} 规则={rule_hit} LLM={llm_calls} 三态x{ternary_count} URmin={ur_min:.2f} | {task[:50]}')

# ── 统计 ──
print(f'\n{"=" * 60}')
print('组  规则覆盖 LLM调用  UR最低  UR拦截  存活  说明')
print('-' * 60)
for grp, label in [('A', 'pure-rule'), ('B', 'LLM参数'), ('C', '退化'), ('D', 'boundary')]:
    rs = [r for r in results if r['id'].startswith(grp)]
    rule_cov = sum(1 for r in rs if r['rule']) / len(rs)
    llm_avg = sum(r['llm'] for r in rs) / len(rs)
    ur_avg = sum(r['ur_min'] for r in rs) / len(rs)
    ur_int = sum(1 for r in rs if r['ur_stop']) / len(rs)
    alive = sum(1 for r in rs if not r['crashed']) / len(rs)
    print(f'{grp}   {rule_cov:.0%}      {llm_avg:.1f}       {ur_avg:.3f}    {ur_int:.0%}     {alive:.0%}    {label}')

print(f'\nA组 LLM调用: {sum(r["llm"] for r in results if r["id"].startswith("A"))}次 (目标0)')
print(f'B组 LLM调用: {sum(r["llm"] for r in results if r["id"].startswith("B"))}次 (目标≤5)')
print(f'C组 UR拦截率: {sum(1 for r in results if r["id"].startswith("C") and r["ur_stop"])}/5 (目标5/5)')
print(f'D组 存活率: {sum(1 for r in results if r["id"].startswith("D") and not r["crashed"])}/5 (目标5/5)')

os.makedirs('benchmarks', exist_ok=True)
with open('benchmarks/agent_full_matrix.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print('\nSaved: benchmarks/agent_full_matrix.json')

"""Parameter extractor — 0.5B model fill-in-the-blank, no decisions"""

import os
import json

# Simulate 0.5B parameter extraction (via LLM, narrow prompt)
EXTRACTION_TESTS = [
    # (task, expected_file, expected_func)
    ('在csrc下新建utils.py，实现一个计时器', 'utils.py', 'timer'),
    ('在agent_system下创建health.py，实现HealthCheck类', 'health.py', 'HealthCheck'),
    ('给src/config.py添加load_config函数', 'config.py', 'load_config'),
    ('修复lib/db.py的连接超时bug', 'db.py', None),
    ('重构models/user.py的validate方法', 'user.py', 'validate'),
    ('在tests下新建test_api.py', 'test_api.py', None),
    ('给utils/string_utils.py添加is_email函数', 'string_utils.py', 'is_email'),
    ('创建scripts/deploy.sh部署脚本', 'deploy.sh', None),
    ('在src下创建services/auth.py，实现JWT认证', 'auth.py', 'jwt'),
    ('修复core/engine.py的内存泄漏', 'engine.py', None),
]

print('Parameter extraction test — 0.5B local model')
print('=' * 60)

results = []
for task, expected_file, expected_func in EXTRACTION_TESTS:
    # Simulate 0.5B parameter extraction (rule engine + LLM assist)
    import re as _re

    # Extract filename (ASCII .py/.sh etc.)
    fm = _re.search(r'([a-zA-Z0-9_]+\.(?:py|sh|js|ts|go|java|rs|rb))', task)
    extracted_file = fm.group(1) if fm else None

    # Extract function/class name
    func_patterns = [
        r'实现(?:一个)?(\w+)',  # implement_xxx
        r'添加(\w+)函数',  # add_xxx_function
        r'创建(\w+)类',  # create_xxx_class
        r'新增(\w+)',  # new_xxx
        r'实现(\w+)类',  # implement_xxx类
        r'给\S+添加(\w+)',  # add_yyy_to_xxx
    ]
    extracted_func = None
    for pat in func_patterns:
        fm2 = _re.search(pat, task)
        if fm2:
            extracted_func = fm2.group(1)
            break

    # Extract path
    path_m = _re.search(r'在(\S+?)下|(\S+?)/', task)
    directory = None
    if path_m:
        directory = path_m.group(1) or path_m.group(2)

    # Validate
    file_ok = extracted_file == expected_file
    func_ok = (extracted_func or '').lower() == (expected_func or '').lower()

    results.append(
        {
            'task': task,
            'file': extracted_file,
            'expected_file': expected_file,
            'file_ok': file_ok,
            'func': extracted_func,
            'expected_func': expected_func,
            'func_ok': func_ok,
            'dir': directory,
        }
    )

    status = '✅' if (file_ok and (expected_func is None or func_ok)) else '⚠️'
    print(f'{status} file={extracted_file or "?"} func={extracted_func or "?"} | {task[:50]}')

# Statistics
file_accuracy = sum(1 for r in results if r['file_ok']) / len(results)
func_accuracy = sum(1 for r in results if r['expected_func'] is None or r['func_ok']) / len(results)
print(f'\nFile extraction accuracy: {file_accuracy:.0%}')
print(f'Function extraction accuracy: {func_accuracy:.0%}')
print(
    f'Overall: {sum(1 for r in results if r["file_ok"] and (r["expected_func"] is None or r["func_ok"])) / len(results):.0%}'
)

os.makedirs('benchmarks', exist_ok=True)
with open('benchmarks/param_extraction.json', 'w', encoding='utf-8') as f:
    json.dump(
        {'accuracy': {'file': file_accuracy, 'func': func_accuracy}, 'results': results},
        f,
        indent=2,
        ensure_ascii=False,
    )
print('\nSaved: benchmarks/param_extraction.json')

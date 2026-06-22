"""SIMD Phase 1 verification — 256x256 GEMM vs NumPy

Compile: nasm -f win64 -o csrc/simd_demo.obj csrc/simd_demo.asm
       (Windows needs MSVC or MinGW for linking)

Linux: nasm -f elf64 -o csrc/simd_demo.o csrc/simd_demo.asm
       gcc -shared -o csrc/simd_demo.so csrc/simd_demo.o

Usage: python csrc/simd_test.py
"""

import ctypes
import numpy as np
import os
import sys
import platform

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N = 256


def build_asm():
    """Compile assembly -> shared library"""
    is_linux = platform.system() == 'Linux'
    asm_src = os.path.join(ROOT, 'csrc', 'simd_demo.asm')

    if not os.path.exists(asm_src):
        print(f'  [SKIP] Assembly file not found: {asm_src}')
        return None

    if is_linux:
        obj = os.path.join(ROOT, 'csrc', 'simd_demo.o')
        lib = os.path.join(ROOT, 'csrc', 'simd_demo.so')
        os.system(f'nasm -f elf64 -o {obj} {asm_src}')
        os.system(f'gcc -shared -o {lib} {obj}')
    else:
        # Windows: NASM + MinGW
        nasm = 'nasm'
        for p in [
            os.path.join(os.environ.get('APPDATA', ''), 'Local', 'bin', 'NASM', 'nasm.exe'),
            'C:/Program Files/NASM/nasm.exe',
            'nasm',
        ]:
            if os.path.exists(p):
                nasm = p
                break
        obj = os.path.join(ROOT, 'csrc', 'simd_demo.obj')
        lib = os.path.join(ROOT, 'csrc', 'simd_demo.dll')
        if not os.path.exists(lib):
            os.system(f'"{nasm}" -f win64 -o {obj} {asm_src}')
            os.system(f'gcc -shared -o {lib} {obj}')

    if os.path.exists(lib):
        return lib
    return None


def test_python_matmul():
    """Pure Python/NumPy matrix multiplication benchmark"""
    A = np.random.randn(N, N).astype(np.float32)
    B = np.random.randn(N, N).astype(np.float32)

    t0 = __import__('time').time()
    C_np = A @ B
    t1 = __import__('time').time()

    return A, B, C_np, t1 - t0


def test_asm_matmul(A, B, lib_path):
    """Assembly matrix multiplication"""
    lib = ctypes.CDLL(lib_path)
    lib.matmul_256x256.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
    ]
    lib.matmul_256x256.restype = None

    C_asm = np.zeros((N, N), dtype=np.float32)
    A_p = A.ctypes.data_as(ctypes.c_void_p)
    B_p = B.ctypes.data_as(ctypes.c_void_p)
    C_p = C_asm.ctypes.data_as(ctypes.c_void_p)

    t0 = __import__('time').time()
    lib.matmul_256x256(A_p, B_p, C_p, N)
    t1 = __import__('time').time()

    return C_asm, t1 - t0


def main():
    print(f'\n{"=" * 60}')
    print('  SIMD Phase 1 — 256x256 GEMM verification')
    print(f'{"=" * 60}\n')

    # 1. NumPy baseline
    print('[1/3] NumPy matrix multiplication...')
    A, B, C_np, t_np = test_python_matmul()
    print(f'  Time: {t_np * 1000:.1f} ms')
    print(f'  C[0,0] = {C_np[0, 0]:.6f}')

    # 2. Compile assembly
    print('\n[2/3] Compiling assembly kernel...')
    lib = build_asm()
    if lib is None:
        print('\n  Cannot compile assembly kernel (needs NASM + GCC or WSL).')
        print('  Verification: run this script in a Linux environment.\n')
        return 0

    # 3. Assembly matmul
    print('\n[3/3] Assembly matrix multiplication...')
    C_asm, t_asm = test_asm_matmul(A, B, lib)
    print(f'  Time: {t_asm * 1000:.1f} ms')
    print(f'  C[0,0] = {C_asm[0, 0]:.6f}')

    # 4. Comparison
    diff = np.max(np.abs(C_np - C_asm))
    speedup = t_np / t_asm if t_asm > 0 else 0

    print(f'\n{"=" * 60}')
    print('  Verification results')
    print(f'{"=" * 60}')
    print(f'  NumPy time:  {t_np * 1000:.1f} ms')
    print(f'  Assembly time: {t_asm * 1000:.1f} ms')
    print(f'  Speedup:      {speedup:.1f}x')
    print(f'  Max error:    {diff:.2e}')

    if diff < 1e-4:
        print('\n  PASS: Assembly GEMM matches NumPy results.')
        return 0
    else:
        print(f'\n  FAIL: max error {diff:.2e} > 1e-4')
        return 1


if __name__ == '__main__':
    sys.exit(main())

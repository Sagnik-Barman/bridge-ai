"""Compile ddsolver/src/solver.cpp into the shared library ddsolver/__init__.py loads.

    python -m ddsolver.build              # build for this machine (uses every instruction its CPU has)
    python -m ddsolver.build --portable   # build for any x86-64 CPU from about 2009 on (for sharing)
    python -m ddsolver.build --windows    # cross-compile the Windows DLL (needs ziglang; always portable)

Compiler, in order of preference:
  1. zig (pip install ziglang) - one pip install, works the same on Windows, Linux and macOS
  2. g++ or clang++ on the PATH
  3. MSVC cl.exe (from a "Developer Command Prompt")
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "src", "solver.cpp")
LIB_DIR = os.path.join(HERE, "lib")


def _have_zig() -> bool:
    try:
        import ziglang  # noqa: F401
        return True
    except ImportError:
        return False


def build(windows: bool = False, portable: bool = False, verbose: bool = True) -> str:
    os.makedirs(LIB_DIR, exist_ok=True)
    win = windows or sys.platform.startswith("win")
    if win:
        out = os.path.join(LIB_DIR, "dds_scratch.dll")
    elif sys.platform == "darwin":
        out = os.path.join(LIB_DIR, "libdds_scratch.dylib")
    else:
        out = os.path.join(LIB_DIR, "libdds_scratch.so")
    common = ["-O3", "-std=c++17", "-shared", "-fno-exceptions", "-fno-rtti", SRC, "-o", out]

    attempts = []
    if _have_zig():
        cmd = [sys.executable, "-m", "ziglang", "c++"] + common
        if win:
            # any x86-64 CPU from about 2009 on (needs the POPCNT instruction)
            cmd += ["-target", "x86_64-windows-gnu", "-mcpu=x86_64_v2"]
        else:
            cmd += ["-fPIC", "-mcpu=x86_64_v2" if portable else "-mcpu=native"]
        attempts.append(cmd)
    if not windows or sys.platform.startswith("win"):
        for cxx in ("g++", "clang++"):
            if shutil.which(cxx):
                extra = ["-static-libgcc", "-static-libstdc++"] if win else ["-fPIC"]
                extra += ["-march=x86-64-v2" if (portable or windows) else "-march=native"]
                attempts.append([cxx] + common + extra)
        if win and shutil.which("cl"):
            attempts.append(["cl", "/O2", "/LD", "/EHsc", "/std:c++17", SRC, f"/Fe:{out}"])

    if not attempts:
        raise RuntimeError("No C++ compiler found. Run `pip install ziglang` and try again.")
    errors = []
    for cmd in attempts:
        if verbose:
            print("building:", " ".join(cmd))
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode == 0 and os.path.exists(out):
            for junk in ("dds_scratch.pdb", "solver.lib", "dds_scratch.lib"):   # debug/import files zig adds
                if os.path.exists(os.path.join(LIB_DIR, junk)):
                    os.remove(os.path.join(LIB_DIR, junk))
            if verbose:
                print("built", out)
            return out
        errors.append(f"{cmd[0]}: {r.stderr.strip()[:2000]}")
    raise RuntimeError("Build failed:\n" + "\n".join(errors))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", action="store_true", help="cross-compile dds_scratch.dll (needs ziglang)")
    ap.add_argument("--portable", action="store_true", help="any x86-64 CPU, not just this one")
    args = ap.parse_args()
    build(windows=args.windows, portable=args.portable)

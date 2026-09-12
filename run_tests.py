"""Tiny test runner: discovers test_*.py under tests/ and runs every
test_* function, since pytest may not be installed. Not part of the
shipped app; just a dev convenience (pytest works fine too, if present).

Supports skipping: a test (or an entire module, e.g. because an optional
dependency like torch isn't installed in this environment) can raise
unittest.SkipTest to be counted separately from failures rather than
breaking the whole run.
"""
import importlib
import inspect
import pkgutil
import sys
import traceback
import unittest

import tests as tests_pkg


def main() -> int:
    total = 0
    failures = 0
    skipped = 0
    for _, modname, _ in pkgutil.iter_modules(tests_pkg.__path__, tests_pkg.__name__ + "."):
        try:
            mod = importlib.import_module(modname)
        except unittest.SkipTest as exc:
            print(f"SKIP  {modname} (module): {exc}")
            skipped += 1
            continue
        except ImportError as exc:
            print(f"SKIP  {modname} (module, missing optional dependency): {exc}")
            skipped += 1
            continue
        setup = getattr(mod, "setup_function", None)
        teardown = getattr(mod, "teardown_function", None)
        for name, fn in inspect.getmembers(mod, inspect.isfunction):
            if name.startswith("test_"):
                total += 1
                try:
                    if setup:
                        setup(fn)
                    fn()
                    print(f"PASS {modname}.{name}")
                except unittest.SkipTest as exc:
                    total -= 1
                    skipped += 1
                    print(f"SKIP  {modname}.{name}: {exc}")
                except Exception:
                    failures += 1
                    print(f"FAIL {modname}.{name}")
                    traceback.print_exc()
                finally:
                    if teardown:
                        teardown(fn)
    print(f"\n{total - failures}/{total} passed, {failures} failures, {skipped} skipped")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

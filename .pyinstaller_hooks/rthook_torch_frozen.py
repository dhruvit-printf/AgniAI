"""
Runtime hook: patch inspect AND torch._jit_internal BEFORE any torch import.

The crash:
  torch/nn/functional.py line 4448 calls torch._jit_internal._overload()
  _overload() calls _check_overload_body()
  _check_overload_body() calls torch._sources.parse_def()
  parse_def() calls inspect.getsource() on a frozen .pyc file
  → RuntimeError: Expected a single top-level function

Fix 1: Patch inspect.getsource / getsourcelines / findsource to return safe
        fallbacks when the source is unavailable (frozen bundle).

Fix 2: Monkey-patch torch._jit_internal._check_overload_body and _overload
        so they are no-ops inside a frozen bundle. This prevents parse_def()
        from ever being called on bytecode.

Fix 3: Patch torch._sources.parse_def to never raise on frozen source.
"""

# ── Fix 1: inspect patches ────────────────────────────────────────────────
import inspect as _inspect
import os
import sys

_orig_getsource = _inspect.getsource
_orig_getsourcelines = _inspect.getsourcelines
_orig_findsource = _inspect.findsource


def _safe_getsource(obj):
    try:
        return _orig_getsource(obj)
    except (OSError, TypeError):
        return ""


def _safe_getsourcelines(obj):
    try:
        return _orig_getsourcelines(obj)
    except (OSError, TypeError):
        return ([], 0)


def _safe_findsource(obj):
    try:
        return _orig_findsource(obj)
    except (OSError, TypeError):
        return ([], 0)


_inspect.getsource = _safe_getsource
_inspect.getsourcelines = _safe_getsourcelines
_inspect.findsource = _safe_findsource


# ── Fix 2: pre-patch torch._jit_internal before torch loads ──────────────
# We install an import hook that intercepts torch._jit_internal at import
# time and replaces the problematic functions with safe no-ops.


class _TorchJitPatcher:
    """Import hook that patches torch._jit_internal immediately on import."""

    def find_module(self, fullname, path=None):
        if fullname in (
            "torch._jit_internal",
            "torch._sources",
            "torch.jit._builtins",
        ):
            return self
        return None

    def load_module(self, fullname):
        if fullname in sys.modules:
            return sys.modules[fullname]

        # Remove ourselves temporarily to allow the real import
        sys.meta_path = [h for h in sys.meta_path if h is not self]
        try:
            import importlib

            mod = importlib.import_module(fullname)
        finally:
            # Re-insert ourselves at the front
            sys.meta_path.insert(0, self)

        # ── Patch torch._jit_internal ────────────────────────────────────
        if fullname == "torch._jit_internal":

            def _safe_check_overload_body(fn):
                # Original raises RuntimeError when source can't be parsed.
                # In a frozen bundle we just skip the check.
                return

            def _safe_overload(fn):
                # Original calls _check_overload_body; we make it a no-op
                # decorator that returns the function unchanged.
                return fn

            if hasattr(mod, "_check_overload_body"):
                mod._check_overload_body = _safe_check_overload_body
            if hasattr(mod, "_overload"):
                mod._overload = _safe_overload

        # ── Patch torch._sources.parse_def ──────────────────────────────
        if fullname == "torch._sources":
            _orig_parse_def = getattr(mod, "parse_def", None)

            def _safe_parse_def(src):
                try:
                    if _orig_parse_def is not None:
                        return _orig_parse_def(src)
                except (RuntimeError, OSError, TypeError):
                    # Return a minimal fake AST-like result so callers don't crash
                    return None

            if _orig_parse_def is not None:
                mod.parse_def = _safe_parse_def

        sys.modules[fullname] = mod
        return mod


# Register the patcher as early as possible (position 0 = highest priority)
sys.meta_path.insert(0, _TorchJitPatcher())


# ── Fix 3: environment variables that suppress torch JIT compilation ──────
# These tell PyTorch not to attempt JIT compilation at import time.
os.environ.setdefault("PYTORCH_JIT", "0")
os.environ.setdefault("TORCH_JIT_DISABLE", "1")

# Prevent torch.distributed from calling inspect.getsource at module level
os.environ.setdefault("TORCH_DISTRIBUTED_DEBUG", "OFF")

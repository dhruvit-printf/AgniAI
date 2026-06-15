"""
Runtime hook: patch inspect.getsource/getsourcelines/findsource
to return safe fallbacks when called on frozen modules.
torch.distributed.config calls inspect.getsource() at import time,
which crashes inside a PyInstaller bundle where no source files exist.
"""
import inspect

_orig_getsource = inspect.getsource
_orig_getsourcelines = inspect.getsourcelines
_orig_findsource = inspect.findsource

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

inspect.getsource = _safe_getsource
inspect.getsourcelines = _safe_getsourcelines
inspect.findsource = _safe_findsource
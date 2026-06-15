# .pyinstaller_hooks/hook-torch.py
#
# KEY CHANGE vs original:
#   Added `module_collection_mode` dict to force torch.nn.functional (and
#   other files that use @_overload at module level) to be collected as
#   bytecode-only (.pyc) rather than frozen source.  When PyInstaller
#   freezes a file as source, PyTorch's JIT parser finds the file and
#   tries to parse it — which fails because frozen "source" is actually
#   compiled bytecode.  Bytecode-only mode hides the file from the parser.

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

binaries = collect_dynamic_libs("torch")

datas = collect_data_files("torch", excludes=[
    "test",
    "tests",
    "utils/tensorboard",
])

hiddenimports = [
    "torch",
    "torch.nn",
    "torch.nn.functional",
    "torch.nn.modules",
    "torch.nn.modules.linear",
    "torch.nn.modules.normalization",
    "torch.nn.modules.activation",
    "torch.nn.modules.pooling",
    "torch.cuda",
    "torch.jit",
    "torch._C",
    "torch._tensor",
    "torch.storage",
    "torch.serialization",
    "torch.utils",
    "torch.utils.data",
    "torch.utils.data.dataloader",
    "torch.distributed",
    "torch.distributed.distributed_c10d",
    "torch._jit_internal",
    "torch._sources",
    "torch.jit._builtins",
    "torch.jit.annotations",
    "torch.utils._config_module",
]

# ── Force bytecode-only collection for modules that use @_overload ────────
# This prevents PyTorch's JIT source parser from ever seeing these files
# as text, which is what causes:
#   RuntimeError: Expected a single top-level function: torch/nn/functional.py:1
#
# 'bytecode' mode = collect as .pyc, no frozen source text in the archive.
module_collection_mode = {
    "torch.nn.functional":              "bytecode",
    "torch.nn.modules.activation":      "bytecode",
    "torch.nn.modules.linear":          "bytecode",
    "torch.nn.modules.normalization":   "bytecode",
    "torch.nn.modules.pooling":         "bytecode",
    "torch.nn.modules.sparse":          "bytecode",
    "torch.nn.modules.conv":            "bytecode",
    "torch.nn.modules.rnn":             "bytecode",
    "torch._jit_internal":              "bytecode",
    "torch._sources":                   "bytecode",
    "torch.jit._builtins":              "bytecode",
    "torch.jit.annotations":            "bytecode",
    "torch.functional":                 "bytecode",
    "torch.nn.parallel.distributed":    "bytecode",
    "torch.distributed.distributed_c10d": "bytecode",
    "torch.utils._config_module":       "bytecode",
}
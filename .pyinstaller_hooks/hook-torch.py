# hook-torch.py
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

binaries = collect_dynamic_libs("torch")

datas = collect_data_files("torch", excludes=[
    "test",
    "tests",
    "distributed",
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
]
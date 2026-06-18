# .pyinstaller_hooks/hook-torch.py

from PyInstaller.utils.hooks import (collect_data_files, collect_dynamic_libs,
                                     collect_submodules)

binaries = collect_dynamic_libs("torch")

datas = collect_data_files(
    "torch",
    excludes=[
        "test",
        "tests",
        "utils/tensorboard",
    ],
)

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
    # dynamo + all polyfills
    "torch._dynamo",
    "torch._dynamo.polyfills",
    "torch._dynamo.polyfills.loader",
    "torch._dynamo.polyfills._collections",
    "torch._dynamo.polyfills.builtins",
    "torch._dynamo.polyfills.copy",
    "torch._dynamo.polyfills.functools",
    "torch._dynamo.polyfills.fx",
    "torch._dynamo.polyfills.heapq",
    "torch._dynamo.polyfills.itertools",
    "torch._dynamo.polyfills.operator",
    "torch._dynamo.polyfills.os",
    "torch._dynamo.polyfills.pytree",
    "torch._dynamo.polyfills.struct",
    "torch._dynamo.polyfills.sys",
    "torch._dynamo.polyfills.tensor",
    "torch._dynamo.polyfills.torch_c_nn",
    "torch._dynamo.polyfills.traceback",
]

# "pyc" = external .pyc, no frozen source text — prevents parse_def() crash
module_collection_mode = {
    "torch.nn.functional": "pyc",
    "torch.nn.modules.activation": "pyc",
    "torch.nn.modules.linear": "pyc",
    "torch.nn.modules.normalization": "pyc",
    "torch.nn.modules.pooling": "pyc",
    "torch.nn.modules.sparse": "pyc",
    "torch.nn.modules.conv": "pyc",
    "torch.nn.modules.rnn": "pyc",
    "torch._jit_internal": "pyc",
    "torch._sources": "pyc",
    "torch.jit._builtins": "pyc",
    "torch.jit.annotations": "pyc",
    "torch.functional": "pyc",
    "torch.nn.parallel.distributed": "pyc",
    "torch.distributed.distributed_c10d": "pyc",
    "torch.utils._config_module": "pyc",
    "torch._dynamo": "pyc",
}

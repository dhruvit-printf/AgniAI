# -*- mode: python ; coding: utf-8 -*-
# AgniAI PyInstaller build spec
# Run with:  pyinstaller agniai.spec --clean --noconfirm

block_cipher = None

from PyInstaller.utils.hooks import (
    collect_all,
    collect_data_files,
    collect_dynamic_libs,
    collect_submodules,
    copy_metadata,
)

# ── Package metadata ───────────────────────────────────────────────────────
def safe_copy_metadata(package_name):
    try:
        return copy_metadata(package_name)
    except Exception:
        return []

meta_datas = (
    safe_copy_metadata("anyio")
    + safe_copy_metadata("beautifulsoup4")
    + safe_copy_metadata("certifi")
    + safe_copy_metadata("charset-normalizer")
    + safe_copy_metadata("click")
    + safe_copy_metadata("colorama")
    + safe_copy_metadata("Deprecated")
    + safe_copy_metadata("faiss-cpu")
    + safe_copy_metadata("filelock")
    + safe_copy_metadata("Flask")
    + safe_copy_metadata("flask-cors")
    + safe_copy_metadata("Flask-Limiter")
    + safe_copy_metadata("fsspec")
    + safe_copy_metadata("huggingface_hub")
    + safe_copy_metadata("idna")
    + safe_copy_metadata("itsdangerous")
    + safe_copy_metadata("Jinja2")
    + safe_copy_metadata("joblib")
    + safe_copy_metadata("limits")
    + safe_copy_metadata("lxml")
    + safe_copy_metadata("MarkupSafe")
    + safe_copy_metadata("mpmath")
    + safe_copy_metadata("networkx")
    + safe_copy_metadata("numpy")
    + safe_copy_metadata("packaging")
    + safe_copy_metadata("psutil")
    + safe_copy_metadata("PyMuPDF")
    + safe_copy_metadata("python-docx")
    + safe_copy_metadata("python-dotenv")
    + safe_copy_metadata("PyYAML")
    + safe_copy_metadata("rank-bm25")
    + safe_copy_metadata("regex")
    + safe_copy_metadata("requests")
    + safe_copy_metadata("safetensors")
    + safe_copy_metadata("scikit-learn")
    + safe_copy_metadata("scipy")
    + safe_copy_metadata("sentence-transformers")
    + safe_copy_metadata("sympy")
    + safe_copy_metadata("threadpoolctl")
    + safe_copy_metadata("tokenizers")
    + safe_copy_metadata("torch")
    + safe_copy_metadata("tqdm")
    + safe_copy_metadata("transformers")
    + safe_copy_metadata("typing_extensions")
    + safe_copy_metadata("urllib3")
    + safe_copy_metadata("Werkzeug")
    + safe_copy_metadata("wrapt")
)

# ── Collect full package data + binaries + hidden imports ──────────────────
regex_datas,        regex_binaries,        regex_hiddenimports        = collect_all("regex")
transformers_datas, transformers_binaries, transformers_hiddenimports = collect_all("transformers")
tokenizers_datas,   tokenizers_binaries,   tokenizers_hiddenimports   = collect_all("tokenizers")
senttr_datas,       senttr_binaries,       senttr_hiddenimports       = collect_all("sentence_transformers")
huggingface_datas,  huggingface_binaries,  huggingface_hiddenimports  = collect_all("huggingface_hub")
safetensors_datas,  safetensors_binaries,  safetensors_hiddenimports  = collect_all("safetensors")
docx_datas,         docx_binaries,         docx_hiddenimports         = collect_all("docx")
fitz_datas,         fitz_binaries,         fitz_hiddenimports         = collect_all("fitz")
faiss_datas,        faiss_binaries,        faiss_hiddenimports        = collect_all("faiss")

# ── NEW: collect torch._dynamo fully (fixes polyfills.copy and friends) ───
dynamo_datas,       dynamo_binaries,       dynamo_hiddenimports       = collect_all("torch._dynamo")

hidden_imports = [
    # Flask ecosystem
    "flask", "flask.json", "flask_cors", "flask_limiter", "flask_limiter.util",
    "werkzeug", "werkzeug.middleware.proxy_fix", "werkzeug.utils",
    "jinja2", "click", "itsdangerous",

    # Sentence Transformers
    "sentence_transformers", "sentence_transformers.util",
    "sentence_transformers.cross_encoder",
    "sentence_transformers.backend", "sentence_transformers.backend.load",

    # HuggingFace / Transformers core
    "huggingface_hub", "transformers",
    "transformers.modeling_utils",          # fixes PreTrainedModel import error
    "transformers.models.auto",
    "transformers.models.bert.modeling_bert",
    "transformers.models.roberta.modeling_roberta",
    "transformers.utils", "transformers.utils.versions",
    "transformers.dependency_versions_check",
    # transformers.integrations chain that triggered dynamo import
    "transformers.integrations",
    "transformers.integrations.finegrained_fp8",
    "transformers.integrations.moe",
    "transformers.integrations.sonicmoe",
    "tokenizers", "safetensors", "safetensors.torch",

    # FAISS
    "faiss",

    # NumPy / SciPy / sklearn
    "numpy", "numpy.core", "numpy.core._multiarray_umath",
    "scipy", "scipy.sparse",
    "sklearn", "sklearn.metrics.pairwise", "joblib", "threadpoolctl",

    # PyYAML / sympy / misc torch deps
    "yaml", "sympy", "mpmath", "networkx", "fsspec", "typing_extensions",

    # PyMuPDF
    "fitz", "fitz.utils",

    # python-docx
    "docx", "docx.oxml", "docx.oxml.ns", "docx.oxml.table",
    "docx.oxml.text", "docx.oxml.text.paragraph", "docx.oxml.text.run",
    "docx.oxml.document", "docx.oxml.shared", "docx.oxml.styles",
    "docx.parts", "docx.parts.document", "docx.parts.image",
    "docx.shared", "docx.styles", "docx.styles.style",
    "docx.table", "docx.text", "docx.text.paragraph", "docx.text.run",
    "docx.enum", "docx.enum.text", "docx.enum.style", "docx.enum.table",
    "docx.image", "docx.image.image",

    # lxml
    "lxml", "lxml.etree",

    # BM25 / requests / networking
    "rank_bm25", "requests", "urllib3", "certifi",
    "charset_normalizer", "chardet", "idna",

    # BeautifulSoup
    "bs4", "bs4.builder", "bs4.builder._htmlparser",
    "bs4.builder._lxml", "bs4.formatter",

    # dotenv
    "dotenv",

    # Torch core
    "torch", "torch.nn", "torch.nn.functional",
    "torch.nn.modules", "torch.nn.modules.module",
    "torch.cuda", "torch.jit", "torch.jit._builtins", "torch.jit.annotations",
    "torch._C", "torch._tensor", "torch._jit_internal", "torch._sources",
    "torch.storage", "torch.serialization",
    "torch.utils", "torch.utils.data", "torch.utils.data.dataloader",
    "torch.utils._config_module",
    "torch.distributed", "torch.distributed.distributed_c10d",
    "torch.distributed.device_mesh", "torch.distributed.config",

    # torch._dynamo — ALL polyfills explicitly listed
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

    # regex
    "regex", "regex._regex", "regex._regex_core",

    # stdlib helpers
    "subprocess", "shutil", "tempfile",
    "importlib.metadata", "importlib_metadata",
    "packaging", "packaging.version",
    "packaging.requirements", "packaging.specifiers",

    # misc
    "colorama", "wrapt", "psutil", "tqdm", "filelock", "anyio", "rich", "system_messages",
]

datas = [
    ("static/swagger.json", "static"),
    (".env.example", "."),
]

# ── Merge all collected datas ──────────────────────────────────────────────
all_datas = (
    meta_datas + datas
    + regex_datas + transformers_datas + tokenizers_datas
    + senttr_datas + huggingface_datas + safetensors_datas
    + docx_datas + fitz_datas + faiss_datas
    + dynamo_datas
)

all_binaries = (
    regex_binaries + transformers_binaries + tokenizers_binaries
    + senttr_binaries + huggingface_binaries + safetensors_binaries
    + docx_binaries + fitz_binaries + faiss_binaries
    + dynamo_binaries
)

all_hidden_imports = (
    hidden_imports
    + regex_hiddenimports + transformers_hiddenimports + tokenizers_hiddenimports
    + senttr_hiddenimports + huggingface_hiddenimports + safetensors_hiddenimports
    + docx_hiddenimports + fitz_hiddenimports + faiss_hiddenimports
    + dynamo_hiddenimports
)

# ── Module collection mode ─────────────────────────────────────────────────
# "pyc" = external .pyc file on disk (no source text in PYZ archive).
# Prevents torch._jit_internal.parse_def() from finding a "source" file
# and trying to parse frozen bytecode as Python text.
# Valid PyInstaller 6.x modes: pyz | pyc | py | pyz+py
_module_collection_mode = {
    # torch modules that use @_overload at import time
    "torch.nn.functional":                "pyc",
    "torch.nn.modules.activation":        "pyc",
    "torch.nn.modules.linear":            "pyc",
    "torch.nn.modules.normalization":     "pyc",
    "torch.nn.modules.pooling":           "pyc",
    "torch.nn.modules.sparse":            "pyc",
    "torch.nn.modules.conv":              "pyc",
    "torch.nn.modules.rnn":               "pyc",
    "torch._jit_internal":                "pyc",
    "torch._sources":                     "pyc",
    "torch.jit._builtins":                "pyc",
    "torch.jit.annotations":              "pyc",
    "torch.functional":                   "pyc",
    "torch.nn.parallel.distributed":      "pyc",
    "torch.distributed.distributed_c10d": "pyc",
    "torch.utils._config_module":         "pyc",
    # torch._dynamo — collect all as pyc so no source parsing occurs
    "torch._dynamo":                      "pyc",
    # transformers modules that fail similarly
    "transformers.generation.logits_process":      "pyc",
    "transformers.generation.configuration_utils": "pyc",
    "transformers.configuration_utils":            "pyc",
}

a = Analysis(
    ["app_launcher.py"],
    pathex=["."],
    binaries=all_binaries,
    datas=all_datas,
    hiddenimports=all_hidden_imports,
    hookspath=[".pyinstaller_hooks"],
    hooksconfig={},
    runtime_hooks=[".pyinstaller_hooks/rthook_torch_frozen.py"],
    excludes=[
        "tkinter", "wx", "PyQt5", "PyQt6", "PySide2", "PySide6",
        "matplotlib", "IPython", "notebook", "pytest",
        "torch.utils.tensorboard", "tensorboard",
        "setuptools", "distutils", "pip",
        "pandas", "cv2",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
    module_collection_mode=_module_collection_mode,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="agniai",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="agniai",
)

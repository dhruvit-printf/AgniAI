# -*- mode: python ; coding: utf-8 -*-
# AgniAI PyInstaller build spec
# Run with:  pyinstaller agniai.spec --clean --noconfirm
# Built against agniai-env — pip list verified 2026-06-15

block_cipher = None

from PyInstaller.utils.hooks import (
    collect_all,
    collect_data_files,
    collect_dynamic_libs,
    copy_metadata,
)

# ── Package metadata ───────────────────────────────────────────────────────
# Every name below is the EXACT pip name from `pip list` in agniai-env.
# copy_metadata() will crash the build if the name doesn't match — no guessing.
meta_datas = (
    copy_metadata("anyio")
    + copy_metadata("beautifulsoup4")
    + copy_metadata("certifi")
    + copy_metadata("charset-normalizer")
    + copy_metadata("click")
    + copy_metadata("colorama")
    + copy_metadata("Deprecated")
    + copy_metadata("faiss-cpu")
    + copy_metadata("filelock")
    + copy_metadata("Flask")
    + copy_metadata("flask-cors")
    + copy_metadata("Flask-Limiter")
    + copy_metadata("flask-swagger-ui")
    + copy_metadata("fsspec")
    + copy_metadata("huggingface_hub")
    + copy_metadata("idna")
    + copy_metadata("itsdangerous")
    + copy_metadata("Jinja2")
    + copy_metadata("joblib")
    + copy_metadata("limits")
    + copy_metadata("lxml")
    + copy_metadata("MarkupSafe")
    + copy_metadata("mpmath")
    + copy_metadata("networkx")
    + copy_metadata("numpy")
    + copy_metadata("packaging")
    + copy_metadata("psutil")
    + copy_metadata("PyMuPDF")
    + copy_metadata("python-docx")
    + copy_metadata("python-dotenv")
    + copy_metadata("PyYAML")
    + copy_metadata("rank-bm25")
    + copy_metadata("regex")
    + copy_metadata("requests")
    + copy_metadata("safetensors")
    + copy_metadata("scikit-learn")
    + copy_metadata("scipy")
    + copy_metadata("sentence-transformers")
    + copy_metadata("sympy")
    + copy_metadata("threadpoolctl")
    + copy_metadata("tokenizers")
    + copy_metadata("torch")
    + copy_metadata("tqdm")
    + copy_metadata("transformers")
    + copy_metadata("typing_extensions")
    + copy_metadata("urllib3")
    + copy_metadata("Werkzeug")
    + copy_metadata("wrapt")
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

hidden_imports = [
    # Flask ecosystem
    "flask",
    "flask.json",
    "flask_cors",
    "flask_limiter",
    "flask_limiter.util",
    "werkzeug",
    "werkzeug.middleware.proxy_fix",
    "werkzeug.utils",
    "jinja2",
    "click",
    "itsdangerous",

    # Sentence Transformers
    "sentence_transformers",
    "sentence_transformers.util",
    "sentence_transformers.cross_encoder",
    "sentence_transformers.backend",
    "sentence_transformers.backend.load",

    # HuggingFace / Transformers
    "huggingface_hub",
    "transformers",
    "transformers.models.auto",
    "transformers.models.bert.modeling_bert",
    "transformers.models.roberta.modeling_roberta",
    "transformers.utils",
    "transformers.utils.versions",
    "transformers.dependency_versions_check",
    "tokenizers",
    "safetensors",
    "safetensors.torch",

    # FAISS
    "faiss",

    # NumPy
    "numpy",
    "numpy.core",
    "numpy.core._multiarray_umath",

    # SciPy
    "scipy",
    "scipy.sparse",

    # scikit-learn
    "sklearn",
    "sklearn.metrics.pairwise",
    "joblib",
    "threadpoolctl",

    # PyYAML — transformers require_version checks this at startup
    "yaml",

    # sympy / torch deps
    "sympy",
    "mpmath",
    "networkx",
    "fsspec",

    # typing
    "typing_extensions",

    # PyMuPDF
    "fitz",
    "fitz.fitz",
    "fitz.utils",

    # python-docx
    "docx",
    "docx.oxml",
    "docx.oxml.ns",
    "docx.oxml.table",
    "docx.oxml.text",
    "docx.oxml.text.paragraph",
    "docx.oxml.text.run",
    "docx.oxml.document",
    "docx.oxml.shared",
    "docx.oxml.styles",
    "docx.parts",
    "docx.parts.document",
    "docx.parts.image",
    "docx.shared",
    "docx.styles",
    "docx.styles.style",
    "docx.table",
    "docx.text",
    "docx.text.paragraph",
    "docx.text.run",
    "docx.enum",
    "docx.enum.text",
    "docx.enum.style",
    "docx.enum.table",
    "docx.image",
    "docx.image.image",

    # lxml (used by docx and bs4)
    "lxml",
    "lxml.etree",

    # BM25
    "rank_bm25",

    # Requests / networking
    "requests",
    "urllib3",
    "certifi",
    "charset_normalizer",
    "chardet",
    "idna",

    # BeautifulSoup
    "bs4",
    "bs4.builder",
    "bs4.builder._htmlparser",
    "bs4.builder._lxml",
    "bs4.formatter",

    # dotenv
    "dotenv",

    # Torch
    "torch",
    "torch.nn",
    "torch.nn.functional",
    "torch.nn.modules",
    "torch.nn.modules.module",
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
    "torch.distributed.device_mesh",
    "torch.distributed.config",
    "torch.utils._config_module",

    # regex
    "regex",
    "regex._regex",
    "regex._regex_core",

    # stdlib helpers
    "subprocess",
    "shutil",
    "tempfile",

    # importlib metadata
    "importlib.metadata",
    "importlib_metadata",
    "packaging",
    "packaging.version",
    "packaging.requirements",
    "packaging.specifiers",

    # colorama (Windows color output)
    "colorama",

    # wrapt / Deprecated
    "wrapt",

    # Other
    "yaml",
    "psutil",
    "tqdm",
    "filelock",
    "anyio",
    "rich",
]

datas = [
    ("static/swagger.json", "static"),
    (".env.example", "."),
]

# ── Merge all collected datas ──────────────────────────────────────────────
all_datas = (
    meta_datas
    + datas
    + regex_datas
    + transformers_datas
    + tokenizers_datas
    + senttr_datas
    + huggingface_datas
    + safetensors_datas
    + docx_datas
    + fitz_datas
    + faiss_datas
)

# ── Merge all collected binaries ───────────────────────────────────────────
all_binaries = (
    regex_binaries
    + transformers_binaries
    + tokenizers_binaries
    + senttr_binaries
    + huggingface_binaries
    + safetensors_binaries
    + docx_binaries
    + fitz_binaries
    + faiss_binaries
)

# ── Merge all collected hidden imports ─────────────────────────────────────
all_hidden_imports = (
    hidden_imports
    + regex_hiddenimports
    + transformers_hiddenimports
    + tokenizers_hiddenimports
    + senttr_hiddenimports
    + huggingface_hiddenimports
    + safetensors_hiddenimports
    + docx_hiddenimports
    + fitz_hiddenimports
    + faiss_hiddenimports
)

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
        "torch.utils.tensorboard",
        "tensorboard",
        "setuptools", "distutils", "pip",
        "pandas", "cv2",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
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

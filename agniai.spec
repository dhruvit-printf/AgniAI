# -*- mode: python ; coding: utf-8 -*-
# AgniAI PyInstaller build spec
# Run with:  pyinstaller agniai.spec --clean --noconfirm

block_cipher = None

from PyInstaller.utils.hooks import (
    collect_all,
    collect_data_files,
    collect_dynamic_libs,
    copy_metadata,
)

# ── Package metadata (fixes importlib.metadata.PackageNotFoundError) ───────
# transformers calls require_version() at import time which reads .dist-info.
# PyInstaller doesn't copy .dist-info by default — copy_metadata fixes this.
# NOTE: metadata keys must match the pip package name (use hyphens, not underscores).
meta_datas = (
    copy_metadata("packaging")
    + copy_metadata("transformers")
    + copy_metadata("sentence-transformers")   # pip: sentence-transformers
    + copy_metadata("tokenizers")
    + copy_metadata("huggingface-hub")         # pip: huggingface-hub
    + copy_metadata("safetensors")
    + copy_metadata("tqdm")
    + copy_metadata("numpy")
    + copy_metadata("filelock")
    + copy_metadata("requests")
    + copy_metadata("regex")
    + copy_metadata("flask")
    + copy_metadata("flask-cors")              # pip: flask-cors
    + copy_metadata("flask-limiter")           # pip: flask-limiter
    + copy_metadata("werkzeug")
    + copy_metadata("click")
    + copy_metadata("itsdangerous")
    + copy_metadata("jinja2")
    + copy_metadata("rank-bm25")               # pip: rank-bm25
    + copy_metadata("faiss-cpu")               # pip: faiss-cpu (not "faiss")
    + copy_metadata("python-dotenv")           # pip: python-dotenv
    + copy_metadata("beautifulsoup4")
    + copy_metadata("psutil")
    + copy_metadata("certifi")
    + copy_metadata("charset-normalizer")      # pip: charset-normalizer
    + copy_metadata("urllib3")
)

# ── Collect full package data + metadata for packages that need it ─────────
regex_datas,        regex_binaries,        regex_hiddenimports        = collect_all("regex")
transformers_datas, transformers_binaries, transformers_hiddenimports = collect_all("transformers")
tokenizers_datas,   tokenizers_binaries,   tokenizers_hiddenimports   = collect_all("tokenizers")
senttr_datas,       senttr_binaries,       senttr_hiddenimports       = collect_all("sentence_transformers")
huggingface_datas,  huggingface_binaries,  huggingface_hiddenimports  = collect_all("huggingface_hub")
safetensors_datas,  safetensors_binaries,  safetensors_hiddenimports  = collect_all("safetensors")
docx_datas,         docx_binaries,         docx_hiddenimports         = collect_all("docx")
fitz_datas,         fitz_binaries,         fitz_hiddenimports         = collect_all("fitz")
# FIX: faiss was collected but never merged into all_* lists — fixed below
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

    # PyMuPDF — PDF + legacy .doc fallback
    "fitz",
    "fitz.fitz",
    "fitz.utils",

    # python-docx — full submodule tree
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
    "bs4.formatter",

    # dotenv
    "dotenv",

    # Torch
    "torch",
    "torch.nn",
    "torch.nn.functional",
    "torch.cuda",

    # regex — must be explicit
    "regex",
    "regex._regex",
    "regex._regex_core",

    # stdlib helpers used by .doc conversion path
    "subprocess",
    "shutil",
    "tempfile",

    # importlib metadata — needed by packaging / transformers
    "importlib.metadata",
    "importlib_metadata",
    "packaging",
    "packaging.version",
    "packaging.requirements",
    "packaging.specifiers",

    # Other
    "psutil",
    "tqdm",
    "filelock",
    "PIL",
    "PIL.Image",
]

datas = [
    ("static/swagger.json", "static"),
    (".env.example", "."),
]

# ── Merge all collected datas ──────────────────────────────────────────────
all_datas = (
    meta_datas          # ← .dist-info directories (CRITICAL for transformers)
    + datas
    + regex_datas
    + transformers_datas
    + tokenizers_datas
    + senttr_datas
    + huggingface_datas
    + safetensors_datas
    + docx_datas
    + fitz_datas
    + faiss_datas       # FIX: was collected but never merged
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
    + faiss_binaries    # FIX: was collected but never merged
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
    + faiss_hiddenimports   # FIX: was collected but never merged
)

a = Analysis(
    ["app_launcher.py"],
    pathex=["."],
    binaries=all_binaries,
    datas=all_datas,
    hiddenimports=all_hidden_imports,
    hookspath=[".pyinstaller_hooks"],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "tkinter", "wx", "PyQt5", "PyQt6", "PySide2", "PySide6",
        "matplotlib", "IPython", "notebook", "pytest",
        "torch.distributed",
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

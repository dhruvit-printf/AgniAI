# -*- mode: python ; coding: utf-8 -*-
# AgniAI PyInstaller build spec
# Run with:  pyinstaller agniai.spec

import sys
from pathlib import Path

block_cipher = None

# ── Collect all hidden imports that PyInstaller misses ────────────────────
hidden_imports = [
    # Flask ecosystem
    "flask", "flask.json", "flask_cors", "flask_limiter",
    "flask_limiter.util", "werkzeug", "werkzeug.middleware.proxy_fix",
    "werkzeug.utils", "jinja2", "click", "itsdangerous",

    # Sentence Transformers + HuggingFace
    "sentence_transformers", "sentence_transformers.models",
    "sentence_transformers.losses", "sentence_transformers.evaluation",
    "transformers", "transformers.models.bert",
    "transformers.models.roberta", "huggingface_hub",
    "tokenizers", "safetensors",

    # FAISS
    "faiss", "faiss.swigfaiss",

    # NumPy / SciPy
    "numpy", "numpy.core", "numpy.core._multiarray_umath",
    "scipy", "scipy.sparse",

    # PyMuPDF
    "fitz",

    # python-docx
    "docx",

    # BM25
    "rank_bm25",

    # Requests / networking
    "requests", "urllib3", "certifi", "charset_normalizer", "idna",

    # BeautifulSoup
    "bs4", "bs4.builder", "bs4.formatter",

    # Storage / serialisation
    "pickle", "json", "hashlib", "threading", "queue",

    # dotenv
    "dotenv", "python_dotenv",

    # Torch (required by sentence-transformers)
    "torch", "torch.nn", "torch.nn.functional",

    # Other
    "psutil", "tqdm", "packaging", "filelock", "regex",
    "PIL", "Pillow",
]

# ── Data files to bundle ──────────────────────────────────────────────────
# Format: (source_path, dest_folder_inside_bundle)
datas = [
    # Swagger UI JSON spec
    ("static/swagger.json", "static"),

    # .env.example so the user knows what to configure
    (".env.example", "."),

    # NOTE: data/ and index/ folders are created automatically at runtime
    # by app_launcher.py — no need to bundle them here.
]

# ── Analysis ──────────────────────────────────────────────────────────────
a = Analysis(
    ["app_launcher.py"],            # entry point (wrapper around app.py)
    pathex=["."],
    binaries=[],
    datas=datas,
    hiddenimports=hidden_imports,
    hookspath=[".pyinstaller_hooks"],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "matplotlib", "IPython", "notebook", "pytest",
        "setuptools", "distutils", "test", "tests",
        "tkinter", "wx", "PyQt5", "PyQt6",
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
    exclude_binaries=True,          # onedir mode — faster startup, easier to debug
    name="agniai",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,                       # compress binaries (requires UPX installed)
    console=True,                   # keep console for log output
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,                      # set to "icon.ico" if you have one
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="agniai",                  # output folder: dist/agniai/
)

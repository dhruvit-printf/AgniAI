# -*- mode: python ; coding: utf-8 -*-
# AgniAI PyInstaller build spec
# Run with:  pyinstaller agniai.spec

block_cipher = None

hidden_imports = [
    # Flask ecosystem
    "flask", "flask.json", "flask_cors", "flask_limiter",
    "flask_limiter.util", "werkzeug", "werkzeug.middleware.proxy_fix",
    "werkzeug.utils", "jinja2", "click", "itsdangerous",

    # Sentence Transformers (correct submodule names)
    "sentence_transformers",
    "sentence_transformers.models.Transformer",
    "sentence_transformers.models.Pooling",
    "sentence_transformers.models.Dense",
    "sentence_transformers.models.Normalize",
    "sentence_transformers.cross_encoder",
    "sentence_transformers.util",

    # HuggingFace
    "huggingface_hub",
    "transformers",
    "transformers.models.auto",
    "transformers.models.bert.modeling_bert",
    "transformers.models.roberta.modeling_roberta",
    "tokenizers",
    "safetensors",
    "safetensors.torch",

    # FAISS
    "faiss",

    # NumPy
    "numpy",
    "numpy.core",
    "numpy.core._multiarray_umath",
    "numpy.core._multiarray_umath",

    # SciPy
    "scipy",
    "scipy.sparse",

    # PyMuPDF
    "fitz",

    # python-docx
    "docx",
    "docx.oxml",
    "docx.oxml.ns",

    # BM25
    "rank_bm25",

    # Requests / networking
    "requests",
    "urllib3",
    "certifi",
    "charset_normalizer",
    "idna",

    # BeautifulSoup
    "bs4",
    "bs4.builder",
    "bs4.builder._htmlparser",
    "bs4.builder._lxml",
    "bs4.formatter",

    # dotenv (correct import name)
    "dotenv",

    # Torch (only what's actually needed)
    "torch",
    "torch.nn",
    "torch.nn.functional",
    "torch.cuda",

    # Other
    "psutil",
    "tqdm",
    "packaging",
    "filelock",
    "regex",
    "PIL",
    "PIL.Image",
]

datas = [
    ("static/swagger.json", "static"),
    (".env.example", "."),
]

a = Analysis(
    ["app_launcher.py"],
    pathex=["."],
    binaries=[],
    datas=datas,
    hiddenimports=hidden_imports,
    hookspath=[".pyinstaller_hooks"],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # GUI toolkits — not needed
        "tkinter", "wx", "PyQt5", "PyQt6", "PySide2", "PySide6",
        # Dev / test tools
        "matplotlib", "IPython", "notebook", "pytest", "pytest_cov",
        # Torch distributed (huge, not needed for inference)
        "torch.distributed",
        "torch.utils.tensorboard",
        "tensorboard",
        # Build tools
        "setuptools", "distutils", "pip",
        # Other heavy unused
        "pandas", "sklearn", "cv2",
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

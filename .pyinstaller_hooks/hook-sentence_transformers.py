# .pyinstaller_hooks/hook-sentence_transformers.py

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

datas = collect_data_files("sentence_transformers")
hiddenimports = collect_submodules("sentence_transformers")

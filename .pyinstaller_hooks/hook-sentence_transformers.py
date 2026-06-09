# .pyinstaller_hooks/hook-transformers.py

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

datas = collect_data_files("transformers")
hiddenimports = collect_submodules("transformers")
# .pyinstaller_hooks/hook-torch.py
# Torch has many dynamic imports — collect all submodules

from PyInstaller.utils.hooks import collect_data_files, collect_submodules, collect_dynamic_libs

datas = collect_data_files("torch")
hiddenimports = collect_submodules("torch")
binaries = collect_dynamic_libs("torch")
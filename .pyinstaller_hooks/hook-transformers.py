# .pyinstaller_hooks/hook-transformers.py

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

datas = collect_data_files("transformers")
hiddenimports = collect_submodules("transformers")

# "pyc" mode: no frozen source text — prevents parse_def() failures
module_collection_mode = {
    "transformers.generation.logits_process": "pyc",
    "transformers.generation.configuration_utils": "pyc",
    "transformers.configuration_utils": "pyc",
}

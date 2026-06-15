# .pyinstaller_hooks/hook-transformers.py
# (was misnamed hook-sentence_transformers.py — now correctly named)

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

datas = collect_data_files("transformers")
hiddenimports = collect_submodules("transformers")

# Force bytecode-only for transformers modules that also use @_overload
# or call inspect.getsource at module level
module_collection_mode = {
    "transformers.generation.logits_process":   "bytecode",
    "transformers.generation.configuration_utils": "bytecode",
    "transformers.configuration_utils":         "bytecode",
}
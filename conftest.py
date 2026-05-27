"""Repo-wide pytest setup.

On Windows + conda, numpy/scipy/torch are linked against MKL/BLAS DLLs
that live in ``<env>/Library/bin``. That directory is added to PATH by
``conda activate`` — but only by activation. When pytest is launched
via the env's ``python.exe`` directly (e.g. from a VS Code launch
config or another non-conda shell), the DLL search path is missing
``Library/bin`` and numpy's first matmul aborts the process with
``ERROR_DELAY_LOAD_FAILED`` (Windows 0xC06D007F) — no Python exception,
just a thread dump and exit code -1066598273.

We make pytest robust against that by adding the env's ``Library/bin``
(and a few related dirs) to the DLL search path explicitly. Must
happen before numpy is imported, which is why this lives in the root
conftest (pytest loads it as one of the first things).

Also sets ``KMP_DUPLICATE_LIB_OK=TRUE`` because torch's bundled
OpenMP/MKL can collide with numpy's when both are loaded into the same
session (test_pinn alongside test_ukf).
"""

import os
import sys

# Keep DLL-directory handles alive at module level so they are not GC'd.
_dll_handles = []

if sys.platform == "win32":
    env_root = os.path.dirname(sys.executable)
    for sub in ("Library/bin", "Library/mingw-w64/bin", "Library/usr/bin", "DLLs"):
        path = os.path.join(env_root, *sub.split("/"))
        if os.path.isdir(path):
            if hasattr(os, "add_dll_directory"):
                _dll_handles.append(os.add_dll_directory(path))
            os.environ["PATH"] = path + os.pathsep + os.environ.get("PATH", "")

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

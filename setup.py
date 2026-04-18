import os
import shlex

from Cython.Build import cythonize
from setuptools import setup
from setuptools_cuda import CudaExtension


def _prepend_cudahome_bin_to_path():
    cuda_home = os.environ.get("CUDAHOME")
    if not cuda_home:
        return

    cuda_bin = os.path.join(cuda_home, "bin")
    if not os.path.isdir(cuda_bin):
        return

    path_entries = [entry for entry in os.environ.get("PATH", "").split(os.pathsep) if entry]
    cuda_bin_realpath = os.path.realpath(cuda_bin)
    path_entries = [entry for entry in path_entries if os.path.realpath(entry) != cuda_bin_realpath]
    os.environ["PATH"] = os.pathsep.join([cuda_bin] + path_entries)


def _ensure_nvcc_allow_unsupported_compiler():
    required_flag = "-allow-unsupported-compiler"
    existing = os.environ.get("NVCC_PREPEND_FLAGS", "")
    try:
        flags = shlex.split(existing)
    except ValueError:
        flags = existing.split()

    if required_flag not in flags:
        flags.append(required_flag)
    os.environ["NVCC_PREPEND_FLAGS"] = " ".join(flags)


_prepend_cudahome_bin_to_path()
_ensure_nvcc_allow_unsupported_compiler()

setup(
    cuda_extensions=cythonize(
        [
            CudaExtension(
                "omnisolver.bruteforce.ext.gpu",
                [
                    "omnisolver/extensions/bruteforce_gpu.cu",
                    "omnisolver/extensions/bruteforce_wrapper_gpu.pyx",
                ],
            )
        ]
    )
)

from Cython.Build import cythonize
from setuptools import setup
from setuptools_cuda import CudaExtension

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

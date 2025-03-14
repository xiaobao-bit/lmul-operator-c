from setuptools import setup, Extension
import pybind11

ext_modules = [
    Extension(
        "lmul",  # Output Python module name
        ["bindings.cpp", "operator.cpp"],  # Source file
        include_dirs=[pybind11.get_include()],  # Include Pybind11 headers
        language="c++",
        extra_compile_args=["-std=c++11", "-fopenmp"],  # Add OpenMP if needed
    ),
]

setup(
    name="lmul",
    version="0.1",
    ext_modules=ext_modules,
)

import os
from setuptools import setup, find_packages

CUDA_MODULE = "./hastl/build_stl.py:build_stl_cuda"
OPENCL_MODULE = "./hastl/build_stl.py:build_stl_opencl"
C_MODULE = "./hastl/build_stl.py:build_stl_c"
MULTICORE_MODULE = "./hastl/build_stl.py:build_stl_multicore"

VERSION = "0.1.2"

def run_setup(cffi_mods):
    return setup(
        name="hastl",
        version=VERSION,
        author="Dmitry Serykh",
        author_email="dmitry.serykh@gmail.com",
        description=("A fast GPU implementation of STL decomposition with missing values"),
        long_description=open("README.rst", "rt").read(),
        url="https://github.com/mortvest/hastl",
        license="MIT",
        packages=find_packages(),
        classifiers=[
            "Development Status :: 3 - Alpha",
            "Environment :: GPU",
            "Intended Audience :: Science/Research",
            "Intended Audience :: Developers",
            "License :: OSI Approved :: MIT License",
            "Operating System :: POSIX :: Linux",
            "Operating System :: MacOS",
            "Programming Language :: Python",
            "Programming Language :: Python :: 3 :: Only",
            "Programming Language :: Python :: 3.6",
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Topic :: Software Development",
            "Topic :: Scientific/Engineering",
        ],
        install_requires=[
            "futhark-ffi>=0.13.0",
        ],
        setup_requires=[
            "futhark-ffi>=0.13.0"
        ],
        cffi_modules=cffi_mods
    )


def check(backend_str):
    allowed_backends = ["cuda", "opencl", "c", "multicore"]
    backend = backend_str.lower()
    if backend not in allowed_backends:
        raise ValueError("Invalid backend '{}' encountered in the environment variable. Must be one of {}".format(backend_str, allowed_backends))
    if backend == "cuda":
        return CUDA_MODULE
    elif backend == "opencl":
        return OPENCL_MODULE
    elif backend == "multicore":
        return MULTICORE_MODULE
    return C_MODULE

# read environment variable
env_backends = os.environ.get("HASTL_BACKENDS", None)
if env_backends:
    CFFI_MODULES = list(set([check(backend) for backend in env_backends.split(" ")]))
else:
    # if not set, try compiling all available backends
    CFFI_MODULES = [CUDA_MODULE, OPENCL_MODULE, C_MODULE, MULTICORE_MODULE]

run_setup(CFFI_MODULES)

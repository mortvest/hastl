import os
from setuptools import setup, find_packages

BACKENDS = ["cuda", "opencl", "c", "multicore"]
MODULE_BASE = "./hastl/build_stl.py:build_stl_"
ENV_VAR = "HASTL_BACKENDS"

VERSION = "0.1.3"


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
            "futhark-ffi>=0.14.0",
        ],
        setup_requires=[
            "futhark-ffi>=0.14.0"
        ],
        cffi_modules=cffi_mods
    )

def check(backend_str):
    backend = backend_str.lower()
    if backend not in BACKENDS:
        raise ValueError("Invalid backend '{}' encountered in the environment variable. Must be one of {}".format(backend_str, allowed_backends))
    return MODULE_BASE + backend

# read environment variable
env_backends = os.environ.get(ENV_VAR, None)
# build the list of CFFI modules
if env_backends:
    CFFI_MODULES = list(set([check(backend) for backend in env_backends.split(" ")]))
else:
    # if not set, compile all available backends
    CFFI_MODULES = [MODULE_BASE + backend for backend in BACKENDS]

run_setup(CFFI_MODULES)

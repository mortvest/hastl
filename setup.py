import os
from setuptools import setup, find_packages

ALL_BACKENDS = ["cuda", "opencl", "c", "multicore"]
CFFI_BASE = "./hastl/build_stl.py:"
STL_MODULE_BASE = CFFI_BASE + "build_stl_"
LOESS_MODULE_BASE = CFFI_BASE + "build_loess_"
ENV_VAR = "HASTL_BACKENDS"

VERSION = "0.1.7"


def check_backend(backend_str, base):
    backend = backend_str.lower()
    if backend not in ALL_BACKENDS:
        raise ValueError("Invalid backend '{}' encountered in the environment variable. Must be one of {}".
                         format(backend_str, ALL_BACKENDS))
    return base + backend

def find_cffi_modules():
    # read environment variable
    env_backends = os.environ.get(ENV_VAR, None)
    backends = env_backends.split(" ") if env_backends else ["c"]
    modules = []
    for module_base in [STL_MODULE_BASE, LOESS_MODULE_BASE]:
        modules += list({check_backend(backend, module_base) for backend in backends})
    assert modules, "List of cffi modules can not be empty"
    print("Attempting to compile following modules:", modules)
    return modules

setup(
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
        "futhark-ffi==0.14.2",
    ],
    setup_requires=[
        "futhark-ffi==0.14.2"
    ],
    cffi_modules=find_cffi_modules()
)

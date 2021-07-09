import sys
from setuptools import setup, find_packages
import os

CUDA_MODULE = "./hastl/build_stl.py:build_stl_cuda"
OPENCL_MODULE = "./hastl/build_stl.py:build_stl_opencl"
C_MODULE = "./hastl/build_stl.py:build_stl_c"

VERSION = "0.1.2"

def gen_setup(cffi_mods):
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
        install_requires=[
            "futhark-ffi>=0.13.0",
        ],
        setup_requires=[
            "futhark-ffi>=0.13.0"
        ],
        cffi_modules=cffi_mods
    )

# TODO: REWRITE THIS
try:
    gen_setup([
        CUDA_MODULE,
        OPENCL_MODULE,
        C_MODULE
    ])
except:
    print("Could not locate a working CUDA installation, skipping..")
    try:
        gen_setup([
            OPENCL_MODULE,
            C_MODULE
        ])
    except:
        print("Could not locate a working OpenCL installation, skipping..")
        gen_setup([
            C_MODULE
        ])

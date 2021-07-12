from setuptools import setup, find_packages

CUDA_MODULE = "./hastl/build_stl.py:build_stl_cuda"
OPENCL_MODULE = "./hastl/build_stl.py:build_stl_opencl"
C_MODULE = "./hastl/build_stl.py:build_stl_c"

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

CFFI_MODULES = [CUDA_MODULE, OPENCL_MODULE, C_MODULE]

while CFFI_MODULES:
    try:
        run_setup(CFFI_MODULES)
        break
    except:
        print("Compilation of CFFI module '{}': failed, skipping...".format(CFFI_MODULES.pop(0)))

from setuptools import setup, find_packages
setup(
    name = "hastl",
    version = "0.1",
    author = "Dmitry Serykh",
    author_email = "dmitry.serykh@gmail.com",
    description = ("A fast GPU implementation of STL decomposition with missing values"),
    license = "MIT",
    packages=find_packages(),
    package_data={'': ["_stl.cpython-39-x86_64-linux-gnu.so"]},
    install_requires=[
        "numpy>=1.21.0",
        "futhark-ffi>=0.13.0"
    ],
    include_package_data=True
)

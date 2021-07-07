from distutils.sysconfig import get_config_var
import subprocess
from setuptools import setup, find_packages
from setuptools.command.install import install as BaseInstall

import futhark_ffi

BACKENDS = ["cuda", "opencl", "c"]

class HaSTLInstall(BaseInstall):
    def run(self):
        suffix = get_config_var("EXT_SUFFIX")
        for backend in BACKENDS:
            retval = subprocess.call(("make",
                                      "suffix={}".format(suffix),
                                      "_stl_{0}{1}".format(backend, suffix)), cwd="_futhark-ffi")
            if retval != 0:
                print("Compilation of {} backend failed, skipping..".format(backend))
        subprocess.call("cp *.so ../hastl/", cwd="_futhark-ffi", shell=True)
        return super().run()

setup(
    name="hastl",
    version="0.1",
    author="Dmitry Serykh",
    author_email="dmitry.serykh@gmail.com",
    description=("A fast GPU implementation of STL decomposition with missing values"),
    license="MIT",
    packages=find_packages(),
    package_data={'': ["*.so"]},
    install_requires=[
        "futhark-ffi>=0.13.0"
    ],
    setup_requires=[
        "futhark-ffi>=0.13.0"
    ],
    include_package_data=True,
    cmdclass={
        "install":  HaSTLInstall,
    }
)

import re
import sys

from cffi import FFI
from futhark_ffi.build import build


def strip_includes(header):
    return re.sub('^(#ifdef __cplusplus\n.*\n#endif|#.*)\n', '', header, flags=re.M)

build_stl_c = build("hastl/src/stl_c", "hastl/stl_c")
build_stl_opencl = build("hastl/src/stl_opencl", "hastl/stl_opencl")
build_stl_cuda = build("hastl/src/stl_cuda", "hastl/stl_cuda")
build_stl_multicore = build("hastl/src/stl_multicore", "hastl/stl_multicore")


build_loess_c = build("hastl/src/loess_c", "hastl/loess_c")
build_loess_opencl = build("hastl/src/loess_opencl", "hastl/loess_opencl")
build_loess_cuda = build("hastl/src/loess_cuda", "hastl/loess_cuda")
build_loess_multicore = build("hastl/src/loess_multicore", "hastl/loess_multicore")

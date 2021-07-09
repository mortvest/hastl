import re
import sys
import os

from cffi import FFI


def strip_includes(header):
    return re.sub('^(#ifdef __cplusplus\n.*\n#endif|#.*)\n', '', header, flags=re.M)

def build(source_name, out_name):
    ffibuilder = FFI()

    header_file = source_name + '.h'
    source_file = source_name + '.c'

    search = re.search('#define FUTHARK_BACKEND_([a-z0-9_]*)', open(header_file).read())
    if not search:
        sys.exit('Cannot determine Futhark backend from {}'.format(header_file))

    backend=search.group(1)

    with open(source_file) as source:
        libraries = ['m']
        extra_compile_args = ['-std=c99']
        if backend == 'opencl':
            if sys.platform == 'darwin':
                extra_compile_args += ['-framework', 'OpenCL']
            libraries += ['OpenCL']
        elif backend == 'cuda':
            libraries += ['cuda', 'cudart', 'nvrtc']
        elif backend == 'multicore':
            extra_compile_args += ['-pthread']
        ffibuilder.set_source(out_name,
                              source.read(),
                              libraries=libraries,
                              extra_compile_args=extra_compile_args)

    with open(header_file) as header:
        cdef = 'typedef void* cl_command_queue;'
        cdef += '\ntypedef void* cl_mem;'
        cdef += '\ntypedef void* CUdeviceptr;'
        cdef += strip_includes(header.read())
        cdef += "\nvoid free(void *ptr);"
        ffibuilder.cdef(cdef)

    return ffibuilder

build_stl_c = build("hastl/src/stl_c", "hastl._stl_c")
build_stl_opencl = build("hastl/src/stl_opencl", "hastl._stl_opencl")
build_stl_cuda = build("hastl/src/stl_cuda", "hastl._stl_cuda")


if __name__ == "__main__":
    build_stl_c.compile()
    build_stl_opencl.compile()
    build_stl_cuda.compile()

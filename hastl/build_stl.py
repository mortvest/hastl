import re
import sys

from cffi import FFI
# from futhark_ffi.build import build


def strip_includes(header):
    return re.sub('^(#ifdef __cplusplus\n.*\n#endif|#.*)\n', '', header, flags=re.M)


def build(input_name, output_name):
    ffibuilder = FFI()

    header_file = input_name + '.h'
    source_file = input_name + '.c'

    output_name_lst = output_name.split("/")
    output_name_lst[-1] = "_" + output_name_lst[-1]
    output_name = ".".join(output_name_lst)

    search = re.search('#define FUTHARK_BACKEND_([a-z0-9_]*)', open(header_file).read())
    if not search:
        sys.exit('Cannot determine Futhark backend from {}'.format(header_file))

    backend = search.group(1)

    with open(source_file) as source:
        libraries = ['m']
        extra_compile_args = ['-std=c99']
        if backend == 'opencl':
            if sys.platform == 'darwin':
                extra_compile_args += ['-framework', 'OpenCL']
            else:
                libraries += ['OpenCL']
        elif backend == 'cuda':
            libraries += ['cuda', 'cudart', 'nvrtc']
        elif backend == 'multicore':
            extra_compile_args += ['-pthread']
        ffibuilder.set_source(output_name,
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


build_stl_c = build("hastl/src/stl_c", "hastl/stl_c")
build_stl_opencl = build("hastl/src/stl_opencl", "hastl/stl_opencl")
build_stl_cuda = build("hastl/src/stl_cuda", "hastl/stl_cuda")
build_stl_multicore = build("hastl/src/stl_multicore", "hastl/stl_multicore")

if __name__ == "__main__":
    build_stl_c.compile()
    build_stl_opencl.compile()
    build_stl_cuda.compile()
    build_stl_multicore.compile()

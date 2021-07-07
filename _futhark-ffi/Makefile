backends = cuda opencl c

.PHONY: $(backends:%=%)

.PRECIOUS: $(backends:%=stl_%.c) $(backends:%=stl_%.h)

stl_%.c stl_%.h: stl.fut
	futhark pkg add github.com/diku-dk/sorts
	futhark pkg sync
	futhark $* $^ --library -o stl_$*

build_%: stl_%.c stl_%.h
	build_futhark_ffi stl_$*

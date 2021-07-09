backends = cuda opencl c
suffix = cpython-39-x86_64-linux-gnu

.PRECIOUS: $(backends:%=stl_%.c) $(backends:%=stl_%.h)

stl_%.c stl_%.h: stl.fut
	futhark pkg add github.com/diku-dk/sorts
	futhark pkg sync
	futhark $* $^ --library -o stl_$*

_stl_%$(suffix): stl_%.c stl_%.h
	build_futhark_ffi stl_$*

clean:
	rm -f _stl*

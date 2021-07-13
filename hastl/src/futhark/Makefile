backends = cuda opencl c multicore

.PHONY: all c_files h_files clean

all: c_files h_files

c_files: $(backends:%=stl_%.c)
h_files: $(backends:%=stl_%.h)

stl_%.c stl_%.h: stl.fut
	futhark pkg add github.com/diku-dk/sorts
	futhark pkg sync
	futhark $* $^ --library -o stl_$*

clean:
	rm -rf *.c *.h lib/ futhark.pkg

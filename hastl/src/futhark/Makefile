backends = cuda opencl c multicore

.PHONY: all move stl loess loess_c_files loess_h_files stl_c_files stl_h_files clean

move:
	cp *.c *.h ../

all: stl loess

loess: loess_c_files loess_h_files

stl: stl_c_files stl_h_files

stl_c_files: $(backends:%=stl_%.c)
stl_h_files: $(backends:%=stl_%.h)

loess_c_files: $(backends:%=loess_%.c)
loess_h_files: $(backends:%=loess_%.h)

stl_%.c stl_%.h: stl.fut loess.fut utils.fut
	futhark pkg add github.com/diku-dk/sorts
	futhark pkg sync
	futhark $* $< --library -o stl_$*

loess_%.c loess_%.h: loess.fut utils.fut
	futhark pkg add github.com/diku-dk/sorts
	futhark pkg sync
	futhark $* $< --library -o loess_$*

clean:
	rm -rf *.c *.h lib/ futhark.pkg

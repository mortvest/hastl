HaSTL
=================================================================

HaSTL [ˈheɪstiɛl]: A fast GPU implementation of batched Seasonal and Trend
decomposition using Loess (STL) [1] with missing values and support for both
CUDA and OpenCL (C and multicore backends are also available).
Loosely based on `stlplus <https://github.com/hafen/stlplus>`_, a
popular library for the R programming language. The GPU code is written in
`Futhark <https://futhark-lang.org>`_, a functional language that compiles
to efficient parallel code.


Requirements
------------

You would need a working OpenCL or CUDA installation/header files, C compiler and these Python packages:

- futhark-ffi>=0.14.0
- wheel


Installation
------------

You may want to run the program in a Python virtual environment. Create it via::

  python -m venv env

Then, activate the virtual environment via::

  . env/bin/activate

Upgrade pip via::

  pip install --upgrade pip

Then select the backends (choose from opencl, cuda, c and multicore) that you wish to build by setting the environment variable::

  export HASTL_BACKENDS="opencl multicore c" 

If no environmental variable is set, an attempt will be made to compile all the
available backends.

The package can then be easily installed using pip. This will take a while, since we need
to compile the shared libraries for your particular system, Python implementation and all selected backends::

  pip install hastl

To install the package from the sources, first get the current stable release via::

  git clone https://github.com/mortvest/hastl

Install the bfast dependencies via::

  pip install -r requirements.txt

Afterwards, you can install the package. This can also take a while::

  python setup.py sdist bdist_wheel
  pip install .


Usage
-----
Set backend to "cuda", "opencl", "multicore" or "c" and run::

  from hastl import STL
  stl = STL(backend="opencl")
  seasonal, trend, remainder = stl.fit(data, n_p=12)


References
----------
[1] Cleveland, Robert B., et al. "STL: A seasonal-trend decomposition." J. Off. Stat 6.1 (1990): 3-73.

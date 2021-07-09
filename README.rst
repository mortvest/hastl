HaSTL
=================================================================

HaSTL [ˈheɪstiɛl]: A fast GPU implementation of batched Seasonal and Trend
decomposition using Loess (STL) [1] with missing values and support for both
CUDA and OpenCL (a rather slow C backend is also available).
Loosely based on `stlplus <https://github.com/hafen/stlplus>`_, a
popular library for the R programming language. The GPU code is written in
`Futhark <https://futhark-lang.org>`_, a small functional language that compiles
to efficient parallel code.


Requirements
------------

You would need a working OpenCL or CUDA installation, C compiler and these Python packages:

- futhark-ffi>=0.13.0
- wheel


Installation
------------

You may want to run the program in a Python virtual environment. Create it via::

  python -m venv env

Then, activate the virtual environment via::

  . env/bin/activate

Upgrade pip via::

  pip install --upgrade pip

The package can then be easily installed using pip via::

  pip install hastl

To install the package from the sources, first get the current stable release via::

  git clone https://github.com/mortvest/hastl

Install the bfast dependencies via::

  pip install -r requirements.txt

Afterwards, you can install the package. This will take a while, since we need
to compile the shared libraries for your particular system and Python
implementation and all available backends::

  python setup.py sdist bdist_wheel
  pip install .


Usage
-----
Set backend to "cuda", "opencl" or "c" and run::

  from hastl import STL
  stl = STL(backend=backend)
  seasonal, trend, remainder = stl.fit(data, n_p=12)


References
----------
[1] Cleveland, Robert B., et al. "STL: A seasonal-trend decomposition." J. Off. Stat 6.1 (1990): 3-73.

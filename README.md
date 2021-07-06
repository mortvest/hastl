# HaSTL
HaSTL \[ˈheɪstiɛl\]: A fast GPU implementation of batched Seasonal and Trend decomposition using Loess (STL) [1] with missing values and support for both CUDA and OpenCL. Loosely based on [stlplus](https://github.com/hafen/stlplus), a popular library for the R programming language. The GPU code is written in [Futhark](https://futhark-lang.org/), a small functional language that compiles to efficient parallel code.
## Usage
```python
from hastl import STL
stl = STL()
seasonal, trend, remainder = stl.fit(x, n_p = 12)

```
## References
[1] Cleveland, Robert B., et al. "STL: A seasonal-trend decomposition." J. Off. Stat 6.1 (1990): 3-73.

from importlib import import_module

from futhark_ffi import Futhark
import numpy as np

from . import stl


class LOESS():
    """
    Batched LOESS Smoother for GPUs
    """
    def __init__(self,
                 backend="opencl",
                 jump_threshold=9,
                 max_group_size=1024,
                 tuning=None,
                 device=None,
                 platform=None,
                 debug=False
                 ):
        # set device-specific parameters
        self.backend = backend
        self.jump_threshold = jump_threshold
        self.max_group_size = max_group_size
        self.device = device
        self.platform = platform
        self.debug = debug

        if tuning is None:
            # TODO: add tuning
            self.tuning = {}

        self._backends = ["opencl", "cuda", "multicore", "c"]

        if self.backend not in self._backends:
            raise ValueError("Unknown backend: '{}'".format(self.backend))

        fut_lib = stl._try_importing(backend, "loess")

        if self.debug and self.backend in ["opencl", "cuda"]:
            print("Initializing the device")
        try:
            self._fut_obj = Futhark(fut_lib, tuning=self.tuning, device=self.device, platform=self.platform)
        except ValueError as err:
            from_err = err if self.debug else None
            raise ValueError("An error occurred while initializing the device") from from_err

    def fit(self,
            Y,
            q,
            degree=1,
            jump=None
            ):
        Y = np.asarray(Y)

        if Y.ndim != 2:
            raise TypeError("Y should be a 2d array")
        m, n = Y.shape

        q = stl._wincheck(q)
        degree = stl._degcheck(degree)

        if jump is None:
            jump = np.ceil(min(q, n) / 10)
        jump = stl._jump_check(jump, n)

        if self.debug:
            print("Running the program")

        try:
            result_fut = self._fut_obj.main(Y,
                                            q,
                                            degree,
                                            jump,
                                            self.jump_threshold,
                                            self.max_group_size)

            result = self._fut_obj.from_futhark(result_fut)
        except ValueError as err:
            from_err = err if self.debug else None
            raise ValueError("An internal error occurred while running the GPU program") from from_err

        return result


    def fit_1d(self,
               y,
               q,
               degree=1,
               jump=None
               ):
        y = np.asarray(y)
        if y.ndim != 1:
            raise TypeError("y should be a 1d array")
        n = y.shape[0]
        Y = y.reshape(1, n)

        result = self.fit(Y,
                          q,
                          degree,
                          jump)
        return result[0]

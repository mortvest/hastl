from importlib import import_module

from futhark_ffi import Futhark
import numpy as np


class STL():
    """
    Batched STL decomposition for GPUs
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

        fut_lib = _try_importing(backend)

        if self.debug and self.backend in ["opencl", "cuda"]:
            print("Initializing the device")
        try:
            self._fut_obj = Futhark(fut_lib, tuning=self.tuning, device=self.device, platform=self.platform)
        except ValueError as err:
            from_err = err if self.debug else None
            raise ValueError("An error occurred while initializing the device") from from_err

    def fit(self,
            Y,
            n_p,
            t_window=None,
            l_window=None,
            t_degree=1,
            l_degree=None,
            t_jump=None,
            l_jump=None,
            inner=2,
            outer=1,
            critval=0.05,
            ):
        Y = np.asarray(Y)

        if Y.ndim != 2:
            raise TypeError("Y should be a 2d array")
        m, n = Y.shape

        if n_p < 4:
            raise ValueError("n_p was set to {}. Must be at least 4".format(n_p))
        n_p = int(n_p)

        if t_window is None:
            # t_window = self._nextodd(np.ceil(1.5 * n_p / (1 - 1.5 / self.s_window)))
            t_window = self._get_t_window(t_degree, n, n_p, critval)
        t_window = self._wincheck(t_window)

        if l_window is None:
            l_window = self._nextodd(n_p)
        l_window = self._wincheck(l_window)

        t_degree = self._degcheck(t_degree)

        if l_degree is None:
            l_degree = t_degree
        l_degree = self._degcheck(l_degree)

        if t_jump is None:
            t_jump = np.ceil(t_window / 10)
        t_jump = self._jump_check(t_jump)

        if l_jump is None:
            l_jump = np.ceil(l_window / 10)
        l_jump = self._jump_check(l_jump)

        inner = self._iter_check(inner)
        outer = self._iter_check(outer)

        if self.debug:
            print("Running the program")

        try:
            s_data, t_data, r_data = self._fut_obj.main(Y,
                                                        n_p,
                                                        t_window,
                                                        l_window,
                                                        t_degree,
                                                        l_degree,
                                                        t_jump,
                                                        l_jump,
                                                        inner,
                                                        outer,
                                                        self.jump_threshold,
                                                        self.max_group_size)

            season = self._fut_obj.from_futhark(s_data)
            trend = self._fut_obj.from_futhark(t_data)
            remainder = self._fut_obj.from_futhark(r_data)
        except ValueError as err:
            err_type = err.args[0].split("\n")[0]
            from_err = err if self.debug else None
            if err_type == "Assertion is false: sub_series_is_not_all_nan":
                raise ValueError("There is at least one sub-series for which all values are NaNs") from from_err
            else:
                raise ValueError("An internal error occurred while running the GPU program") from from_err

        return trend, season, remainder

    def fit_1d(self,
            y,
            n_p,
            t_window=None,
            l_window=None,
            t_degree=1,
            l_degree=None,
            t_jump=None,
            l_jump=None,
            inner=2,
            outer=1,
            critval=0.05,
            ):
        y = np.asarray(y)
        if y.ndim != 1:
            raise TypeError("y should be a 1d array")
        n = y.shape[0]
        Y = y.reshape(1, n)

        season, trend, remainder = self.fit(Y,
                                            n_p,
                                            t_window,
                                            l_window,
                                            t_degree,
                                            l_degree,
                                            t_jump,
                                            l_jump,
                                            inner,
                                            outer,
                                            critval)
        return season[0], trend[0], remainder[0]

    def _degcheck(self, x):
        x = int(x)
        if not (0 <= x <= 2):
            raise ValueError("Smoothing degree must be 0, 1, or 2")
        return x

    def _nextodd(self, x):
        x = round(x)
        x2 = x + 1 if x % 2 == 0 else x
        return int(x2)

    def _wincheck(self, x):
        x = self._nextodd(x)
        if x <= 0:
            raise ValueError("Window lengths must be positive")
        return x

    def _jump_check(self, x):
        return self._len_check(x, "Jump")

    def _iter_check(self, x):
        return self._len_check(x, "Number of iterations")

    def _len_check(self, x, name):
        x = int(x)
        if x < 0:
            raise ValueError("{} value must be non-negative".format(name))
        return x

    def _get_t_window(self, t_degree, n, n_p, omega):
        t_dg = max(t_degree - 1, 0)
        n_s = 10 * n + 1

        coefs_a_a = (0.000103350651767650, 3.81086166990428e-6)
        coefs_a_b = (-0.000216653946625270, 0.000708495976681902)

        coefs_b_a = (1.42686036792937, 2.24089552678906)
        coefs_b_b = (-3.1503819836694, -3.30435316073732)
        coefs_b_c = (5.07481807116087, 5.08099438760489)

        coefs_c_a = (1.66534145060448, 2.33114333880815)
        coefs_c_b = (-3.87719398039131, -1.8314816166323)
        coefs_c_c = (6.46952900183769, 1.85431548427732)

        # estimate critical frequency for seasonal
        betac0 = coefs_a_a[0] + coefs_a_b[0] * omega
        betac1 = coefs_b_a[0] + coefs_b_b[0] * omega + coefs_b_c[0] * omega**2
        betac2 = coefs_c_a[0] + coefs_c_b[0] * omega + coefs_c_c[0] * omega**2
        f_c = (1 - (betac0 + betac1 / n_s + betac2 / n_s**2)) / n_p

        # choose
        betat0 = coefs_a_a[t_dg] + coefs_a_b[t_dg] * omega
        betat1 = coefs_b_a[t_dg] + coefs_b_b[t_dg] * omega + coefs_b_c[t_dg] * omega**2
        betat2 = coefs_c_a[t_dg] + coefs_c_b[t_dg] * omega + coefs_c_c[t_dg] * omega**2

        betat00 = betat0 - f_c

        n_t = self._nextodd((-betat1 - np.sqrt(betat1**2 - 4 * betat00 * betat2)) / (2 * betat00))
        return n_t

def print_installed_backends():
    installed_backends = []
    for backend in ["cuda", "opencl", "multicore", "c"]:
        try:
            _try_importing(backend)
        except ValueError:
            pass
        else:
            installed_backends.append(backend)
    print("Installed HaSTL backens:")
    print(installed_backends)

def _try_importing(backend):
    module_name = "_stl_" + backend
    try:
        mod = import_module("hastl." + module_name)
    except:
        raise ValueError("Failed loading the {} backend".format(backend)) from None
    else:
        globals()[module_name] = mod
        return mod

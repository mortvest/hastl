from importlib import import_module
import re

from futhark_ffi import Futhark
import numpy as np


class STL():
    """Class for the STL decomposition.
    :param backend: which backend to use
    :type str: one of [c, opencl, cuda, multicore]
    :param jump_threshold: determines when to switch from the outer-parallel
        version. This optimal value for this parameter depends on your GPU.
        Should be set higher for higher-end GPUs.
    :type int: integer > 1, optional
    :param max_group_size: the maximal size for the workgroup. 1024 for the
        NVIDIA cards and 256 for AMD
    :type int: integer > 1
    :param tuning: The dictionary of threshold values.
        See https://futhark-lang.org/docs.html
    :type dict: dictionary of "string": integer
    :param debug: debug mode
    :type bool: true to enable debugging messages
    """
    def __init__(self,
                 backend="opencl",
                 jump_threshold=9,
                 max_group_size=1024,
                 tuning=None,
                 device=None,
                 platform=None,
                 profiling=False,
                 debug=False
                 ):
        # set device-specific parameters
        self.backend = backend
        self.jump_threshold = jump_threshold
        self.max_group_size = max_group_size
        self.device = device
        self.platform = platform
        self.profiling = profiling
        self.debug = debug

        if tuning is None:
            # TODO: add default tuning
            self.tuning = None

        self._backends = ["opencl", "cuda", "multicore", "c"]

        if self.backend not in self._backends:
            raise ValueError("Unknown backend: '{}'".format(self.backend))

        fut_lib = _try_importing(backend, "stl")

        if self.debug and self.backend in ["opencl", "cuda"]:
            print("Initializing the device")
        try:
            self._fut_obj = Futhark(fut_lib,
                                    tuning=self.tuning,
                                    device=self.device,
                                    platform=self.platform,
                                    profiling=self.profiling)
        except ValueError as err:
            from_err = err if self.debug else None
            raise ValueError("An error occurred while initializing the device") from from_err

    def fit(self,
            Y,
            n_p,
            q_s,
            q_t=None,
            q_l=None,
            d_s=1,
            d_t=1,
            d_l=None,
            jump_s=None,
            jump_t=None,
            jump_l=None,
            n_inner=2,
            n_outer=1,
            critfreq=0.05,
            dump=False,
            ):
        """Decomposes (m x n) array Y into a tuple of three (m x n) arrays which
        correspond to the seasonal, trend and remainder components.
        m is the number of time series that is being processed (batch size),
        and n is the time series length.
        :param Y: input data
        :type array: 2d array of float
        :param n_p: the number of observations in each cycle of the seasonal
            component
        :type int: integer >= 4
        :param q_s: smoothing parameter for the seasonal component
        :type int: positive, odd
        :param q_t: smoothing parameter for the trend component
        :type int: positive, odd, optional
        :param q_l: smoothing parameter for the low-pass filter
        :type int: positive, odd, optional
        :param d_s: polynomial degree for the smoothing of the seasonal component
        :type int: one of [0, 1, 2], optional
        :param d_t: polynomial degree for the smoothing of the trend component
        :type int: one of [0, 1, 2], optional
        :param d_l: polynomial degree for the smoothing of the low-pass filter
        :type int: one of [0, 1, 2], optional
        :param jump_s: stride of the jump parameter for the smoothing of the
            seasonal component (higher - faster, but less precise)
        :type int: integer >= 1, optional
        :param jump_t: stride of the jump parameter for the smoothing of the
            trend component (higher - faster, but less precise)
        :type int: integer >= 1, optional
        :param jump_l: stride of the jump parameter for the smoothing of the
            low-pass filter (higher - faster, but less precise)
        :type int: integer >= 1, optional
        :param n_inner: number of the inner loop iterations. 2 is the default
            value, set to 1 if n_outer is > 1
        :type int: integer >= 1, optional
        :param n_outer: number of the outer loop iterations, set to 1 for
            faster execution, set to 5 if extra robustness is required
        :type int: integer >= 1, optional
        :param critfreq: critical frequency to use for automatic calculation of
            smoothing windows for the trend and high-pass filter.
        :type float: the default value is almost always sufficient, optional
        :return: the decomposed result: seasonal, trend, remainder
        :rtype: tuple of 3 2d arrays
    """
        Y = np.asarray(Y)

        if Y.ndim != 2:
            raise TypeError("Y should be a 2d array")
        m, n = Y.shape

        if n_p < 4:
            raise ValueError("n_p was set to {}. Must be at least 4".format(n_p))
        n_p = int(n_p)

        if q_s < 7:
            raise ValueError("q_s was set to {}. Must be at least 7".format(q_s))
        q_s = _wincheck(q_s)

        if q_t is None:
            q_t = _nextodd(1.5 * n_p / (1 - 1.5 / q_s))
            # q_t = self._get_q_t(d_t, d_s, q_s, n_p, critfreq)
        q_t = _wincheck(q_t)

        if q_l is None:
            q_l = _nextodd(n_p)
        q_l = _wincheck(q_l)

        d_s = _degcheck(d_s)
        d_t = _degcheck(d_t)

        if d_l is None:
            d_l = d_t
        d_l = _degcheck(d_l)

        if jump_s is None:
            jump_s = np.ceil(q_s / 10)
        jump_s = _jump_check(jump_s, n)

        if jump_t is None:
            jump_t = np.ceil(q_t / 10)
        jump_t = _jump_check(jump_t, n)

        if jump_l is None:
            jump_l = np.ceil(q_l / 10)
        jump_l = _jump_check(jump_l, n)

        n_inner = _iter_check(n_inner)
        n_outer = _iter_check(n_outer)

        jump_threshold = 10000000 if self.backend in ["c", "multicore"] else self.jump_threshold

        if self.debug:
            print("Running the program")

        if dump:
            import futhark_data
            f = open("dump.in", "wb")
            Y_64 = Y.astype(np.float64)
            futhark_data.dump(Y_64, f)

            params = [(n_p, "n_p"),
                      (q_s, "q_s"),
                      (q_t, "q_t"),
                      (q_l, "q_l"),
                      (d_s, "d_s"),
                      (d_t, "d_t"),
                      (d_l, "d_l"),
                      (jump_s, "n_jump_s"),
                      (jump_t, "n_jump_t"),
                      (jump_l, "n_jump_l"),
                      (n_inner, "n_inner"),
                      (n_outer, "n_outer"),
                      (self.jump_threshold, "jump threshold"),
                      (self.max_group_size, "q_threshold")]

            for (par, name) in params:
                print("{}: {}".format(name, par))
                futhark_data.dump(np.int64(par), f)

            f.close()
            exit()

        try:
            s_data, t_data, r_data = self._fut_obj.main(Y,
                                                        n_p,
                                                        q_s,
                                                        q_t,
                                                        q_l,
                                                        d_s,
                                                        d_t,
                                                        d_l,
                                                        jump_s,
                                                        jump_t,
                                                        jump_l,
                                                        n_inner,
                                                        n_outer,
                                                        jump_threshold,
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

        return season, trend, remainder

    def fit_1d(self,
            y,
            n_p,
            q_s,
            q_t=None,
            q_l=None,
            d_s=1,
            d_t=1,
            d_l=None,
            jump_s=None,
            jump_t=None,
            jump_l=None,
            n_inner=2,
            n_outer=1,
            critfreq=0.05,
            dump=False
            ):
        y = np.asarray(y)
        if y.ndim != 1:
            raise TypeError("y should be a 1d array")
        n = y.shape[0]
        Y = y.reshape(1, n)

        season, trend, remainder = self.fit(Y,
                                            n_p,
                                            q_s,
                                            q_t,
                                            q_l,
                                            d_s,
                                            d_t,
                                            d_l,
                                            jump_s,
                                            jump_t,
                                            jump_l,
                                            n_inner,
                                            n_outer,
                                            critfreq,
                                            dump)
        return season[0], trend[0], remainder[0]

    # def _get_q_t(self, d_t, d_s, n_s, n_p, omega):
    #     t_dg = max(d_t - 1, 0)
    #     s_dg = max(d_s - 1, 0)

    #     coefs_a_a = (0.000103350651767650, 3.81086166990428e-6)
    #     coefs_a_b = (-0.000216653946625270, 0.000708495976681902)

    #     coefs_b_a = (1.42686036792937, 2.24089552678906)
    #     coefs_b_b = (-3.1503819836694, -3.30435316073732)
    #     coefs_b_c = (5.07481807116087, 5.08099438760489)

    #     coefs_c_a = (1.66534145060448, 2.33114333880815)
    #     coefs_c_b = (-3.87719398039131, -1.8314816166323)
    #     coefs_c_c = (6.46952900183769, 1.85431548427732)

    #     # estimate critical frequency for seasonal
    #     betac0 = coefs_a_a[s_dg] + coefs_a_b[s_dg] * omega
    #     betac1 = coefs_b_a[s_dg] + coefs_b_b[s_dg] * omega + coefs_b_c[s_dg] * omega**2
    #     betac2 = coefs_c_a[s_dg] + coefs_c_b[s_dg] * omega + coefs_c_c[s_dg] * omega**2
    #     f_c = (1 - (betac0 + betac1 / n_s + betac2 / n_s**2)) / n_p

    #     # choose
    #     betat0 = coefs_a_a[t_dg] + coefs_a_b[t_dg] * omega
    #     betat1 = coefs_b_a[t_dg] + coefs_b_b[t_dg] * omega + coefs_b_c[t_dg] * omega**2
    #     betat2 = coefs_c_a[t_dg] + coefs_c_b[t_dg] * omega + coefs_c_c[t_dg] * omega**2

    #     betat00 = betat0 - f_c

    #     n_t = _nextodd((-betat1 - np.sqrt(betat1**2 - 4 * betat00 * betat2)) / (2 * betat00))
    #     return n_t


def _degcheck(x):
    x = int(x)
    if not (0 <= x <= 2):
        raise ValueError("Smoothing degree must be 0, 1, or 2")
    return x

def _nextodd(x):
    x = round(x)
    x2 = x + 1 if x % 2 == 0 else x
    return int(x2)

def _wincheck(x):
    x = _nextodd(x)
    if x <= 0:
        raise ValueError("Window lengths must be positive")
    return x

def _jump_check(j, n):
    n_m = n if j == 1 else n / j + 1
    if n_m < 2:
        raise ValueError("Jump value is set too high, must be <= n")
    return _len_check(j, "Jump")

def _iter_check(x):
    return _len_check(x, "Number of iterations")

def _len_check(x, name):
    x = int(x)
    if x < 0:
        raise ValueError("{} value must be non-negative".format(name))
    return x

def _try_importing(backend, name):
    module_name = "_" + name + "_" + backend
    try:
        mod = import_module("hastl." + module_name)
    except:
        raise ValueError("Failed loading the {} backend".format(backend)) from None
    else:
        globals()[module_name] = mod
        return mod

def print_installed_backends():
    installed_backends = []
    for backend in ["cuda", "opencl", "multicore", "c"]:
        try:
            _try_importing(backend, "stl")
        except ValueError:
            pass
        else:
            installed_backends.append(backend)
    print("Installed HaSTL backens:")
    print(installed_backends)

def load_tuning_file(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()
    return {k : int(v) for k, v in map(lambda l: re.search(r"(.*)=([0-9]*)", l).groups(), lines)}

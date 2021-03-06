-- Implementation of LOESS Smoother with missing values
import "utils"

module loess_m = {
module T = f64

type t = T.t
type^ fun_t = (t -> t -> t -> t -> t -> t -> t -> t -> t -> t -> t)
type^ loess_t [m][n][n_m] = ([m][n]i64 ->
                               [m][n]t ->
                               i64 ->
                               (i64 -> i64) ->
                               [m][n]t ->
                               [m][n_m]i64 ->
                               [m][n_m]t ->
                               [m]i64 ->
                               fun_t ->
                               fun_t ->
                               ([m][n_m]t, [m][n_m]t))

type^ loess_css_t [m][n_p][n][n_m] = ([m][n_p][n]i64 ->
                                        [m][n_p][n]t ->
                                        i64 ->
                                        (i64 -> i64) ->
                                        [m][n_p][n]t ->
                                        [m][n_p][n_m]i64 ->
                                        [m][n_p][n_m]t ->
                                        [m][n_p]i64 ->
                                        fun_t ->
                                        fun_t ->
                                        ([m][n_p][n_m]t, [m][n_p][n_m]t))

--------------------------------------------------------------------------------
-- FIT/SLOPE calculation function for polynomial degrees [0,1,2]              --
--------------------------------------------------------------------------------
let fit_fun_zero (w_j: t) (yy_j: t) (_: t) (_: t) (a0: t)
                 (_: t) (_: t) (_: t) (_: t) (_: t): t =
  w_j * a0 * yy_j

let fit_fun_one (w_j: t) (yy_j: t) (xw_j: t) (_: t) (_: t)
                (a11: t) (_: t) (b11: t) (_: t) (_: t): t =
  (w_j * a11 + xw_j * b11) * yy_j

let fit_fun_two (w_j: t) (yy_j: t) (xw_j: t) (x2w_j: t) (_: t)
                (_: t) (a12: t) (_: t) (b12: t) (c12: t): t =
  (w_j * a12 + xw_j * b12 + x2w_j * c12) * yy_j

let slope_fun_zero (_: t) (_: t) (_: t) (_: t) (_: t) (_: t)
                   (_: t) (_: t) (_: t) (_: t): t =
  T.i64 0

let slope_fun_one (w_j: t) (yy_j: t) (xw_j: t) (_: t) (_: t)
                    (_: t) (_: t) (b11: t) (_: t) (c11: t): t =
  (w_j * b11 + xw_j * c11) * yy_j

let slope_fun_two (w_j: t) (yy_j: t) (xw_j: t) (x2w_j: t) (c2: t)
                  (_: t) (a2: t) (_: t) (b2: t) (_: t): t =
  (w_j * a2 + xw_j * b2 + x2w_j * c2) * yy_j


--------------------------------------------------------------------------------
-- Main LOESS procedure - outer parallel version, with extra work             --
--------------------------------------------------------------------------------
let loess_outer [n] [n_m] (xx: [n]i64)
                          (yy: [n]t)
                          (q: i64)
                          (m_fun: i64 -> i64)
                          (ww: [n]t)
                          (l_idx: [n_m]i64)
                          (lambda: [n_m]t)
                          (n_nn: i64)
                          (fit_fun: fun_t)
                          (slope_fun: fun_t)
                          : ([n_m]t, [n_m]t) =
  let q_slice 'a (arr: [n]a) (l_idx_i: i64) (v: a) (add: a -> a -> a) (zero: a): [q]a =
    #[unsafe]
    tab (\j -> if j >= n_nn then zero else add arr[l_idx_i + j] v) q
  -- need the duplicate to prevent manifestation
  let q_slice' 'a (arr: [n]a) (l_idx_i: i64) (v: a) (add: a -> a -> a) (zero: a): [q]a =
    #[unsafe]
    tab (\j -> if j >= n_nn then zero else add arr[l_idx_i + j] v) q
  in
  -- [n_m]
  #[sequential_inner]
  map3 (\i l_idx_i lambda_i ->
         -----------------------------------
         -- REDOMAP 1
         -----------------------------------
         #[unsafe]
         let xx_slice = q_slice xx l_idx_i 1 (+) 0
         let ww_slice = q_slice ww l_idx_i 0 (+) 0
         let (w, xw, x2w, x3w, x4w) =
           map2 (\xx_j ww_j ->
                   let x_j = (xx_j - m_fun i) |> T.i64
                   -- tricube
                   let r = T.abs x_j
                   let tmp1 = r / lambda_i
                   let tmp2 = 1.0 - tmp1 * tmp1 * tmp1
                   let tmp3 = tmp2 * tmp2 * tmp2
                   -- scale by user-defined weights
                   let w_j = tmp3 * ww_j
                   let xw_j = x_j * w_j
                   let x2w_j = x_j * xw_j
                   let x3w_j = x_j * x2w_j
                   let x4w_j = x_j * x3w_j
                   in (w_j, xw_j, x2w_j, x3w_j, x4w_j)
                ) xx_slice ww_slice |> unzip5
         -- then, compute fit and slope based on polynomial degree
         let a_0 = T.sum w
         let b_0 = T.sum xw
         let c_0 = T.sum x2w
         let d_0 = T.sum x3w
         let e_0 = T.sum x4w

         let a = a_0 + T.epsilon
         let b = b_0 + T.epsilon
         let c = c_0 + T.epsilon
         let d = d_0 + T.epsilon
         let e = e_0 + T.epsilon

         -- degree 0
         let a0 = 1 / a

         -- degree 1
         let det1 = 1 / (a * c - b * b)
         let a11 = c * det1
         let b11 = -b * det1
         let c11 = a * det1

         -- degree 2
         let a12 = e * c - d * d
         let b12 = c * d - e * b
         let c12 = b * d - c * c
         let a2 = c * d - e * b
         let b2 = e * a - c * c
         let c2 = b * c - d * a
         let det = 1 / (a * a12 + b * b12 + c * c12)
         let a12 = a12 * det
         let b12 = b12 * det
         let c12 = c12 * det
         let a2 = a2 * det
         let b2 = b2 * det
         let c2 = c2 * det

         -----------------------------------
         -- REDOMAP 2
         -----------------------------------
         let xx_slice' = q_slice' xx l_idx_i 1 (+) 0
         let ww_slice' = q_slice' ww l_idx_i 0 (+) 0
         let (x', w') =
           map2 (\xx_j ww_j ->
                   let x_j = (xx_j - m_fun i) |> T.i64
                   -- tricube
                   let r = T.abs x_j
                   let tmp1 = r / lambda_i
                   let tmp2 = 1.0 - tmp1 * tmp1 * tmp1
                   let tmp3 = tmp2 * tmp2 * tmp2
                   -- scale by user-defined weights
                   let tmp4 = tmp3 * ww_j
                   in (x_j, tmp4)
                ) xx_slice' ww_slice' |> unzip2
         -- then, compute fit and slope based on polynomial degree
         let xw' = map2 (*) x' w'
         let x2w' = map2 (*) x' xw'
         let yy_slice' = q_slice' yy l_idx_i 0 (+) 0

         let fit = map4 (
                     \w_j yy_j xw_j x2w_j ->
                       fit_fun w_j yy_j xw_j x2w_j a0 a11 a12 b11 b12 c12
                   ) w' yy_slice' xw' x2w' |> T.sum

         let slope = map4 (
                     \w_j yy_j xw_j x2w_j ->
                       slope_fun w_j yy_j xw_j x2w_j c2 a11 a2 b11 b2 c11
                   ) w' yy_slice' xw' x2w' |> T.sum

         in (fit, slope)
       ) (iota n_m) l_idx lambda |> unzip

let loess_outer_l [m] [n] [n_m] (xx_l: [m][n]i64)
                                (yy_l: [m][n]t)
                                (q: i64)
                                (m_fun: i64 -> i64)
                                (ww_l: [m][n]t)
                                (l_idx_l: [m][n_m]i64)
                                (lambda_l: [m][n_m]t)
                                (n_nn_l: [m]i64)
                                (fit_fun: fun_t)
                                (slope_fun: fun_t)
                                : ([m][n_m]t, [m][n_m]t) =
  #[incremental_flattening(no_intra)]
  map5 (\xx yy ww l_idx (lambda, n_nn) ->
          loess_outer xx
                      yy
                      q
                      m_fun
                      ww
                      l_idx
                      lambda
                      n_nn
                      fit_fun
                      slope_fun
       ) xx_l yy_l ww_l l_idx_l (zip lambda_l n_nn_l) |> unzip |> opaque

let loess_outer_css_l [m] [n_p] [n] [n_m] (xx_css_l: [m][n_p][n]i64)
                                          (yy_css_l: [m][n_p][n]t)
                                          (q: i64)
                                          (m_fun: i64 -> i64)
                                          (ww_css_l: [m][n_p][n]t)
                                          (l_idx_css_l: [m][n_p][n_m]i64)
                                          (lambda_css_l: [m][n_p][n_m]t)
                                          (n_nn_css_l: [m][n_p]i64)
                                          (fit_fun: fun_t)
                                          (slope_fun: fun_t)
                                          : ([m][n_p][n_m]t, [m][n_p][n_m]t) =
    #[incremental_flattening(no_intra)]
    map5 (\xx_css yy_css ww_css l_idx_css (lambda_css, n_nn_css) ->
            map5 (\xx yy ww l_idx (lambda, n_nn) ->
                    loess_outer xx
                                yy
                                q
                                m_fun
                                ww
                                l_idx
                                lambda
                                n_nn
                                fit_fun
                                slope_fun
                 ) xx_css yy_css ww_css l_idx_css (zip lambda_css n_nn_css) |> unzip
         ) xx_css_l yy_css_l ww_css_l l_idx_css_l (zip lambda_css_l n_nn_css_l) |> unzip |> opaque


--------------------------------------------------------------------------------
-- Main LOESS procedure - flat version, with extra work                       --
--------------------------------------------------------------------------------
let loess_flat [n] [n_m] (xx: [n]i64)
                         (yy: [n]t)
                         (q: i64)
                         (m_fun: i64 -> i64)
                         (ww: [n]t)
                         (l_idx: [n_m]i64)
                         (lambda: [n_m]t)
                         (n_nn: i64)
                         (fit_fun: fun_t)
                         (slope_fun: fun_t)
                         : ([n_m]t, [n_m]t) =
  let q_slice 'a (arr: [n]a) (l_idx_i: i64) (v: a) (add: a -> a -> a) (zero: a): [q]a =
    #[unsafe]
    tab (\j -> if j >= n_nn then zero else add arr[l_idx_i + j] v) q
  -- need the duplicate to prevent manifestation
  let q_slice' 'a (arr: [n]a) (l_idx_i: i64) (v: a) (add: a -> a -> a) (zero: a): [q]a =
    #[unsafe]
    tab (\j -> if j >= n_nn then zero else add arr[l_idx_i + j] v) q
  in
  -- [n_m]
  #[incremental_flattening(no_intra)]
  #[incremental_flattening(no_outer)]
  map3 (\i l_idx_i lambda_i ->
         -----------------------------------
         -- REDOMAP 1
         -----------------------------------
         #[unsafe]
         let xx_slice = q_slice xx l_idx_i 1 (+) 0
         let ww_slice = q_slice ww l_idx_i 0 (+) 0
         let (w, xw, x2w, x3w, x4w) =
           map2 (\xx_j ww_j ->
                   let x_j = (xx_j - m_fun i) |> T.i64
                   -- tricube
                   let r = T.abs x_j
                   let tmp1 = r / lambda_i
                   let tmp2 = 1.0 - tmp1 * tmp1 * tmp1
                   let tmp3 = tmp2 * tmp2 * tmp2
                   -- scale by user-defined weights
                   let w_j = tmp3 * ww_j
                   let xw_j = x_j * w_j
                   let x2w_j = x_j * xw_j
                   let x3w_j = x_j * x2w_j
                   let x4w_j = x_j * x3w_j
                   in (w_j, xw_j, x2w_j, x3w_j, x4w_j)
                ) xx_slice ww_slice |> unzip5
         -- then, compute fit and slope based on polynomial degree
         let a_0 = T.sum w
         let b_0 = T.sum xw
         let c_0 = T.sum x2w
         let d_0 = T.sum x3w
         let e_0 = T.sum x4w

         let a = a_0 + T.epsilon
         let b = b_0 + T.epsilon
         let c = c_0 + T.epsilon
         let d = d_0 + T.epsilon
         let e = e_0 + T.epsilon

         -- degree 0
         let a0 = 1 / a

         -- degree 1
         let det1 = 1 / (a * c - b * b)
         let a11 = c * det1
         let b11 = -b * det1
         let c11 = a * det1

         -- degree 2
         let a12 = e * c - d * d
         let b12 = c * d - e * b
         let c12 = b * d - c * c
         let a2 = c * d - e * b
         let b2 = e * a - c * c
         let c2 = b * c - d * a
         let det = 1 / (a * a12 + b * b12 + c * c12)
         let a12 = a12 * det
         let b12 = b12 * det
         let c12 = c12 * det
         let a2 = a2 * det
         let b2 = b2 * det
         let c2 = c2 * det

         -----------------------------------
         -- REDOMAP 2
         -----------------------------------
         let xx_slice' = q_slice' xx l_idx_i 1 (+) 0
         let ww_slice' = q_slice' ww l_idx_i 0 (+) 0
         let (x', w') =
           map2 (\xx_j ww_j ->
                   let x_j = (xx_j - m_fun i) |> T.i64
                   -- tricube
                   let r = T.abs x_j
                   let tmp1 = r / lambda_i
                   let tmp2 = 1.0 - tmp1 * tmp1 * tmp1
                   let tmp3 = tmp2 * tmp2 * tmp2
                   -- scale by user-defined weights
                   let tmp4 = tmp3 * ww_j
                   in (x_j, tmp4)
                ) xx_slice' ww_slice' |> unzip2
         -- then, compute fit and slope based on polynomial degree
         let xw' = map2 (*) x' w'
         let x2w' = map2 (*) x' xw'
         let yy_slice' = q_slice' yy l_idx_i 0 (+) 0

         let fit = map4 (
                     \w_j yy_j xw_j x2w_j ->
                       fit_fun w_j yy_j xw_j x2w_j a0 a11 a12 b11 b12 c12
                   ) w' yy_slice' xw' x2w' |> T.sum

         let slope = map4 (
                     \w_j yy_j xw_j x2w_j ->
                       slope_fun w_j yy_j xw_j x2w_j c2 a11 a2 b11 b2 c11
                   ) w' yy_slice' xw' x2w' |> T.sum

         in (fit, slope)
       ) (iota n_m) l_idx lambda |> unzip

let loess_flat_l [m] [n] [n_m] (xx_l: [m][n]i64)
                               (yy_l: [m][n]t)
                               (q: i64)
                               (m_fun: i64 -> i64)
                               (ww_l: [m][n]t)
                               (l_idx_l: [m][n_m]i64)
                               (lambda_l: [m][n_m]t)
                               (n_nn_l: [m]i64)
                               (fit_fun: fun_t)
                               (slope_fun: fun_t)
                               : ([m][n_m]t, [m][n_m]t) =
  #[incremental_flattening(no_outer)]
  #[incremental_flattening(no_intra)]
  map5 (\xx yy ww l_idx (lambda, n_nn) ->
          loess_flat xx
                     yy
                     q
                     m_fun
                     ww
                     l_idx
                     lambda
                     n_nn
                     fit_fun
                     slope_fun
       ) xx_l yy_l ww_l l_idx_l (zip lambda_l n_nn_l) |> unzip |> opaque

let loess_flat_css_l [m] [n_p] [n] [n_m] (xx_css_l: [m][n_p][n]i64)
                                         (yy_css_l: [m][n_p][n]t)
                                         (q: i64)
                                         (m_fun: i64 -> i64)
                                         (ww_css_l: [m][n_p][n]t)
                                         (l_idx_css_l: [m][n_p][n_m]i64)
                                         (lambda_css_l: [m][n_p][n_m]t)
                                         (n_nn_css_l: [m][n_p]i64)
                                         (fit_fun: fun_t)
                                         (slope_fun: fun_t)
                                         : ([m][n_p][n_m]t, [m][n_p][n_m]t) =
    #[incremental_flattening(no_outer)]
    #[incremental_flattening(no_intra)]
    map5 (\xx_css yy_css ww_css l_idx_css (lambda_css, n_nn_css) ->
            map5 (\xx yy ww l_idx (lambda, n_nn) ->
                    loess_flat xx
                               yy
                               q
                               m_fun
                               ww
                               l_idx
                               lambda
                               n_nn
                               fit_fun
                               slope_fun
                 ) xx_css yy_css ww_css l_idx_css (zip lambda_css n_nn_css) |> unzip
         ) xx_css_l yy_css_l ww_css_l l_idx_css_l (zip lambda_css_l n_nn_css_l) |> unzip |> opaque


--------------------------------------------------------------------------------
-- Main LOESS procedure - intragroup version with no extra work               --
--------------------------------------------------------------------------------
let loess_intragroup_simple [n] [n_m] (xx: [n]i64)
                                      (yy: [n]t)
                                      (q: i64)
                                      (m_fun: i64 -> i64)
                                      (ww: [n]t)
                                      (l_idx: [n_m]i64)
                                      (lambda: [n_m]t)
                                      (n_nn: i64)
                                      (fit_fun: fun_t)
                                      (slope_fun: fun_t)
                                      : ([n_m]t, [n_m]t) =
  let q_slice 'a (arr: [n]a) (l_idx_i: i64) (v: a) (add: a -> a -> a) (zero: a): [q]a =
    #[unsafe]
    -- tab (\j -> if j >= n_nn then zero else add arr[l_idx_i + j] v) q
    tab (\j -> if j >= n_nn then zero else add arr[l_idx_i + j] v) q
  in
  -- [n_m]
  #[incremental_flattening(only_intra)]
  map3 (\i l_idx_i lambda_i ->
         -- [q]
         -- get polynomial weights (from tri-cube), x, and a
         #[unsafe]
         let xx_slice = q_slice xx l_idx_i 1 (+) 0
         let ww_slice = q_slice ww l_idx_i 0 (+) 0
         let (x, w) =
           map2 (\xx_j ww_j ->
                   let x_j = (xx_j - m_fun i) |> T.i64
                   -- tricube
                   let r = T.abs x_j
                   let tmp1 = r / lambda_i
                   let tmp2 = 1.0 - tmp1 * tmp1 * tmp1
                   let tmp3 = tmp2 * tmp2 * tmp2
                   -- scale by user-defined weights
                   let tmp4 = tmp3 * ww_j
                   in (x_j, tmp4)
                ) xx_slice ww_slice |> unzip2
         -- then, compute fit and slope based on polynomial degree
         let xw = map2 (*) x w
         let x2w = map2 (*) x xw
         let x3w = map2 (*) x x2w
         let x4w = map2 (*) x x3w

         let a_0 = T.sum w
         let b_0 = T.sum xw
         let c_0 = T.sum x2w
         let d_0 = T.sum x3w
         let e_0 = T.sum x4w

         let a = a_0 + T.epsilon
         let b = b_0 + T.epsilon
         let c = c_0 + T.epsilon
         let d = d_0 + T.epsilon
         let e = e_0 + T.epsilon

         let det1 = 1 / (a * c - b * b)
         let a11 = c * det1
         let b11 = -b * det1
         let c11 = a * det1

         -- degree 2
         let a12 = e * c - d * d
         let b12 = c * d - e * b
         let c12 = b * d - c * c
         let a2 = c * d - e * b
         let b2 = e * a - c * c
         let c2 = b * c - d * a
         let det = 1 / (a * a12 + b * b12 + c * c12)
         let a12 = a12 * det
         let b12 = b12 * det
         let c12 = c12 * det
         let a2 = a2 * det
         let b2 = b2 * det
         let c2 = c2 * det

         let a0 = 1 / a

         let yy_slice = q_slice yy l_idx_i 0 (+) 0

         let fit =
           map4 (
             \w_j yy_j xw_j x2w_j ->
               fit_fun w_j yy_j xw_j x2w_j a0 a11 a12 b11 b12 c12
           ) w yy_slice xw x2w |> T.sum

         let slope =
           map4 (
             \w_j yy_j xw_j x2w_j ->
               slope_fun w_j yy_j xw_j x2w_j c2 a11 a2 b11 b2 c11
           ) w yy_slice xw x2w |> T.sum
         in (fit, slope)
       ) (iota n_m) l_idx lambda |> unzip

let loess_intragroup_simple_l [m] [n] [n_m] (xx_l: [m][n]i64)
                                            (yy_l: [m][n]t)
                                            (q: i64)
                                            (m_fun: i64 -> i64)
                                            (ww_l: [m][n]t)
                                            (l_idx_l: [m][n_m]i64)
                                            (lambda_l: [m][n_m]t)
                                            (n_nn_l: [m]i64)
                                            (fit_fun: fun_t)
                                            (slope_fun: fun_t)
                                            : ([m][n_m]t, [m][n_m]t) =
  #[incremental_flattening(only_inner)]
  map5 (\xx yy ww l_idx (lambda, n_nn) ->
          loess_intragroup_simple xx
                                  yy
                                  q
                                  m_fun
                                  ww
                                  l_idx
                                  lambda
                                  n_nn
                                  fit_fun
                                  slope_fun
       ) xx_l yy_l ww_l l_idx_l (zip lambda_l n_nn_l) |> unzip |> opaque

let loess_intragroup_simple_css_l [m] [n_p] [n] [n_m] (xx_css_l: [m][n_p][n]i64)
                                                      (yy_css_l: [m][n_p][n]t)
                                                      (q: i64)
                                                      (m_fun: i64 -> i64)
                                                      (ww_css_l: [m][n_p][n]t)
                                                      (l_idx_css_l: [m][n_p][n_m]i64)
                                                      (lambda_css_l: [m][n_p][n_m]t)
                                                      (n_nn_css_l: [m][n_p]i64)
                                                      (fit_fun: fun_t)
                                                      (slope_fun: fun_t)
                                                      : ([m][n_p][n_m]t, [m][n_p][n_m]t) =
    #[incremental_flattening(only_inner)]
    map5 (\xx_css yy_css ww_css l_idx_css (lambda_css, n_nn_css) ->
            #[incremental_flattening(only_inner)]
            map5 (\xx yy ww l_idx (lambda, n_nn) ->
                    loess_intragroup_simple xx
                                            yy
                                            q
                                            m_fun
                                            ww
                                            l_idx
                                            lambda
                                            n_nn
                                            fit_fun
                                            slope_fun
                 ) xx_css yy_css ww_css l_idx_css (zip lambda_css n_nn_css) |> unzip
         ) xx_css_l yy_css_l ww_css_l l_idx_css_l (zip lambda_css_l n_nn_css_l) |> unzip |> opaque
 
--------------------------------------------------------------------------------
-- Lifted LOESS wrapper, choose and run the correct code version for each ts  --
--------------------------------------------------------------------------------
let loess_l [m] [n] [n_m] (xx_l: [m][n]i64)
                          (yy_l: [m][n]t)
                          (degree: i64)
                          (q: i64)
                          (m_fun: i64 -> i64)
                          (ww_l: [m][n]t)
                          (l_idx_l: [m][n_m]i64)
                          (lambda_l: [m][n_m]t)
                          (n_nn_l: [m]i64)
                          (jump: i64)
                          (jump_threshold_1: i64)
                          (jump_threshold_2: i64)
                          (q_threshold_1: i64)
                          (q_threshold_2: i64)
                          : ([m][n_m]t, [m][n_m]t) =
  let use_version (loess_proc: loess_t[m][n][n_m]): ([m][n_m]t, [m][n_m]t) =
    let loess_l_fun (fit_fu: fun_t) (slope_fu: fun_t): ([m][n_m]t, [m][n_m]t) =
      loess_proc xx_l yy_l q m_fun ww_l l_idx_l lambda_l n_nn_l fit_fu slope_fu
    in
    match degree
    -- choose version, based on polynomial degree
    case 0 -> loess_l_fun fit_fun_zero slope_fun_zero |> opaque
    case 1 -> loess_l_fun fit_fun_one  slope_fun_one  |> opaque
    case _ -> loess_l_fun fit_fun_two  slope_fun_two  |> opaque
  in
  -- choose version, based on value of q and jump
  if jump < jump_threshold_1 || jump < jump_threshold_2 && q < q_threshold_2 then
    use_version loess_outer_l
  else if jump >= jump_threshold_2 && q < q_threshold_1 then
    use_version loess_intragroup_simple_l
  else
    use_version loess_flat_l

--------------------------------------------------------------------------------
-- Lifted LOESS wrapper for the cycle subseries smoothing                     --
--------------------------------------------------------------------------------
let loess_css_l [m] [n_p] [n] [n_m] (xx_css_l: [m][n_p][n]i64)
                                    (yy_css_l: [m][n_p][n]t)
                                    (degree: i64)
                                    (q: i64)
                                    (m_fun: i64 -> i64)
                                    (ww_css_l: [m][n_p][n]t)
                                    (l_idx_css_l: [m][n_p][n_m]i64)
                                    (lambda_css_l: [m][n_p][n_m]t)
                                    (n_nn_css_l: [m][n_p]i64)
                                    (jump: i64)
                                    (jump_threshold_1: i64)
                                    (jump_threshold_2: i64)
                                    (q_threshold_1: i64)
                                    (q_threshold_2: i64)
                                    : ([m][n_p][n_m]t, [m][n_p][n_m]t) =
  let use_version (loess_proc: loess_css_t[m][n_p][n][n_m]): ([m][n_p][n_m]t, [m][n_p][n_m]t) =
    let loess_css_l_fun (fit_fu: fun_t) (slope_fu: fun_t): ([m][n_p][n_m]t, [m][n_p][n_m]t) =
      loess_proc xx_css_l yy_css_l q m_fun ww_css_l l_idx_css_l lambda_css_l n_nn_css_l fit_fu slope_fu
    in
    match degree
    -- choose version, based on polynomial degree
    case 0 -> loess_css_l_fun fit_fun_zero slope_fun_zero |> opaque
    case 1 -> loess_css_l_fun fit_fun_one  slope_fun_one  |> opaque
    case _ -> loess_css_l_fun fit_fun_two  slope_fun_two  |> opaque
  in
  -- choose version, based on value of q and jump
  if jump < jump_threshold_1 || jump < jump_threshold_2 && q < q_threshold_2 then
    use_version loess_outer_css_l
  else if jump >= jump_threshold_2 && q < q_threshold_1 then
    use_version loess_intragroup_simple_css_l
  else
    use_version loess_flat_css_l

--------------------------------------------------------------------------------
-- Find q nearest neighbors and return index of the leftmost one              --
--- index array must be sorted, sequential version                            --
--------------------------------------------------------------------------------
let l_indexes [N] (nn_idx: [N]i64)
                  (m_fun: i64 -> i64)
                  (n_m: i64)
                  (q: i64)
                  (n_nn: i64): [n_m]i64 =
  -- [n_m]
  tab(\i ->
        let x = m_fun i
        -- use binary search to find the nearest idx
        let (init_idx, _) =
          -- O(log N)
          loop (low, high) = (0i64, N - 1) while low <= high do
            let mid = (low + high) / 2
            let mid_id = nn_idx[mid]
            let mid_idx = if mid_id < 0 then i64.highest else mid_id
            in
            if mid_idx >= x then (low, mid - 1) else (mid + 1, high)
        let (idx, _, _) =
          -- find the neighbor interval, starting at init_idx
          loop (l_idx, r_idx, span) = (init_idx, init_idx, 1) while span < q do
            -- O(q)
            let l_cand = i64.max (l_idx - 1) 0
            let r_cand = i64.min (r_idx + 1) (n_nn - 1)
            let l_dist = i64.abs (nn_idx[l_cand] - x)
            let r_dist = i64.abs (nn_idx[r_cand] - x)
            in
            if l_cand == l_idx
              then (l_idx, r_idx, q)         -- leftmost found, return
            else if l_dist < r_dist || r_cand == r_idx
              then (l_cand, r_idx, span + 1) -- expand to the left
            else (l_idx, r_cand, span + 1)   -- expand to the right
        let res_idx = i64.max (i64.min (n_nn - q) idx) 0
        in res_idx
     ) n_m

--------------------------------------------------------------------------------
-- Find lambda_q (distnce between x and qth nearest neighbor)                 --
--------------------------------------------------------------------------------
let find_lambda [n_m] (y_idx: []i64)
                        (l_idx: [n_m]i64)
                        (m_fun: i64 -> i64)
                        (q: i64)
                        (n_nn: i64) : [n_m]t=
  map2 (\l i ->
          let mv = m_fun i
          let q' = i64.min q n_nn
          let r = l + q' - 1
          let md_i = i64.max (i64.abs (y_idx[l] - mv))
                             (i64.abs (y_idx[r] - mv)) |> T.i64
          in
          md_i + T.max (((T.i64 q) - (T.i64 n_nn)) / 2) 0
       ) l_idx (iota n_m)


--------------------------------------------------------------------------------
-- Calculate parameters for the main LOESS procedure                          --
--------------------------------------------------------------------------------
let loess_params [N] (q: i64)
                     (m_fun: i64 -> i64)
                     (n_m: i64)
                     (y_idx: [N]i64)
                     (n_nn: i64)
                     : ([n_m]i64, [n_m]t) =
  let y_idx_p1 = (y_idx |> map (+1))
  let q3 = i64.min q N
  -- [n_m]
  let l_idx = l_indexes y_idx_p1 (m_fun >-> (+1)) n_m q3 n_nn
  let lambda = find_lambda y_idx l_idx m_fun q n_nn
  in (l_idx, lambda)


let loess_params_css [N] (q: i64)
                         (m_fun: i64 -> i64)
                         (n_m: i64)
                         (y_idx: [N]i64)
                         (n_nn: i64)
                         : ([n_m]i64, [n_m]t) =
  let y_idx_p1 = (y_idx |> map (+1))
  let q3 = i64.min q N
  -- [n_m]
  let l_idx = l_indexes y_idx_p1 m_fun n_m q3 n_nn
  let lambda = find_lambda y_idx_p1 l_idx m_fun q n_nn
in (l_idx, lambda)


--------------------------------------------------------------------------------
-- Cubic Hermite Interpolator                                                 --
--------------------------------------------------------------------------------
let interpolate_proc [n_m] (a: i64)
                           (j: i64)
                           (m_fun: i64 -> i64)
                           (fits: [n_m]t)
                           (slopes: [n_m]t): t =
  let m_j = m_fun j
  let h = T.i64 (m_fun (j + 1) - m_j)
  let u = (T.i64 (a - m_j)) / h
  let u2 = u * u
  let u3 = u2 * u
  in
  (2 * u3 - 3 * u2 + 1) * fits[j] +
  (3 * u2 - 2 * u3)     * fits[j + 1] +
  (u3 - 2 * u2 + u)     * slopes[j] * h +
  (u3 - u2)             * slopes[j + 1] * h

let interpolate [n_m] (m_fun: i64 -> i64)
                      (fits: [n_m]t)
                      (slopes: [n_m]t)
                      (N: i64)
                      (jump: i64): [N]t =
  tab (\a ->
         let m_v = a / jump
         let j = if m_v == n_m - 1 then m_v - 1 else m_v
         in interpolate_proc a j m_fun fits slopes
      ) N

let interpolate_css [n_m] (m_fun: i64 -> i64)
                          (fits: [n_m]t)
                          (slopes: [n_m]t)
                          (N: i64)
                          (jump: i64): [N]t =
  tab (\a ->
         if a == 0 then
           head fits
         else if a == N - 1 then
           last fits
         else
           let m_v = (a - 1) / jump + 1
           let j = if a == 0 then 0 else if m_v == n_m - 1 then m_v - 1 else m_v
           in interpolate_proc a j m_fun fits slopes
      ) N
}


--------------------------------------------------------------------------------
-- Entry Point for the LOESS application                                      --
--------------------------------------------------------------------------------
entry main [m] [n] (Y: [m][n]f64)
                   (q: i64)
                   (degree: i64)
                   (jump: i64)
                   (jump_threshold_1: i64)
                   (jump_threshold_2: i64)
                   (q_threshold_1: i64)
                   (q_threshold_2: i64): [m][n]f64 =
  -- set up parameters for the low-pass filter smoothing
  let n_m = if jump == 1 then n else n / jump + 1
  let m_fun (x: i64): i64 = i64.min (x * jump) (n - 1)

  -- filter nans and pad non-nan indices
  let (nn_y_l, nn_idx_l, n_nn_l) =
    map (filterPadWithKeys (\i -> !(f64.isnan i)) 0) Y |> unzip3 |> opaque

  -- calculate invariant arrays for the low-pass filter smoothing
  let (l_idx_l, lambda_l) =
    map2 (\nn_idx n_nn ->
            loess_m.loess_params q m_fun n_m nn_idx n_nn
         ) nn_idx_l n_nn_l |> unzip |> opaque

  let weights_l = replicate (m * n) 1f64 |> unflatten m n
  let (results_l, slopes_l) = loess_m.loess_l nn_idx_l
                                              nn_y_l
                                              degree
                                              q
                                              m_fun
                                              weights_l
                                              l_idx_l
                                              lambda_l
                                              n_nn_l
                                              jump
                                              jump_threshold_1
                                              jump_threshold_2
                                              q_threshold_1
                                              q_threshold_2 |> opaque
  in
  if jump > 1 then
    map2 (\results slopes ->
            loess_m.interpolate m_fun results slopes n jump
         ) results_l slopes_l
  else
    results_l :> [m][n]f64

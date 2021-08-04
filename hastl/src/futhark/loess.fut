-- Implementation of LOESS with missing values
import "utils"

module loess_m = {
module T = f32
type t = T.t
type^ fun_t = (t -> t -> t -> t -> t -> t -> t -> t -> t -> t -> t)


let fit_fun_zero (w_j: t) (yy_j: t) (_: t) (_: t) (a0: t) (_: t) (_: t) (_: t) (_: t) (_: t): t =
  w_j * a0 * yy_j

let fit_fun_one (w_j: t) (yy_j: t) (xw_j: t) (_: t) (_: t) (a11: t) (_: t) (b11: t) (_: t) (_: t): t =
  (w_j * a11 + xw_j * b11) * yy_j

let fit_fun_two (w_j: t) (yy_j: t) (xw_j: t) (x2w_j: t) (_: t) (_: t) (a12: t) (_: t) (b12: t) (c12: t): t =
  (w_j * a12 + xw_j * b12 + x2w_j * c12) * yy_j


let slope_fun_zero (_: t) (_: t) (_: t) (_: t) (_: t) (_: t) (_: t) (_: t) (_: t) (_: t): t =
  T.i64 0

let slope_fun_one (w_j: t) (yy_j: t) (xw_j: t) (_: t) (_: t) (_: t) (_: t) (b11: t) (_: t) (c11: t): t =
  (w_j * b11 + xw_j * c11) * yy_j

let slope_fun_two (w_j: t) (yy_j: t) (xw_j: t) (x2w_j: t) (c2: t) (_: t) (a2: t) (_: t) (b2: t) (_: t): t =
  (w_j * a2 + xw_j * b2 + x2w_j * c2) * yy_j


let loess_flat [n] [n_m] (xx: [n]t)
                         (yy: [n]t)
                         (q: i64)
                         (m_fun: i64 -> i64)
                         (ww: [n]t)
                         (l_idx: [n_m]i64)
                         (max_dist: [n_m]t)
                         (n_nn: i64)
                         (fit_fun: fun_t)
                         (slope_fun: fun_t)
                         : ([n_m]t, [n_m]t) =
  let q_slice (arr: [n]t) (l_idx_i: i64) (v: t): [q]t =
    #[unsafe]
    tabulate q (\j -> if j >= n_nn then (T.i64 0) else arr[l_idx_i + j] + v)
  -- need the duplicate to prevent manifestation
  let q_slice' (arr: [n]t) (l_idx_i: i64) (v: t): [q]t =
    #[unsafe]
    tabulate q (\j -> if j >= n_nn then (T.i64 0) else arr[l_idx_i + j] + v)
  in
  -- [n_m]
  #[incremental_flattening(no_intra)]
  map3 (\i l_idx_i max_dist_i ->
         -----------------------------------
         -- REDOMAP 1
         -----------------------------------
         #[unsafe]
         let xx_slice = q_slice xx l_idx_i 1
         let ww_slice = q_slice ww l_idx_i 0
         let (w, xw, x2w, x3w, x4w) =
           map2 (\xx_j ww_j ->
                   let x_j = xx_j - (m_fun i |> T.i64)
                   -- tricube
                   let r = T.abs x_j
                   let tmp1 = r / max_dist_i
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
         let a = T.sum w
         let b = T.sum xw
         let c = T.sum x2w
         let d = T.sum x3w
         let e = T.sum x4w

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
         let xx_slice' = q_slice' xx l_idx_i 1
         let ww_slice' = q_slice' ww l_idx_i 0
         let (x', w') =
           map2 (\xx_j ww_j ->
                   let x_j = xx_j - (m_fun i |> T.i64)
                   -- tricube
                   let r = T.abs x_j
                   let tmp1 = r / max_dist_i
                   let tmp2 = 1.0 - tmp1 * tmp1 * tmp1
                   let tmp3 = tmp2 * tmp2 * tmp2
                   -- scale by user-defined weights
                   let tmp4 = tmp3 * ww_j
                   in (x_j, tmp4)
                ) xx_slice' ww_slice' |> unzip2
         -- then, compute fit and slope based on polynomial degree
         let xw' = map2 (*) x' w'
         let x2w' = map2 (*) x' xw'
         let yy_slice' = q_slice' yy l_idx_i 0

         let fit = map4 (
                     \w_j yy_j xw_j x2w_j ->
                       fit_fun w_j yy_j xw_j x2w_j a0 a11 a12 b11 b12 c12
                   ) w' yy_slice' xw' x2w' |> T.sum

         let slope = map4 (
                     \w_j yy_j xw_j x2w_j ->
                       slope_fun w_j yy_j xw_j x2w_j c2 a11 a2 b11 b2 c11
                   ) w' yy_slice' xw' x2w' |> T.sum

         in (fit, slope)
       ) (iota n_m) l_idx max_dist |> unzip

let loess_flat_l [m] [n] [n_m] (xx_l: [m][n]t)
                               (yy_l: [m][n]t)
                               (q: i64)
                               (m_fun: i64 -> i64)
                               (ww_l: [m][n]t)
                               (l_idx_l: [m][n_m]i64)
                               (max_dist_l: [m][n_m]t)
                               (n_nn_l: [m]i64)
                               (fit_fun: fun_t)
                               (slope_fun: fun_t)
                               : ([m][n_m]t, [m][n_m]t) =
  #[incremental_flattening(no_intra)]
  map5 (\xx yy ww l_idx (max_dist, n_nn) ->
          loess_flat xx
                     yy
                     q
                     m_fun
                     ww
                     l_idx
                     max_dist
                     n_nn
                     fit_fun
                     slope_fun
       ) xx_l yy_l ww_l l_idx_l (zip max_dist_l n_nn_l) |> unzip


let loess_intragroup_simple [n] [n_m] (xx: [n]t)
                                      (yy: [n]t)
                                      (q: i64)
                                      (m_fun: i64 -> i64)
                                      (ww: [n]t)
                                      (l_idx: [n_m]i64)
                                      (max_dist: [n_m]t)
                                      (n_nn: i64)
                                      (fit_fun: fun_t)
                                      (slope_fun: fun_t)
                                      : ([n_m]t, [n_m]t) =
  let q_slice (arr: [n]t) (l_idx_i: i64) (v: t): [q]t =
    #[unsafe]
    tabulate q (\j -> if j >= n_nn then (T.i64 0) else arr[l_idx_i + j] + v)
  in
  -- [n_m]
  #[incremental_flattening(only_intra)]
  map3 (\i l_idx_i max_dist_i ->
         -- [q]
         -- get polynomial weights (from tri-cube), x, and a
         #[unsafe]
         let xx_slice = q_slice xx l_idx_i 1
         let ww_slice = q_slice ww l_idx_i 0
         let (x, w) =
           map2 (\xx_j ww_j ->
                   let x_j = xx_j - (m_fun i |> T.i64)
                   -- tricube
                   let r = T.abs x_j
                   let tmp1 = r / max_dist_i
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

         let a = T.sum w
         let b = T.sum xw
         let c = T.sum x2w
         let d = T.sum x3w
         let e = T.sum x4w

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

         let yy_slice = q_slice yy l_idx_i 0

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
       ) (iota n_m) l_idx max_dist |> unzip

let loess_intragroup_simple_l [m] [n] [n_m] (xx_l: [m][n]t)
                                            (yy_l: [m][n]t)
                                            (q: i64)
                                            (m_fun: i64 -> i64)
                                            (ww_l: [m][n]t)
                                            (l_idx_l: [m][n_m]i64)
                                            (max_dist_l: [m][n_m]t)
                                            (n_nn_l: [m]i64)
                                            (fit_fun: fun_t)
                                            (slope_fun: fun_t)
                                            : ([m][n_m]t, [m][n_m]t) =
  #[incremental_flattening(only_inner)]
  map5 (\xx yy ww l_idx (max_dist, n_nn) ->
          loess_intragroup_simple xx
                                  yy
                                  q
                                  m_fun
                                  ww
                                  l_idx
                                  max_dist
                                  n_nn
                                  fit_fun
                                  slope_fun
       ) xx_l yy_l ww_l l_idx_l (zip max_dist_l n_nn_l) |> unzip


let loess_l [m] [n] [n_m] (xx_l: [m][n]t)
                          (yy_l: [m][n]t)
                          (degree: i64)
                          (q: i64)
                          (m_fun: i64 -> i64)
                          (ww_l: [m][n]t)
                          (l_idx_l: [m][n_m]i64)
                          (max_dist_l: [m][n_m]t)
                          (n_nn_l: [m]i64)
                          (jump: i64)
                          (jump_threshold: i64)
                          (q_threshold: i64)
                          : ([m][n_m]t, [m][n_m]t) =
  let use_version (loess_proc: [m][n]t ->
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
                               : ([m][n_m]t, [m][n_m]t) =
    let loess_l_fun (fit_fun: fun_t) (slope_fun: fun_t): ([m][n_m]t, [m][n_m]t) =
      loess_proc xx_l
                 yy_l
                 q
                 m_fun
                 ww_l
                 l_idx_l
                 max_dist_l
                 n_nn_l
                 fit_fun
                 slope_fun
    in
    match degree
    case 0 -> loess_l_fun fit_fun_zero slope_fun_zero |> opaque
    case 1 -> loess_l_fun fit_fun_one  slope_fun_one |> opaque
    case _ -> loess_l_fun fit_fun_two  slope_fun_two |> opaque
  in
  if jump < jump_threshold || q > q_threshold then
    use_version loess_flat_l
  else
    use_version loess_intragroup_simple_l

--------------------------------------------------------------------------------
-- Find q nearest neighbors and return index of the leftmost one              --
-- arraysmust be sorted, sequential version                                   --
--------------------------------------------------------------------------------
let l_indexes [N] (nn_idx: [N]i64) (m_fun: i64 -> i64) (n_m: i64) (q: i64) (n_nn: i64): [n_m]i64 =
  -- set invalid indexes to max long, so they would be ignored
  let pad_idx = map (\i -> if i < 0 then i64.highest else i) nn_idx
  in
  -- [n_m]
  tabulate n_m (\i ->
         let x = m_fun i
         -- use binary search to find the nearest idx
         let (nearest_idx, _) =
           -- O(log N)
           loop (low, high) = (0i64, N - 1) while low <= high do
               let mid = (low + high) / 2
               in
               if pad_idx[mid] >= x
               then (low, mid - 1)
               else (mid + 1, high)
         -- find initial sum of k distances to k
         let init_idx = i64.max 0 (nearest_idx - q)
         -- O(q)
         -- replace with loop?
         let init_sum = nn_idx[init_idx:(init_idx + q)] |> map (\x_i -> i64.abs (x_i - x)) |> i64.sum
         -- check q possible sums and find the index, corresponding to the lowest
         let (idx, _, _, _) =
           -- O(q)
           loop (best_idx, last_idx, best_sum, curr_sum) =
                (init_idx, init_idx, init_sum, init_sum) for j < q do
             -- move frame 1 to right
             let new_idx = init_idx + j
             let new_sum = if new_idx + q > n_nn
                           then i64.highest
                           -- if valid, calculate the sum of distances for the new frame
                           else curr_sum
                                - (i64.abs <| x - nn_idx[last_idx])
                                + (i64.abs <| x - nn_idx[i64.min (last_idx + q) (N - 1)])
             in
             if new_sum <= best_sum
             then
               (new_idx, new_idx, new_sum, new_sum)
             else
               (best_idx, new_idx, best_sum, new_sum)

         let res_idx = i64.min (n_nn - q) idx
         in res_idx
      )

--------------------------------------------------------------------------------
-- Find lambda_q                                                              --
--------------------------------------------------------------------------------
let find_max_dist [n_m] (y_idx: []i64)
                        (l_idx: [n_m]i64)
                        (m_fun: i64 -> i64)
                        (q: i64)
                        (q3: i64)
                        (n: i64) : [n_m]t=
  -- [n_m]
  map2 (\l i ->
          let mv = m_fun i
          let r = l + q3 - 1
          let md_i = i64.max (i64.abs (y_idx[l] - mv)) (i64.abs (y_idx[r] - mv)) |> T.i64
          in
          md_i + T.max (((T.i64 q) - (T.i64 n)) / 2) 0
       ) l_idx (iota n_m)

let loess_params [N] (q: i64)         -- should be odd
                     (m_fun: i64 -> i64)
                     (n_m: i64)
                     (y_idx: [N]i64)  -- indexes of non-nan vals in y
                     (n: i64)         -- number of non-nan vals
                     : ([n_m]i64, [n_m]t) =
  let y_idx_p1 = (y_idx |> map (+1))
  let q3 = i64.min q N
  -- [n_m]
  let l_idx = l_indexes y_idx_p1 (m_fun >-> (+1)) n_m q3 n
  let max_dist = find_max_dist y_idx l_idx m_fun q q3 n
  in (l_idx, max_dist)


--------------------------------------------------------------------------------
-- Cubic Hermite Interpolator                                                 --
--------------------------------------------------------------------------------
let interpolate [n_m] (m_fun: i64 -> i64)
                            (fits: [n_m]t)
                            (slopes: [n_m]t)
                            (N: i64)
                            (jump: i64): [N]t =
  tabulate N (\a ->
                let m_v = a / jump
                let j = if m_v == n_m - 1 then m_v - 1 else m_v
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
             )
}

entry main [m] [n] (Y: [m][n]f32)
                   (q: i64)
                   (degree: i64)
                   (jump: i64)
                   (jump_threshold: i64)
                   (q_threshold: i64): [m][n]f32 =
  -- set up parameters for the low-pass filter smoothing
  let n_m = n / jump + 1
  let m_fun (x: i64): i64 = i64.min (x * jump) (n - 1)

  -- filter nans and pad non-nan indices
  let (nn_y_l, nn_idx_l, n_nn_l) = map (filterPadWithKeys (\i -> !(f32.isnan i)) 0) Y |> unzip3

  -- calculate invariant arrays for the low-pass filter smoothing
  let (l_idx_l, max_dist_l) =
    map2 (\nn_idx n_nn ->
            loess_m.loess_params q m_fun n_m nn_idx n_nn
         ) nn_idx_l n_nn_l |> unzip

  let weights_l = replicate (m * n) 1f32 |> unflatten m n
  let nn_idx_f_l = map (map f32.i64) nn_idx_l
  let (results_l, slopes_l) = loess_m.loess_l nn_idx_f_l
                                         nn_y_l
                                         degree
                                         q
                                         m_fun
                                         weights_l
                                         l_idx_l
                                         max_dist_l
                                         n_nn_l
                                         jump
                                         jump_threshold
                                         q_threshold
  in
  if jump > 1 then
    map2 (\results slopes ->
            loess_m.interpolate m_fun results slopes n jump
         ) results_l slopes_l
  else
    results_l :> [m][n]f32

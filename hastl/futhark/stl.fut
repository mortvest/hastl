-- Seasonal-Trend decomposition by LOESS (STL) - Periodic Version
--- Loosely based on:
---- https://github.com/hafen/stlplus
-- ==
-- compiled input @ batchednan.in

-- input @ flat_small_per.in output @ flat_small_per.out
-- input @ batch_small_per.in output @ batch_small_per.out

-- compiled input @ flatnan.in
-- compiled input @ co2nan.in
import "lib/github.com/diku-dk/sorts/radix_sort"
import "loess"
import "utils"

module stl_periodic = {

module T = f32
type t = T.t

module loess = loess_m

local let filterPadNans = filterPadWithKeys (\i -> !(T.isnan i)) 0

--------------------------------------------------------------------------------
-- Three moving averages                                                      --
--------------------------------------------------------------------------------
local let moving_averages [n] (x: [n]t) (n_p: i64): []t =
  let single_ma (n_p: i64) (n: i64) (x: []t) : [n]t =
    let ma_tmp = T.sum x[:n_p]
    let n_p_f = T.i64 n_p
        in
        tabulate n (\i ->
                      if i == 0
                      then
                        ma_tmp / n_p_f
                      else
                        (x[i + n_p - 1] - x[i - 1]) / n_p_f
               ) |> scan (+) 0
  let nn = n - n_p * 2
  in
  -- apply three moving averages
  single_ma n_p (nn + n_p + 1) x |> single_ma n_p (nn + 2) |> single_ma 3 nn


--------------------------------------------------------------------------------
-- Interpolation with slopes                                                  --
--------------------------------------------------------------------------------
local let interpolate [n_m] (m_fun: i64 -> i64)
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

--------------------------------------------------------------------------------
-- Main STL function                                                          --
--------------------------------------------------------------------------------
let stl [m] [n] (Y: [m][n]t)
                (n_p: i64)
                (t_window: i64)
                (l_window: i64)
                (t_degree: i64)
                (l_degree: i64)
                (t_jump: i64)
                (l_jump: i64)
                (inner: i64)
                (outer: i64)
                (jump_threshold: i64)
                (max_group_size: i64)
                : ([m][n]t, [m][n]t, [m][n]t) =

  ------------------------------------------------------------------------------
  -- PARAMETER SETUP                                                          --
  ------------------------------------------------------------------------------
  -- check arguments
  let n_p = assert (n_p >= 4) n_p

  let t_window = check_win t_window
  let l_window = check_win l_window

  let t_degree = check_deg t_degree
  let l_degree = check_deg l_degree

  let t_jump = check_nonneg t_jump
  let l_jump = check_nonneg l_jump

  let inner = check_pos inner
  let outer = check_nonneg outer

  -- set up parameters for cycle-subseries averaging
  let max_css_len = T.to_i64 <| T.ceil <| ((T.i64 n) / (T.i64 n_p))
  let css_chunk_len = max_css_len + 2
  let C_len = n + 2 * n_p

  -- set up parameters for the low-pass filter smoothing
  let l_n_m = if l_jump == 1 then n else n / l_jump + 1
  let l_m_fun (x: i64): i64 = i64.min (x * l_jump) (n - 1)

  -- set up parameters for trend smoothing
  let t_n_m = if t_jump == 1 then n else n / t_jump + 1
  let t_m_fun (x: i64): i64 = i64.min (x * t_jump) (n - 1)

  -- set up parameters for median calculation
  let mid1 = T.floor ((T.i64 n) / 2 + 1) |> T.to_i64
  let mid2 = n - mid1 + 1
  let (mid1, mid2) = if mid1 == mid2
                     then (mid1 - 1, mid1)
                     else ((i64.min mid1 mid2) - 1, i64.max mid1 mid2)

  ------------------------------------------------------------------------------
  -- LOOP-INVARIANT VALUES                                                    --
  ------------------------------------------------------------------------------
  -- filter nans and pad non-nan indices
  let (_, nn_idx_l, n_nn_l) = map filterPadNans Y |> unzip3 |> opaque
  let nn_idx_f_l = map (\nn_idx -> map (\i -> T.i64 i) nn_idx) nn_idx_l |> opaque

  -- calculate invariant arrays for the cycle sub-series averaging
  let css_idxs =
    -- [n_p]
    tabulate n_p
             (\css_i ->
               -- [css_chunk_len]
               tabulate css_chunk_len (\j ->
                                         let res_idx = css_i + (j * n_p)
                                         in
                                         if res_idx > n - 1
                                         then -1
                                         else res_idx
                                      )
             ) |> opaque

  -- find number of non-nan values in each subseries of each time series
  let css_n_nn_l =
    map (\y ->
            -- [n_p]
            map (\css_idx ->
                  -- [css_chunk_len]
                  let vals = pad_gather y css_idx T.nan
                  let n_nn = map (\v -> if T.isnan v then 0 else 1) vals
                  let n_nn_sum_ = T.sum n_nn
                  let sub_series_is_not_all_nan = n_nn_sum_ > 0
                  -- Fail for all-zero subseries - nothing can be done here
                  let n_nn_sum = assert sub_series_is_not_all_nan n_nn_sum_
                  in n_nn_sum
                ) css_idxs
        ) Y |> opaque

  -- calculate invariant arrays for the low-pass filter smoothing
  let (l_l_idx_l, l_max_dist_l) =
    map2 (\nn_idx n_nn ->
            -- [l_n_m]
            loess.loess_params l_window l_m_fun l_n_m nn_idx n_nn
         ) nn_idx_l n_nn_l |> unzip |> opaque

  -- calculate invariant arrays for the trend smoothing
  let (t_l_idx_l, t_max_dist_l) =
    map2 (\nn_idx n_nn ->
            -- [t_n_m]
            loess.loess_params t_window t_m_fun t_n_m nn_idx n_nn
         ) nn_idx_l n_nn_l |> unzip |> opaque

  --- initialize components to 0s
  let seasonal_l = replicate (m * n) (T.i64 0) |> unflatten m n
  let trend_l = replicate (m * n) (T.i64 0) |> unflatten m n
  let weights_l = replicate (m * n) (T.i64 1) |> unflatten m n

  ------------------------------------------------------------------------------
  -- NESTED LOOPS                                                             --
  ------------------------------------------------------------------------------
  let (seasonal_l, trend_l, _) =
    ------------------------------------
    -- Outer loop start               --
    ------------------------------------
    loop (seasonal_l, trend_l, weights_l) for i_outer < outer do
      let (seasonal_l, trend_l) =
        ------------------------------------
        -- Inner loop start               --
        ------------------------------------
        loop (_, trend_l) = (seasonal_l, trend_l) for _i_inner < inner do
          -- Step 1: Detrending
          let Y_detrended_l = map2 (\y trend ->
                                      -- [n]
                                      map2 (-) y trend
                                   ) Y trend_l |> opaque

          -- Step 2: Averaging cycle-subseries
          --- find averages for each cycle sub-series
          let css_avgs_l =
            map2 (\y_detrended css_n_nns ->
                    -- [n_p]
                    map2 (\css_idx css_n_nn ->
                          -- [css_chunk_len]
                          let vals = pad_gather y_detrended css_idx T.nan
                          let filt = map (\v -> if T.isnan v then 0 else v) vals
                          in (T.sum filt / css_n_nn)
                        ) css_idxs css_n_nns
                 ) Y_detrended_l css_n_nn_l |> opaque

          -- combine chunks into one array
          let C_l =
            map (\css_avgs ->
                    -- [C_len]
                    tabulate C_len (\i -> css_avgs[i % n_p])
            ) css_avgs_l |> opaque

          -- Step 3: Low-pass filtering of collection of all the cycle-subseries
          --- apply 3 moving averages
          let ma3_l = map (\C ->
                             moving_averages C n_p :> [n]t
                          ) C_l |> opaque


          --- then apply LOESS
          let (ma3_pad_l, w_pad_l) =
            map3 ( \ma3 nn_idx weights ->
                     -- [n_nn]
                     let ma3_pad = pad_gather ma3 nn_idx 0
                     let w_pad = pad_gather weights nn_idx 0
                     in (ma3_pad, w_pad)
                 ) ma3_l nn_idx_l weights_l |> unzip |> opaque

          let (l_results_l, l_slopes_l) = loess.loess_l nn_idx_f_l
                                                        ma3_pad_l
                                                        l_degree
                                                        l_window
                                                        t_m_fun
                                                        w_pad_l
                                                        l_l_idx_l
                                                        l_max_dist_l
                                                        n_nn_l
                                                        l_jump
                                                        jump_threshold
                                                        max_group_size
          let L_l =
            if l_jump > 1 then
              map2 (\l_results l_slopes ->
                    -- [n]
                    interpolate l_m_fun l_results l_slopes n l_jump
                   ) l_results_l l_slopes_l
            else
              l_results_l :> [m][n]t

          -- Step 4: Detrend smoothed cycle-subseries
          --- can be fused with above
          let seasonal_l =
            map2 (\C L ->
                    -- [n]
                    let c_slice = C[n_p:(n + n_p)] :> [n]t
                    in
                    map2 (-) c_slice L
                 ) C_l L_l

          -- Step 5: Deseasonalize
          --- can be fused with above
          let D_l =
            map2 (\y seasonal ->
                    -- [n]
                    map2 (-) y seasonal
                 ) Y seasonal_l

          -- Step 6: Trend Smoothing
          let D_pad_l =
            map2 (\D nn_idx ->
                    -- [n_nn]
                    pad_gather D nn_idx 0
                 ) D_l nn_idx_l |> opaque

          -- apply LOESS
          let (t_results_l, t_slopes_l) = loess.loess_l nn_idx_f_l
                                                        D_pad_l
                                                        t_degree
                                                        t_window
                                                        t_m_fun
                                                        w_pad_l
                                                        t_l_idx_l
                                                        t_max_dist_l
                                                        n_nn_l
                                                        t_jump
                                                        jump_threshold
                                                        max_group_size
          --- interpolate
          let trend_l =
            -- [n]
            if t_jump > 1 then
              map2 (\t_results t_slopes ->
                      interpolate t_m_fun t_results t_slopes n t_jump
                   ) t_results_l t_slopes_l |> opaque
            else
              t_results_l :> [m][n]t

          in (seasonal_l, trend_l)
          ------------------------------------
          -- Inner loop end                 --
          ------------------------------------

      -- validate the types
      let (seasonal_l, trend_l) = (seasonal_l, trend_l) :> ([m][n]t, [m][n]t)

      ------------------------------------
      -- Update the weights             --
      ------------------------------------
      let weights_l =
        if i_outer < outer - 1
        then
          --- calculate remainder estimate
          let R_abs_l =
            map3 (\y seasonal trend ->
                    -- [n]
                    let R = map3 (\v s t -> v - s - t) y seasonal trend
                    in
                    map (\r -> if T.isnan r then T.nan else T.abs r) R
                 ) Y seasonal_l trend_l |> opaque
          let R_pad_l =
            map2 (\R_abs nn_idx ->
                    -- [n_nn]
                    pad_gather R_abs nn_idx T.inf
                 ) R_abs_l nn_idx_l |> opaque

          let R_sorted_l = map (\R_pad ->
                                  radix_sort_float T.num_bits T.get_bit R_pad
                               ) R_pad_l |> opaque
          -- find boundaries
          let (h_l, h9_l, h1_l)=
            map (\R_sorted ->
                   -- median
                   let h = 3 * (R_sorted[mid1:mid2] |> T.sum)
                   -- boundaries
                   let h9 = 0.999 * h
                   let h1 = 0.001 * h
                   in (h, h9, h1)
                ) R_sorted_l |> unzip3 |> opaque

          -- calculate new weights
          let w_l =
            -- [n]
            map4 (\R_abs h h9 h1 ->
                 map (\r_abs ->
                        let zero_val = 10 ** (-6)
                        let bicube = (1 - (r_abs / h)**2)**2
                        in
                        if T.isnan r_abs || r_abs <= h1
                        then
                          1
                        else if r_abs >= h9 || r_abs <= zero_val
                        then
                          zero_val
                        else
                          bicube
                     ) R_abs
                 ) R_abs_l h_l h9_l h1_l |> opaque
          in w_l
        else
          weights_l
      in (seasonal_l, trend_l, weights_l)
      ------------------------------------
      -- Outer loop end                 --
      ------------------------------------

  let remainder_l = map3 (\y seasonal trend ->
                            -- [n]
                            map3 (\v s t -> v - s - t) y seasonal trend
                         ) Y seasonal_l trend_l |> opaque
  in (seasonal_l, trend_l, remainder_l)
}


entry main [m] [n] (Y: [m][n]f32)
                   (n_p: i64)
                   (t_window: i64)
                   (l_window: i64)
                   (t_degree: i64)
                   (l_degree: i64)
                   (t_jump: i64)
                   (l_jump: i64)
                   (inner: i64)
                   (outer: i64)
                   (jump_threshold: i64)
                   (max_group_size: i64)
                   : ([m][n]f32, [m][n]f32, [m][n]f32) =
  stl_periodic.stl Y
                   n_p
                   t_window
                   l_window
                   t_degree
                   l_degree
                   t_jump
                   l_jump
                   inner
                   outer
                   jump_threshold
                   max_group_size

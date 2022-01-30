-- Seasonal-Trend decomposition by LOESS (STL)
--- Loosely based on:
---- https://github.com/hafen/stlplus
-- ==
-- compiled input @ batchednan.in

import "loess"
import "utils"
import "median"

module stl_batched = {

module T = f64
type t = T.t

module loess = loess_m
module median = median_batched

--------------------------------------------------------------------------------
-- Three moving averages                                                      --
--------------------------------------------------------------------------------
local let moving_averages_l [m][n] (x_l: [m][n]t) (n_p: i64): [m][]t =
  let single_ma_l [m] (n_p: i64) (n: i64) (x_l: [m][]t) : [m][n]t =
    let n_p_f = T.i64 n_p
    let ma_tmp_l = map (\x -> T.sum x[:n_p]) x_l |> opaque
    in
    map2 (\x ma_tmp ->
            tab (\i ->
                   if i == 0
                   then
                     ma_tmp / n_p_f
                   else
                     (x[i + n_p - 1] - x[i - 1]) / n_p_f
                ) n |> scan (+) 0
         ) x_l ma_tmp_l |> opaque
  let nn = n - n_p * 2
  in
  -- apply three moving averages
  single_ma_l n_p (nn + n_p + 1) x_l |> single_ma_l n_p (nn + 2) |> single_ma_l 3 nn


local let filterPadNans = filterPadWithKeys (\i -> !(T.isnan i)) 0
local let filterPadNans32 = filterPadWithKeys (\i -> !(f32.isnan i)) 0
local let pad_gather_floats = \vs idxs -> pad_gather vs idxs (T.i64 0)
local let pad_gather_ints = \vs idxs -> pad_gather vs idxs 0i64

--------------------------------------------------------------------------------
-- Main STL function                                                          --
--------------------------------------------------------------------------------
let stl [m] [n] (Y: [m][n]f32)
                (n_p: i64)
                (q_s: i64)
                (q_t: i64)
                (q_l: i64)
                (d_s: i64)
                (d_t: i64)
                (d_l: i64)
                (jump_s: i64)
                (jump_t: i64)
                (jump_l: i64)
                (n_inner: i64)
                (n_outer: i64)
                (jump_threshold: i64)
                (q_threshold: i64) =
  ------------------------------------------------------------------------------
  -- PARAMETER SETUP                                                          --
  ------------------------------------------------------------------------------
  -- check arguments
  let n_p = assert (n_p >= 4) n_p

  let q_t = check_win q_t
  let q_l = check_win q_l

  let d_t = check_deg d_t
  let d_l = check_deg d_l

  let jump_t = check_pos jump_t
  let jump_l = check_pos jump_l

  let n_inner = check_pos n_inner
  let n_outer = check_pos n_outer

  -- set up parameters for cycle-subseries smoothing
  let max_css_len = T.ceil ((T.i64 n) / (T.i64 n_p)) |> T.to_i64
  let pad_css_len = max_css_len + 2
  let C_len = n + 2 * n_p

  let s_n_m = if jump_s == 1 then pad_css_len else max_css_len / jump_s + 3
  let s_m_fun (x: i64) = if x == 0 then 0
                         else if x == s_n_m - 1 then pad_css_len - 1
                         else i64.min ((x - 1) * jump_s + 1) (max_css_len)

  -- set up parameters for the low-pass filter smoothing
  let l_n_m = if jump_l == 1 then n else n / jump_l + 1
  let l_m_fun (x: i64): i64 = i64.min (x * jump_l) (n - 1)

  -- set up parameters for trend smoothing
  let t_n_m = if jump_t == 1 then n else n / jump_t + 1
  let t_m_fun (x: i64): i64 = i64.min (x * jump_t) (n - 1)

  ------------------------------------------------------------------------------
  -- LOOP-INVARIANT VALUES                                                    --
  ------------------------------------------------------------------------------
  -- filter nans and pad non-nan indices
  let (_, nn_idx_l, n_nn_l) = map filterPadNans32 Y |> unzip3 |> opaque

  -- extract all cycle subseries, resulting in [m][n_p][max_css_len] array
  let csss_l =
    map (\y ->
           tab ( \i ->
                   -- extract css, padded to max_css_len with a NaN value
                   tab ( \j ->
                           let new_i = i + n_p * j
                           in
                           if new_i > n - 1
                           then T.nan
                           else T.f32 y[new_i]
                       ) max_css_len
               ) n_p
        ) Y |> opaque

  -- compute local indexes of non-nan values and their quantity for each css
  let (css_nn_idxss_l, css_n_nns_l) =
    map (\csss ->
           map ( \css ->
                  let (_, css_nn_idx, css_n_nn) = filterPadNans css
                  in (css_nn_idx, css_n_nn)
               ) csss |> unzip
        ) csss_l |> unzip |> opaque

  -- calculate invariant arrays for the seasonal smoothing: nearest neighbors and lamba
  let (css_l_idxs_l, css_lambdas_l) =
    map2 (\css_nn_idxss css_n_nns ->
           map2 (\css_nn_idxs css_n_nn ->
                   loess.loess_params_css q_s s_m_fun s_n_m css_nn_idxs css_n_nn
                ) css_nn_idxss css_n_nns |> unzip
        ) css_nn_idxss_l css_n_nns_l |> unzip |> opaque

  -- calculate invariant arrays for the low-pass filter smoothing
  let (l_l_idx_l, l_lambdas_l) =
    map2 (\nn_idx n_nn ->
            -- [l_n_m]
            loess.loess_params q_l l_m_fun l_n_m nn_idx n_nn
         ) nn_idx_l n_nn_l |> unzip |> opaque

  -- calculate invariant arrays for the trend smoothing
  let (t_l_idx_l, t_lambdas_l) =
    map2 (\nn_idx n_nn ->
            -- [t_n_m]
            loess.loess_params q_t t_m_fun t_n_m nn_idx n_nn
         ) nn_idx_l n_nn_l |> unzip |> opaque

  --- initialize components to 0s, should not be instantiated
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
    loop (seasonal_l, trend_l, weights_l) for i_outer < n_outer do
      let (seasonal_l, trend_l) =
        ------------------------------------
        -- Inner loop start               --
        ------------------------------------
        loop (_, trend_l) = (seasonal_l, trend_l) for _i_inner < n_inner do
          -- Step 1: Detrending
          let Y_detrended_l = map2 (\y trend ->
                                      map2 (\v t -> (T.f32 v) - t) y trend
                                   ) Y trend_l |> opaque

          -- Step 2: Cycle subseries smoothing
          --- extract the padded non-NaN values for each css and corresponding weights
          let (css_nns_l, css_ws_l) =
            map3 (\css_nn_idxss y_detrended w ->
                    -- [n_p]
                    map2 (\i css_nn_idx ->
                            -- [css_max_len]
                           map (\nn_id ->
                                   let idx = nn_id * n_p + i
                                   in
                                   if idx > n - 1 || nn_id < 0
                                   then (0, 0)
                                   else (y_detrended[idx], w[idx])
                                ) css_nn_idx |> unzip
                         ) (iota n_p) css_nn_idxss |> unzip
                 ) css_nn_idxss_l Y_detrended_l weights_l |> unzip |> opaque

          -- apply LOESS to each css
          let (css_fits_l, css_slopes_l) = loess.loess_css_l css_nn_idxss_l
                                                             css_nns_l
                                                             d_s
                                                             q_s
                                                             s_m_fun
                                                             css_ws_l
                                                             css_l_idxs_l
                                                             css_lambdas_l
                                                             css_n_nns_l
                                                             jump_s
                                                             jump_threshold
                                                             q_threshold

          -- apply interpolation to each css if necessary. The result has inner dimension (max_css_length + 2)
          let css_results_l =
            if jump_s == 1
            then
              css_fits_l
            else
              map2 (\css_fits css_slopes ->
                      map2 (\css_fit css_slope ->
                              loess.interpolate_css s_m_fun css_fit css_slope pad_css_len jump_s
                           ) css_fits css_slopes
                   ) css_fits_l css_slopes_l |> opaque

          -- rebuild time series of size (N + 2 * n_p) out of smoothed cycle subseries
          let C_l = map (\css_results ->
                           tab (\i -> css_results[i % n_p, i / n_p]) C_len
                        ) css_results_l |> opaque

          -- Step 3: Low-pass filtering of collection of all the cycle-subseries
          --- apply 3 moving averages
          let ma3_l = moving_averages_l C_l n_p :> [m][n]t

          --- then apply LOESS
          let (l_results_l, l_slopes_l) = loess.loess_l (replicate m (iota n))
                                                        ma3_l
                                                        d_l
                                                        q_l
                                                        (t_m_fun >-> (+1))
                                                        weights_l
                                                        l_l_idx_l
                                                        l_lambdas_l
                                                        (replicate m n)
                                                        jump_l
                                                        jump_threshold
                                                        q_threshold

          -- interpolate, if needed
          let L_l =
            if jump_l > 1 then
              map2 (\l_results l_slopes ->
                      -- [n]
                      loess.interpolate l_m_fun l_results l_slopes n jump_l
                   ) l_results_l l_slopes_l
            else
              l_results_l :> [m][n]t

          -- Step 4: Detrend smoothed cycle-subseries
          --- extract the slice from the L array of size n. Can be fused with above
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
                    map2 (\v s -> (T.f32 v) - s) y seasonal
                 ) Y seasonal_l

          -- Step 6: Trend Smoothing
          let (D_pad_l, w_pad_l) =
            --- Gather non-nan values
            ----- can be fused with above?
            map3 (\D nn_idx weights ->
                    -- [n]
                    let D_pad = pad_gather_floats D nn_idx
                    let w_pad = pad_gather_floats weights nn_idx
                    in (D_pad, w_pad)
                 ) D_l nn_idx_l weights_l |> unzip

          -- apply LOESS
          let (t_results_l, t_slopes_l) = loess.loess_l nn_idx_l
                                                        D_pad_l
                                                        d_t
                                                        q_t
                                                        (t_m_fun >-> (+1))
                                                        w_pad_l
                                                        t_l_idx_l
                                                        t_lambdas_l
                                                        n_nn_l
                                                        jump_t
                                                        jump_threshold
                                                        q_threshold
          --- interpolate
          let trend_l =
            -- [n]
            if jump_t > 1 then
              map2 (\t_results t_slopes ->
                      loess.interpolate t_m_fun t_results t_slopes n jump_t
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
        if i_outer < n_outer - 1
        then
          --- calculate remainder estimate
          let R_abs_l =
            map3 (\y seasonal trend ->
                    -- [n]
                    let R = map3 (\v s t -> (T.f32 v) - s - t) y seasonal trend
                    in
                    map (\r -> if T.isnan r then r else T.abs r) R
                 ) Y seasonal_l trend_l
          let R_pad_l =
            map2 (\R_abs nn_idx ->
                    -- [n]
                    pad_gather R_abs nn_idx T.inf
                 ) R_abs_l nn_idx_l |> opaque

          let med_l = median.median_l R_pad_l n_nn_l
          -- find boundaries
          let (h_l, h9_l, h1_l)=
            map (\med ->
                   -- median
                   let h = 6 * med
                   -- boundaries
                   let h9 = 0.999 * h
                   let h1 = 0.001 * h
                   in (h, h9, h1)
                ) med_l |> unzip3 |> opaque

          -- calculate new weights
          let w_l =
            -- [n]
            map4 (\R_abs h h9 h1 ->
                 map (\r_abs ->
                        let zero_val = 10 ** (-6)
                        let bicube = (1 - (r_abs / h)**2)**2
                        in
                        if T.isnan r_abs || r_abs <= h1 then 1 else
                        if r_abs >= h9 || r_abs <= zero_val then zero_val
                        else bicube
                     ) R_abs
                 ) R_abs_l h_l h9_l h1_l |> opaque
          in w_l
        else
          weights_l
      in (seasonal_l, trend_l, weights_l)
      ------------------------------------
      -- Outer loop end                 --
      ------------------------------------
  let tof32 = f32.f64 <-< T.to_f64
  in
  map (\s -> tof32 (T.maximum s - T.minimum s)) seasonal_l


  let stl_filt [m] [n] (Y: [m][n]f32)
                       (n_p: i64)
                       (q_s: i64)
                       (q_t: i64)
                       (q_l: i64)
                       (d_s: i64)
                       (d_t: i64)
                       (d_l: i64)
                       (jump_s: i64)
                       (jump_t: i64)
                       (jump_l: i64)
                       (n_inner: i64)
                       (n_outer: i64)
                       (jump_threshold: i64)
                       (q_threshold: i64) =
    let max_css_len = T.ceil ((T.i64 n) / (T.i64 n_p)) |> T.to_i64
    -- let all_nans_l = map (all (T.isnan)) Y
    -- detect if at least one of css in each time series is all NaNs
    let all_nans_l = map (\y ->
                            tab (\i ->
                                   tab (\j ->
                                          let idx = j * n_p + i
                                          in idx >= n || f32.isnan y[idx]
                                       ) max_css_len |> (all id)
                                ) n_p |> any id
                         ) Y
    -- filter out all such time series
    let (Y_filt, _, idxs) =
      filter (\(_, flag, _) -> !flag) (zip3 Y all_nans_l (iota m)) |> unzip3

    -- apply STL to each time series that passed the filtering
    let seasonal_magn_filt = stl Y_filt
                                 n_p
                                 q_s
                                 q_t
                                 q_l
                                 d_s
                                 d_t
                                 d_l
                                 jump_s
                                 jump_t
                                 jump_l
                                 n_inner
                                 n_outer
                                 jump_threshold
                                 q_threshold

    -- write the decomposed values into the bufferes of full batch size m
    in scatter (replicate m (f32.nan)) idxs seasonal_magn_filt
}


entry main [m] [n] (Y: [m][n]f32)
                   (n_p: i64)
                   (q_s: i64)
                   (q_t: i64)
                   (q_l: i64)
                   (d_s: i64)
                   (d_t: i64)
                   (d_l: i64)
                   (jump_s: i64)
                   (jump_t: i64)
                   (jump_l: i64)
                   (n_inner: i64)
                   (n_outer: i64)
                   (jump_threshold: i64)
                   (q_threshold: i64) =
  stl_batched.stl_filt Y
                       n_p
                       q_s
                       q_t
                       q_l
                       d_s
                       d_t
                       d_l
                       jump_s
                       jump_t
                       jump_l
                       n_inner
                       n_outer
                       jump_threshold
                       q_threshold

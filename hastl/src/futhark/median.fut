import "utils"

module median_batched = {

module T = f64
type t = T.t

local let imap  as f = map f as
local let imap2 as bs f = map2 f as bs
local let ones [q] 't (_xs: [q]t) = replicate q 1i32

-- meds: hopefully a decent estimate of the median values for each partition
-- ks:   the k-th smallest element to be searched for each partition
-- shp, II1, A:  the rep of the iregular array: shape, II1-helper (plus 1) and flat data
local let rankSearchBatch [m][n] (meds: [m]t) (ks: [m]i32)
                                 (shp: [m]i32) (II1: *[n]i32) (A: *[n]t) : [m]t =
  let II1_bak = replicate n 0i32
  let A_bak = replicate n (T.i64 0)
  let res = replicate m (T.i64 0)
  let q = 0i64

  let (_, _shp, _,_,_,_,_, res) =
    loop (ks : [m]i32, shp : [m]i32, II1, II1_bak, A, A_bak, q, res)
      while (length A > 0) do
        -- compute helpers based on shape
        let shp_sc = scan (+) 0 shp

        -- F(let pivot = last A)
        let pivots =
            imap2 shp_sc (indices shp_sc)
              (\ off i -> if q == 0i64
                          then meds[i]
                          else if off == 0
                               then T.i64 0
                               else A[off - 1]
              ) |> opaque

        -- compute lt_len and eq_len by means of histograms:
        let h_inds =
            imap2 II1 A
                  (\ sgmindp1 a ->
                    let sgmind = sgmindp1 - 1
                    let pivot  = pivots[sgmind]
                    let h_ind  = sgmind << 1
                    in  i64.i32 <|
                          if a < pivot then h_ind
                          else if pivot == a then h_ind + 1
                          else -1i32
                  )
        let h_vals = ones A
        let lens = reduce_by_index (replicate (2*m) 0i32) (+) 0i32 h_inds h_vals

        let (shp', kinds, ks') =
          imap2 ks (indices ks)
            (\ k i ->
                if k < 0 then (0, 3i8, -1) -- already processed
                else let lt_len = lens[i << 1] in
                     if k < lt_len then (lt_len, 0i8, k)
                     else let eq_len = lens[ (i << 1) + 1]
                          let lteq_len = lt_len + eq_len in
                          if k < lteq_len then (0, 1i8, -1)
                          else (shp[i] - lteq_len, 2i8, k - lteq_len)
            )
          |> unzip3

        -- write the subarrays that have finished
        let (scat_inds, scat_vals) =
            imap2 (indices kinds) kinds
                  (\ i knd ->
                    if knd == 1i8
                    then (i, pivots[i])
                    else (-1, 0.0)
                  )
            |> unzip
        let res' = scatter res scat_inds scat_vals

        -- use a filter to extract elements
        let keepElem sgmindp1 a =
          let sgmind = sgmindp1 - 1
          let pivot = pivots[sgmind]
          let kind  =  kinds[sgmind]
          in (a < pivot && kind == 0) || (a > pivot && kind == 2)

        let conds = map2 keepElem II1 A |> opaque -- strange fusion with duplicating computation

        let tmp_inds = map i32.bool conds |> scan (+) 0i32
        let tot_len = i64.i32 (last tmp_inds)
        let scat_inds = imap2 conds tmp_inds (\c ind -> if c then i64.i32 (ind - 1) else -1i64)
        let A'   = scatter A_bak scat_inds A
        let II1' = scatter II1_bak scat_inds II1
        let II1''= II1'[:tot_len]
        let A''  = A'[:tot_len]

        in  (ks', shp', II1'', II1, A'', A, q+1, res')
  in res

let median_l [m][n] (arr_l: [m][n]t) (n_nn_l: [m]i64) =
  -- find the initial estimate for the medians
  let mins = map (reduce_comm T.min T.highest) arr_l
  let maxs = map (reduce_comm T.max T.lowest ) arr_l
  let inits = map2 (+) mins maxs |> map (/ 2) |> opaque

  let ks = map (/2) n_nn_l
  let N  = m * n
  let shp = replicate m (i32.i64 n)

  -- II1 should in principle be computed with (mkII1 shp)
  let II1 = (tabulate_2d m n (\i _j -> i32.i64 i + 1) |> flatten) :> *[N]i32
  let A = copy (flatten arr_l) :> *[N]t

  -- find the candidate for the median
  let median_r = rankSearchBatch inits (map (i32.i64) ks) shp II1 A

  -- mark elements (strict!) < median_r with 1s
  let lts = map2 (\ar med ->
                   map (\a -> if a < med then 1i64 else 0) ar
                 ) arr_l median_r

  -- find the closest rightmost point < median_r and the number of points < median_r
  let (cs, _, nlts) = map2 (\ar lt ->
                        reduce_comm (\(v1, i1, f1) (v2, i2, f2) ->
                                       let (v3, i3) =
                                         if f1 == 0 && f2 > 0 then (v2, i2) else
                                         if f1 > 0 && f2 == 0 then (v1, i1) else
                                           if v1 > v2 then (v1, i1) else
                                           if v2 > v1 then (v2, i2) else
                                         if i1 < i2 then (v1, i1) else (v2, i2)
                                       in (v3, i3, f1 + f2)
                               ) (-T.inf, -1i64, 0i64) (zip3 ar (iota n) lt)
                           ) arr_l lts |> unzip3
  in map5 (\c nlt med k n_nn ->
             if k == 0 || n_nn % 2 == 1 || nlt < k
             then med -- keep the right median candidate
             else c + (med - c) / 2 -- find the average between the left and right candidates
          ) cs nlts median_r ks n_nn_l
}

let main [m][n] (Y: [m][n]f64) =
  let filterPadNans = filterPadWithKeys (\i -> !(f64.isnan i)) 0
  let (_, nn_idx_l, n_nn_l) = map filterPadNans Y |> unzip3
  let Y_pad_l = map2 (\y nn_idx -> pad_gather y nn_idx f64.inf) Y nn_idx_l
  in median_batched.median_l Y_pad_l n_nn_l

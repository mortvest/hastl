let nonans (arr: [][]f32): bool =
  !(any (any (f32.isnan)) arr)

-- compact wrapper for the tabulate function
let tab 't (fun: (i64 -> t)) (n: i64): [n]t =
  tabulate n fun

-- swap two values in an array (arr[i] <-> arr[j])
let swap 't (arr: *[]t) (i: i64) (j: i64): *[]t =
  #[unsafe]
  let v = copy arr[i]
  let arr[i] = copy arr[j]
  let arr[j] = v
  in arr

let check_pos (x: i64): i64 =
  assert (x > 0) x

let check_odd (x: i64): i64 =
  assert (x % 2 == 1) x

let check_deg (x: i64): i64 =
  assert (x >= 0 && x <= 2) x

let check_win =
  check_odd >-> check_pos

let check_nonneg (x: i64): i64 =
  assert (x >= 0) x

-- gather for the padded (with -1) indexes
let pad_gather [n] 'a (vs: []a) (idxs: [n]i64) (zero: a): [n]a =
  map (\i -> if i >= 0 then vs[i] else zero) idxs


let pad_idx (n: i64) (i: i64) (n_p: i64) (i_max: i64): [n]i64 =
  tab (\j -> let ind = i + (j * n_p)
               in
               if ind < i_max then ind else -1
      ) n

-- Filter the array with a predicate, then pad to original size with dummy
-- returns:
--- rs: array of matching values, padded with 0 of that type
--- ks: array of indexes of matching values, padded with -1
--- n : number of matching values
let filterPadWithKeys [n] 't
           (p : (t -> bool))
           (dummy : t)
           (arr : [n]t) : ([n]t, [n]i64, i64) =
  let tfs = map (\a -> if p a then 1i64 else 0i64) arr
  let isT = scan (+) 0i64 tfs
  let i   = last isT
  let inds= map2 (\a iT -> if p a then iT - 1 else -1i64) arr isT
  let rs  = scatter (replicate n dummy) inds arr
  let ks  = scatter (replicate n (-1i64)) inds (iota n)
  in (rs, ks, i)

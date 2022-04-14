func @std_for_yield(%arg0 : index, %arg1 : index, %arg2 : index) {
  %s0 = arith.constant 0.0 : f32
  %result = scf.for %i0 = %arg0 to %arg1 step %arg2 iter_args(%si = %s0) -> (f32) {
    %sn = arith.addf %si, %si : f32
    scf.yield %sn : f32
  }
  return
}


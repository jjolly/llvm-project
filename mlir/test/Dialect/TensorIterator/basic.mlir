#CSR = #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ]}>

func @iterate(%tensor: tensor<?x?xf32, #CSR>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %sum_0 = arith.constant 0.0 : f32
  tensor_iterator.iterate %outer in rank %c0 of %tensor
    : tensor<?x?xf32, #CSR> {
    tensor_iterator.iterate %inner in rank %c1 of %tensor
      : tensor<?x?xf32, #CSR> {
      tensor_iterator.yield
    }
    tensor_iterator.yield
  }
  return
}

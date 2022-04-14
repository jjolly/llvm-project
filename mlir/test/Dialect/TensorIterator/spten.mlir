// RUN: mlir-opt %s -sparsification | FileCheck %s

#DV = #sparse_tensor.encoding<{ dimLevelType = [ "dense"      ] }>
#SV = #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ] }>

#trait1 = {
  indexing_maps = [
    affine_map<(i) -> (i)>,  // a
    affine_map<(i) -> (i)>,  // b
    affine_map<(i) -> (i)>   // x (out)
  ],
  iterator_types = ["parallel"],
  doc = "x(i) = a(i) OP b(i)"
}

func @add_d(%arga: tensor<32xf32, #SV>, %argb: tensor<32xf32, #DV>, %argx: tensor<32xf32>) -> tensor<32xf32> {
  %0 = linalg.generic #trait1
     ins(%arga, %argb: tensor<32xf32, #SV>, tensor<32xf32, #DV>)
    outs(%argx: tensor<32xf32>) {
      ^bb(%a: f32, %b: f32, %x: f32):
        %0 = arith.addf %a, %b : f32
        linalg.yield %0 : f32
  } -> tensor<32xf32>
  return %0 : tensor<32xf32>
}

#DDM = #sparse_tensor.encoding<{ dimLevelType = [ "dense",      "dense"      ] }>
#SOM = #sparse_tensor.encoding<{ dimLevelType = [ "singleton",  "compressed" ] }>
#COO = #sparse_tensor.encoding<{ dimLevelType = [ "compressed", "singleton"  ] }>

#trait2 = {
  indexing_maps = [
    affine_map<(i,j) -> (i,j)>,  // a
    affine_map<(i,j) -> (i,j)>,  // b
    affine_map<(i,j) -> (i,j)>   // x (out)
  ],
  iterator_types = ["parallel", "parallel"],
  doc = "x(i,j) = a(i,j) OP b(i,j)"
}

func @add_dd(%arga: tensor<8x32xf32, #COO>, %argb: tensor<8x32xf32, #DDM>, %argx: tensor<8x32xf32>) -> tensor<8x32xf32> {
  %0 = linalg.generic #trait2
     ins(%arga, %argb: tensor<8x32xf32, #COO>, tensor<8x32xf32, #DDM>)
    outs(%argx: tensor<8x32xf32>) {
      ^bb(%a: f32, %b: f32, %x: f32):
        %0 = arith.addf %a, %b : f32
        linalg.yield %0 : f32
  } -> tensor<8x32xf32>
  return %0 : tensor<8x32xf32>
}

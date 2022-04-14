// RUN: mlir-opt %s -linalg-generalize-named-ops -sparsification | FileCheck %s

#DENSE = #sparse_tensor.encoding<{ dimLevelType = [ "dense",      "dense"      ] }>
#CSR   = #sparse_tensor.encoding<{ dimLevelType = [ "dense",      "compressed" ] }>
#TARGA = #CSR
#TARGB = #DENSE

func @mul_dd(%arga: tensor<?x?xf32, #TARGA>, %argb: tensor<?x?xf32, #TARGB>, %argc: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = linalg.matmul
    ins(%arga, %argb: tensor<?x?xf32, #TARGA>, tensor<?x?xf32, #TARGB>)
    outs(%argc: tensor<?x?xf32>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}


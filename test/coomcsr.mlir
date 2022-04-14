// RUN: mlir-opt %s -test-linalg-codegen-strategy="anchor-func=matmul anchor-op=linalg.matmul tile-sizes=16,32,64 fuse pad vectorize" | FileCheck %s

#COO = #sparse_tensor.encoding<{ dimLevelType = [ "singleton",  "compressed" ] }>
#CSR = #sparse_tensor.encoding<{ dimLevelType = [ "dense",      "compressed" ] }>

func @mul_dd(%arga: tensor<10x20xf32, #COO>, %argb: tensor<20x30xf32, #CSR>, %argc: tensor<10x30xf32>) -> tensor<10x30xf32> {
  %0 = linalg.matmul
    ins(%arga, %argb: tensor<10x20xf32, #COO>, tensor<20x30xf32, #CSR>)
    outs(%argc: tensor<10x30xf32>) -> tensor<10x30xf32>
  return %0 : tensor<10x30xf32>
}


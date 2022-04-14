add_mlir_dialect_library(MLIRTensorIterator
  TensorIteratorDialect.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/TensorIterator

  DEPENDS
  #  MLIRTensorIteratorAttrDefsIncGen
  MLIRTensorIteratorOpsIncGen

  LINK_LIBS PUBLIC
  MLIRDialect
  MLIRIR
  MLIRSupport
  )


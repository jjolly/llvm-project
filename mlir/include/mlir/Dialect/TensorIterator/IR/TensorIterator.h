#ifndef MLIR_DIALECT_TENSORITERATOR_IR_TENSORITERATOR_H_
#define MLIR_DIALECT_TENSORITERATOR_IR_TENSORITERATOR_H_

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "mlir/Dialect/TensorIterator/IR/TensorIteratorOps.h.inc"

#include "mlir/Dialect/TensorIterator/IR/TensorIteratorOpsDialect.h.inc"

namespace mlir {
  namespace tensor_iterator {
  } // namespace tensor_iterator
} // namespace mlir

#endif // MLIR_DIALECT_TENSORITERATOR_IR_TENSORITERATOR_H_

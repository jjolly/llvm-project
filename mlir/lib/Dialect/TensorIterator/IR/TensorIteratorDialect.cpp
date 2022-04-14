#include "mlir/Dialect/TensorIterator/IR/TensorIterator.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"


using namespace mlir;
using namespace mlir::tensor_iterator;

#include "mlir/Dialect/TensorIterator/IR/TensorIteratorOpsDialect.cpp.inc"

void TensorIteratorDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/TensorIterator/IR/TensorIteratorOps.cpp.inc"
    >();
}

static ParseResult parseIterateOp(OpAsmParser &parser, OperationState &result) {
  auto &builder = parser.getBuilder();
  (void)builder;
  SmallVector<OpAsmParser::OperandType, 4> inductionOperands;
  SmallVector<OpAsmParser::OperandType, 4> ranks;
  SmallVector<OpAsmParser::OperandType, 4> tensors;
  SmallVector<Type, 1> inductionTypes;
  SmallVector<Type, 1> tensorTypes;
  OpAsmParser::OperandType rank, tensor;

  if (parser.parseOperandList(inductionOperands) ||
      parser.parseKeyword("in") ||
      parser.parseKeyword("rank"))
    return ::mlir::failure();

  for(;;) {
    // Rank has already been processed. Get on with the index and operand
    if (parser.parseOperand(rank) ||
        parser.parseKeyword("in") ||
        parser.parseOperand(tensor))
      return ::mlir::failure();

    ranks.push_back(rank);
    tensors.push_back(tensor);

    if (!succeeded(parser.parseOptionalComma()))
      break;

    if (parser.parseKeyword("rank"))
      return ::mlir::failure();
  }

  if (parser.parseColonTypeList(inductionTypes) ||
      parser.parseArrowTypeList(tensorTypes))
    return ::mlir::failure();

  for (auto operandType : llvm::zip(inductionOperands, inductionTypes))
    if (parser.resolveOperand(std::get<0>(operandType), std::get<1>(operandType), result.operands))
      return failure();

  for (auto operandType : llvm::zip(ranks, llvm::zip(tensors, tensorTypes))) {
    if (parser.resolveOperand(std::get<0>(operandType), builder.getIndexType(), result.operands))
      return failure();

    if (parser.resolveOperand(std::get<0>(std::get<1>(operandType)), std::get<1>(std::get<1>(operandType)), result.operands))
      return failure();
  }

  return parser.parseOptionalAttrDict(result.attributes);
}

void print(OpAsmPrinter &p, IterateOp &o) {
}

LogicalResult verify(IterateOp &o) {
  return success();
}

LogicalResult verify(IteratorYieldOp &y) {
  return success();
}

#define GET_OP_CLASSES
#include "mlir/Dialect/TensorIterator/IR/TensorIteratorOps.cpp.inc" 

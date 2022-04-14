#ifndef TENSORITERATOR_OPS
#define TENSORITERATOR_OPS

include "mlir/Dialect/TensorIterator/IR/TensorIteratorDialect.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/LoopLikeInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class TensorIteratorOp<string mnemonic, list<Trait> traits = []>
  : Op<TensorIteratorDialect, mnemonic, traits> {
  let printer = [{ return ::print(p, *this); }];
  let verifier = [{ return ::verify(*this); }];
  let parser = [{ return ::parse$cppClass(parser, result); }];
}

def IterateOp : TensorIteratorOp<"iterate", 
    [SingleBlockImplicitTerminator<"IteratorYieldOp">]> {
  let description = [{
    The tensor iterator iterate operator will take the form:
    ```mlir
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %sum_0 = arith.constant 0.0 : f32
    %result = tensor_iterator.iterate %outer in rank %c0 of %tensor
      : tensor<?x?xf32, #CSR> -> f32 {
      %sum_row = tensor_iterator.iterate $inner in rank %c1 of %tensor
        : tensor<?x?xf32, #CSR> -> f32 {
      }
    }
    ```
  }];

  let arguments = (ins Variadic<AnyType>:$varsRanksTensors);
  let results = (outs Variadic<AnyType>:$results);
  // let regions = (region SizedRegion<1>:$region);

  let extraClassDeclaration = [{
  }];
}

def IteratorYieldOp : TensorIteratorOp<"yield", [NoSideEffect, ReturnLike, Terminator]> {
  let summary = "loop yield and termination operation";
  let description = [{
    "tensor_iterator.yield" yields an SSA value from the SCF dialect op region and
    terminates the regions. The semantics of how the values are yielded is
    defined by the parent operation.
    If "tensor_iterator.yield" has any operands, the operands must match the parent
    operation's results.
    If the parent operation defines no values, then the "tensor_iterator.yield" may be
    left out in the custom syntax and the builders will insert one implicitly.
    Otherwise, it has to be present in the syntax to indicate which values are
    yielded.
  }];

  let arguments = (ins Variadic<AnyType>:$results);
  let builders = [
    OpBuilder<(ins), [{ build($_builder, $_state, llvm::None); }]>
  ];

  let assemblyFormat =
      [{ attr-dict ($results^ `:` type($results))? }];
  // let hasVerifier = 1;
}

#endif // SPARSEITERATOR_OPS

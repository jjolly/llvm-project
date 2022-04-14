// Definition of the 'TensorIterator_Dialect' in the ODS framework
// This is for the base dialect

#ifndef TENSORITERATOR_BASE
#define TENSORITERATOR_BASE

include "mlir/IR/OpBase.td"

def TensorIteratorDialect : Dialect {
  let name = "tensor_iterator";
  let cppNamespace = "::mlir::tensor_iterator";
  let description = [{
    This is the `TensorIterator` dialect. It supports individual tensor iterators by abstracting the iterator away and allowing for the dynamic definition of the iterator at a later stage.
    
    Within sparse datatypes, the iterator for one dimension of a sparse tensor may have a definition that does not fit traditional definitions. The two built-in definitions are:
    * Dense: This is a fairly typical affine loop through the dimension of a tensor
    * Compressed: This will iterate through a vector of indices that identify the range of nonzero values and the offset index for that nonzero. 
  }];
}

#endif // TENSORITERATOR_BASE

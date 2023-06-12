// #include "mlir/Dialect/Tosa/IR/TosaOps.h"       // from @llvm-project
// #include "mlir/Dialect/Tosa/Utils/QuantUtils.h" // from @llvm-project
// #include "torch-mlir/Conversion/TorchToTosa/TosaLegalizeCommon.h"
// #include "torch-mlir/Conversion/TorchToTosa/TosaLegalizeUtils.h"

// namespace mlir::tf {
    
// // Template specialization for float
// template <>
// std::optional<Value> getConstTensor<float>(PatternRewriter &rewriter,
//                                            Operation *op, ArrayRef<float> vec,
//                                            ArrayRef<int64_t> shape) {
//   uint64_t num_total_elements = 1;
//   for (int64_t a : shape) {
//     num_total_elements *= a;
//   }

//   if (vec.size() != num_total_elements) {
//     op->emitOpError("getConstTensor(): number of elements mismatch.");
//     return std::nullopt;
//   }

//   auto const_type = RankedTensorType::get(shape, rewriter.getF32Type());
//   auto const_attr = DenseElementsAttr::get(const_type, vec);

//   auto const_op =
//       rewriter.create<TF::ConstOp>(op->getLoc(), const_type, const_attr);
//   return const_op.getResult();
// }

// } // namespace mlir::tf
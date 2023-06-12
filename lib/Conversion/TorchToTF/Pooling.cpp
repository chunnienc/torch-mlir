// #include "torch-mlir/Conversion/TorchToTF/TFLegalizeUtils.h"
#include "torch-mlir/Conversion/TorchToTF/TorchToTF.h"
#include "torch-mlir/Conversion/Utils/Utils.h"

#include "../PassDetail.h"
#include "PopulatePatterns.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Traits.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/DialectConversion.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/IR/TorchTypes.h"
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"
#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionDialect.h"
#include "torch-mlir/Dialect/TorchConversion/Transforms/BackendTypeConversion.h"

namespace mlir {
namespace torch {
namespace torch_to_tf {

namespace {

using int32_t = std::int32_t;
using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;

// Apply the transposeDims vector on input to generate a transposed form.
// Value transposeTensor(AtenOpT op, ConversionPatternRewriter &rewriter,
//                       Value input, ArrayRef<int32_t> transposeDims) {
//   auto inputTy = input.getType().template cast<RankedTensorType>();
//   auto inputElemTy = inputTy.getElementType();
//   auto inputShape = makeShapeTorchCompatible(inputTy.getShape());
//   auto inputRank = inputTy.getRank();

//   std::optional<Value> transposeDimsConst =
//       getConstTensor<int32_t>(rewriter, op,
//                               /*vec=*/transposeDims,
//                               /*shape=*/{static_cast<int32_t>(inputRank)});

//   SmallVector<int64_t> transposedInputShape;
//   for (auto &dim : transposeDims) {
//     transposedInputShape.push_back(inputShape[dim]);
//   }
//   auto transposedInputType = RankedTensorType::get(
//       makeShapeLLVMCompatible(transposedInputShape), inputElemTy);
//   return rewriter
//       .create<TF::TransposeOp>(op->getLoc(), transposedInputType, input,
//                                transposeDimsConst.value())
//       .getResult();
// }

// Value transposePoolingInputToHwc(AtenOpT op,
//                                  ConversionPatternRewriter &rewriter,
//                                  Value input) {
//   auto inputRank = input.getType().template
//   cast<RankedTensorType>().getRank();

//   SmallVector<int32_t> nchwToNhwc4DTransposeDims({0, 2, 3, 1});
//   SmallVector<int32_t> chwToHwc3DTransposeDims({1, 2, 0});

//   return transposeTensor(op, rewriter, input,
//                          inputRank == 3 ? chwToHwc3DTransposeDims
//                                         : nchwToNhwc4DTransposeDims);
// }

} // namespace

void populatePoolingOpPatternsAndLegality(TypeConverter &typeConverter,
                                          RewritePatternSet &patterns,
                                          ConversionTarget &target,
                                          const TorchToTFOptions &options) {}

} // namespace torch_to_tf
} // namespace torch
} // namespace mlir
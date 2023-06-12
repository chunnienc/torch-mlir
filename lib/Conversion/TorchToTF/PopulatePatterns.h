
// #ifndef TORCHMLIR_LIB_CONVERSION_TORCHTOTF_POPULATEPATTERNS_H
// #define TORCHMLIR_LIB_CONVERSION_TORCHTOTF_POPULATEPATTERNS_H

// #include "mlir/Transforms/DialectConversion.h"
// #include "tensorflow/compiler/mlir/tensorflow/ir/tf_dialect.h"
// #include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
// #include "torch-mlir/Conversion/TorchToTF/TFLegalizeUtils.h"
// #include "torch-mlir/Dialect/Torch/Utils/Utils.h"

// namespace mlir {
// namespace torch {
// namespace torch_to_tf {

// using namespace mlir::torch::Torch;

// struct TorchToTFOptions {};

// template <typename AtenOpT>
// class ConvertAtenOp : public OpConversionPattern<AtenOpT> {
// public:
//   using OpAdaptor = typename AtenOpT::Adaptor;
//   ConvertAtenOp(TypeConverter &typeConverter, MLIRContext *context,
//                 const TorchToTFOptions &options = {})
//       : OpConversionPattern<AtenOpT>(typeConverter, context) {
//     this->options = options;
//   }
//   LogicalResult
//   matchAndRewrite(AtenOpT op, OpAdaptor adaptor,
//                   ConversionPatternRewriter &rewriter) const override {
//     return rewriter.notifyMatchFailure(op, "haven't been implemented");
//   }
//   const TorchToTFOptions &getOptions() const { return options; }

//   // Apply the transposeDims vector on input to generate a transposed form.
//   Value transposeTensor(AtenOpT op, ConversionPatternRewriter &rewriter,
//                         Value input, ArrayRef<int32_t> transposeDims) {
//     auto inputTy = input.getType().template cast<RankedTensorType>();
//     auto inputElemTy = inputTy.getElementType();
//     auto inputShape = makeShapeTorchCompatible(inputTy.getShape());
//     auto inputRank = inputTy.getRank();

//     std::optional<Value> transposeDimsConst =
//         getConstTensor<int32_t>(rewriter, op,
//                                 /*vec=*/transposeDims,
//                                 /*shape=*/{static_cast<int32_t>(inputRank)});

//     SmallVector<int64_t> transposedInputShape;
//     for (auto &dim : transposeDims) {
//       transposedInputShape.push_back(inputShape[dim]);
//     }
//     auto transposedInputType = RankedTensorType::get(
//         makeShapeLLVMCompatible(transposedInputShape), inputElemTy);
//     return rewriter
//         .create<TF::TransposeOp>(op->getLoc(), transposedInputType, input,
//                                  transposeDimsConst.value())
//         .getResult();
//   }

// private:
//   TorchToTFOptions options;
// };

// } // namespace torch_to_tf
// } // namespace torch
// } // namespace mlir

// #endif // TORCHMLIR_LIB_CONVERSION_TORCHTOSTABLEHLO_POPULATEPATTERNS_H

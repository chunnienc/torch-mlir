//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "torch-mlir/Conversion/TorchToTF/TorchToTF.h"
#include <cstdio>

#include "../PassDetail.h"

#include "PopulatePatterns.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Traits.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_dialect.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/core/framework/kernel_shape_util.h"
#include "torch-mlir/Conversion/TorchToTF/TFLegalizeUtils.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"
#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionDialect.h"
#include "torch-mlir/Dialect/TorchConversion/Transforms/BackendTypeConversion.h"

namespace mlir {
namespace torch {
namespace torch_to_tf {

namespace {

using namespace mlir;
using namespace mlir::tf;
using namespace mlir::torch;
using namespace mlir::torch::Torch;

template <typename AtenOpT, typename TFOpT>
class ConvertAtenPoolingBaseOp : public OpConversionPattern<AtenOpT> {
public:
  using OpConversionPattern<AtenOpT>::OpConversionPattern;
  using OpAdaptor = typename AtenOpT::Adaptor;

  static Type getOutputTypeForNonAdaptivePoolingOp(
      RankedTensorType inputTy, SmallVectorImpl<int64_t> &kernelSize,
      SmallVectorImpl<int64_t> &strideArray, SmallVectorImpl<int64_t> &padArray,
      SmallVectorImpl<int64_t> &dilationArray, bool ceilMode = false) {
    auto inputShape = makeShapeTorchCompatible(inputTy.getShape());
    auto inputRank = inputTy.getRank();
    auto inputElemTy = inputTy.getElementType();

    int64_t outputHDim =
        getOutputDim(inputShape[inputRank - 2], kernelSize[0], strideArray[0],
                     padArray[0], padArray[0], dilationArray[0], ceilMode);
    int64_t outputWDim =
        getOutputDim(inputShape[inputRank - 1], kernelSize[1], strideArray[1],
                     padArray[1], padArray[1], dilationArray[1], ceilMode);
    // padArray[0] = (outputHDim - 1) * strideArray[0] +
    //               dilationArray[0] * kernelSize[0] - dilationArray[0] + 1 -
    //               padArray[0] * 2 - inputShape[inputRank - 2];
    // padArray[1] = (outputWDim - 1) * strideArray[1] +
    //               dilationArray[0] * kernelSize[1] - dilationArray[0] + 1 -
    //               padArray[1] * 2 - inputShape[inputRank - 1];
    SmallVector<int64_t> outputShape;
    if (inputRank > 3) {
      outputShape.push_back(inputShape[0]);
    }
    outputShape.push_back(outputHDim);
    outputShape.push_back(outputWDim);
    outputShape.push_back(inputShape[inputRank - 3]);
    return RankedTensorType::get(makeShapeLLVMCompatible(outputShape),
                                 inputElemTy);
  }

  template <typename T> class HasDilationAttr {
  private:
    typedef char YesType[1];
    typedef char NoType[2];
    template <typename C> static YesType &test(decltype(&C::getDilation));
    template <typename C> static NoType &test(...);

  public:
    enum { value = sizeof(test<T>(0)) == sizeof(YesType) };
  };

  LogicalResult processInputs(AtenOpT op, OpAdaptor adaptor,
                              ConversionPatternRewriter &rewriter, Value &input,
                              ArrayAttr &kernel, ArrayAttr &stride,
                              Type &outputTy) const {
    SmallVector<int64_t, 2> dilationArray;
    if constexpr (HasDilationAttr<AtenOpT>::value) {
      if (!matchPattern(op.getDilation(),
                        m_TorchListOfConstantInts(dilationArray))) {
        return rewriter.notifyMatchFailure(
            op, "Non-const dilation for pooling op unsupported.");
      }
      // TF pooling only supports unit dilation.
      if (dilationArray[0] > 1 || dilationArray[1] > 1) {
        return rewriter.notifyMatchFailure(
            op, "Cannot process non-unit pooling dilation.");
      }
    } else {
      dilationArray = {1, 1};
    }

    input = adaptor.getSelf();

    if (failed(getOutputTypeAndPoolingParameters(
            op, rewriter, input, dilationArray, outputTy, kernel, stride))) {
      return rewriter.notifyMatchFailure(
          op, "invalid pooling parameters or input type");
    }

    // Transpose to xHWC
    input = transposePoolingInputToHwc(op, rewriter, input);

    return success();
  }

  static LogicalResult getOutputTypeAndPoolingParameters(
      AtenOpT op, ConversionPatternRewriter &rewriter, Value &inputXchw,
      SmallVectorImpl<int64_t> &dilationArray, Type &outputTy,
      ArrayAttr &kernel, ArrayAttr &stride) {

    RankedTensorType inputTy = inputXchw.getType().cast<RankedTensorType>();
    if (!inputTy) {
      return rewriter.notifyMatchFailure(
          op, "Pooling op requires ranked tensor input");
    }
    auto inputElemTy = inputTy.getElementType();
    auto inputShape = makeShapeTorchCompatible(inputTy.getShape());

    auto inputRank = inputTy.getRank();
    // Rank sanity check.
    if (inputTy.getRank() != 4 && inputRank != 3) {
      return rewriter.notifyMatchFailure(
          op, "NCHW->NHWC transpose requires 3D or 4D tensor");
    }

    SmallVector<int64_t, 2> kernelSizeInts, strideInts, paddingInts;
    if (!matchPattern(op.getKernelSize(),
                      m_TorchListOfConstantInts(kernelSizeInts))) {
      return rewriter.notifyMatchFailure(
          op, "Non-const kernel_size for pooling op unsupported");
    }

    if (!matchPattern(op.getStride(), m_TorchListOfConstantInts(strideInts))) {
      return rewriter.notifyMatchFailure(
          op, "Non-const stride for pooling op unsupported");
    }

    // If `stride` is not specified by the user, it is assigned the value of
    // empty list during import. For such a case, the stride value is the kernel
    // size. See:
    // https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html#torch.nn.MaxPool2d
    if (strideInts.empty()) {
      strideInts.assign(kernelSizeInts);
    }

    if (!matchPattern(op.getPadding(),
                      m_TorchListOfConstantInts(paddingInts))) {
      return rewriter.notifyMatchFailure(
          op, "Non-const padding factor for pooling op unsupported");
    }

    kernel =
        rewriter.getI64ArrayAttr({1, kernelSizeInts[0], kernelSizeInts[1], 1});
    stride = rewriter.getI64ArrayAttr({1, strideInts[0], strideInts[1], 1});

    bool ceilMode;
    if (!matchPattern(op.getCeilMode(), m_TorchConstantBool(&ceilMode))) {
      return rewriter.notifyMatchFailure(
          op, "only support constant bool ceil_mode for pooling op");
    }

    outputTy = getOutputTypeForNonAdaptivePoolingOp(inputTy, kernelSizeInts,
                                                    strideInts, paddingInts,
                                                    dilationArray, ceilMode);
    if (paddingInts[0] != 0 || paddingInts[1] != 0) {
      SmallVector<int32_t> paddings(inputRank * 2, 0);
      paddings[(inputRank - 2) * 2] = paddingInts[0];
      paddings[(inputRank - 2) * 2 + 1] = paddingInts[0];
      paddings[(inputRank - 1) * 2] = paddingInts[1];
      paddings[(inputRank - 1) * 2 + 1] = paddingInts[1];
      Value paddingConst = getConstTensor<int32_t>(
                               rewriter, op,
                               /*vec=*/paddings,
                               /*shape=*/{static_cast<int32_t>(inputRank), 2})
                               .value();

      // std::optional<Value> constantValuesTensor =
      //     getConstTensor<int32_t>(rewriter, op, {-1}, {1});
      SmallVector<int64_t> padOutputShape(inputShape.begin(), inputShape.end());
      padOutputShape[inputRank - 2] += paddingInts[0] * 2;
      padOutputShape[inputRank - 1] += paddingInts[1] * 2;
      auto padOutputType = RankedTensorType::get(
          makeShapeLLVMCompatible(padOutputShape), inputElemTy);

      inputXchw = rewriter
                      .create<TF::PadOp>(op->getLoc(), padOutputType, inputXchw,
                                         paddingConst)
                      .getResult();
    }
    return success();
  }

  static int64_t getOutputDim(int64_t inputDim, int64_t kernelDim,
                              int64_t stride, int64_t padBefore,
                              int64_t padAfter, int64_t dilation,
                              bool ceilMode = false) {
    if (inputDim == kUnknownSize) {
      return kUnknownSize;
    } else {
      int64_t dimSize =
          inputDim + padBefore + padAfter - dilation * (kernelDim - 1) - 1;
      if (ceilMode && (dimSize % stride != 0)) {
        return dimSize / stride + 2;
      }
      return dimSize / stride + 1;
    }
  }

  // Apply the transposeDims vector on input to generate a transposed form.
  Value transposeTensor(AtenOpT op, ConversionPatternRewriter &rewriter,
                        Value input, ArrayRef<int32_t> transposeDims) const {
    auto inputTy = input.getType().template cast<RankedTensorType>();
    auto inputElemTy = inputTy.getElementType();
    auto inputShape = makeShapeTorchCompatible(inputTy.getShape());
    auto inputRank = inputTy.getRank();

    std::optional<Value> transposeDimsConst =
        getConstTensor<int32_t>(rewriter, op,
                                /*vec=*/transposeDims,
                                /*shape=*/{static_cast<int32_t>(inputRank)});

    SmallVector<int64_t> transposedInputShape;
    for (auto &dim : transposeDims)
      transposedInputShape.push_back(inputShape[dim]);
    auto transposedInputType = RankedTensorType::get(
        makeShapeLLVMCompatible(transposedInputShape), inputElemTy);
    return rewriter
        .create<TF::TransposeOp>(op->getLoc(), transposedInputType, input,
                                 transposeDimsConst.value())
        .getResult();
  }

  Value transposePoolingInputToHwc(AtenOpT op,
                                   ConversionPatternRewriter &rewriter,
                                   Value input) const {
    auto inputRank =
        input.getType().template cast<RankedTensorType>().getRank();

    SmallVector<int32_t> nchwToNhwc4DTransposeDims({0, 2, 3, 1});
    SmallVector<int32_t> chwToHwc3DTransposeDims({1, 2, 0});

    return transposeTensor(op, rewriter, input,
                           inputRank == 3 ? chwToHwc3DTransposeDims
                                          : nchwToNhwc4DTransposeDims);
  }

  Value transposePoolingOutputToChw(AtenOpT op,
                                    ConversionPatternRewriter &rewriter,
                                    Value input) const {
    auto inputTy = input.getType().template cast<RankedTensorType>();
    auto inputRank = inputTy.getRank();

    SmallVector<int32_t> nhwcToNchw4DTransposeDims({0, 3, 1, 2});
    SmallVector<int32_t> hwcToChw3DTransposeDims({2, 0, 1});

    return transposeTensor(op, rewriter, input,
                           inputRank == 3 ? hwcToChw3DTransposeDims
                                          : nhwcToNchw4DTransposeDims);
  }

  virtual Value applyPooling(AtenOpT op, ConversionPatternRewriter &rewriter,
                             Value &input, Type &outputTy, ArrayAttr &kernel,
                             ArrayAttr &stride) const {
    return input;
  }

  LogicalResult
  matchAndRewrite(AtenOpT op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value input;
    ArrayAttr kernel, stride;
    Type outputTy;

    if (failed(processInputs(op, adaptor, rewriter, input, kernel, stride,
                             outputTy))) {
      return rewriter.notifyMatchFailure(
          op, "Failed to process inputs for pooling");
    }

    auto pooledOutput =
        applyPooling(op, rewriter, input, outputTy, kernel, stride);

    auto transposedOutput =
        transposePoolingOutputToChw(op, rewriter, pooledOutput);

    rewriter.replaceOpWithNewOp<tensor::CastOp>(
        op,
        OpConversionPattern<AtenOpT>::getTypeConverter()->convertType(
            op.getType()),
        transposedOutput);

    return success();
  }
};

class ConvertAtenMaxPool2dOp
    : public ConvertAtenPoolingBaseOp<AtenMaxPool2dOp, TF::MaxPoolOp> {
public:
  using ConvertAtenPoolingBaseOp<AtenMaxPool2dOp,
                                 TF::MaxPoolOp>::ConvertAtenPoolingBaseOp;

  Value applyPooling(AtenMaxPool2dOp op, ConversionPatternRewriter &rewriter,
                     Value &input, Type &outputTy, ArrayAttr &kernel,
                     ArrayAttr &stride) const override {
    return rewriter
        .create<TF::MaxPoolOp>(
            op->getLoc(), outputTy, input, kernel, stride,
            /*padding=*/rewriter.getStringAttr("VALID"),
            /*explicit_paddings=*/rewriter.getI64ArrayAttr({}),
            rewriter.getStringAttr("NHWC"))
        .getResult();
  }
};

class ConvertAtenAvgPool2dOp
    : public ConvertAtenPoolingBaseOp<AtenAvgPool2dOp, TF::AvgPoolOp> {
public:
  using ConvertAtenPoolingBaseOp<AtenAvgPool2dOp,
                                 TF::AvgPoolOp>::ConvertAtenPoolingBaseOp;

  Value applyPooling(AtenAvgPool2dOp op, ConversionPatternRewriter &rewriter,
                     Value &input, Type &outputTy, ArrayAttr &kernel,
                     ArrayAttr &stride) const override {
    return rewriter
        .create<TF::AvgPoolOp>(op->getLoc(), outputTy, input, kernel, stride,
                               /*padding=*/rewriter.getStringAttr("VALID"),
                               rewriter.getStringAttr("NHWC"))
        .getResult();
  }
};

class ConvertTorchToTF : public ConvertTorchToTFBase<ConvertTorchToTF> {
public:
  ConvertTorchToTF() = default;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<TF::TensorFlowDialect>();
    registry.insert<tensor::TensorDialect>();
    registry.insert<arith::ArithDialect>();
    TorchConversion::getBackendTypeConversionDependentDialects(registry);
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ConversionTarget target(*context);
    target.addLegalDialect<TF::TensorFlowDialect, Torch::TorchDialect,
                           tensor::TensorDialect, arith::ArithDialect>();

    TypeConverter typeConverter;
    typeConverter.addConversion([](Type type) { return type; });
    TorchConversion::setupBackendTypeConversion(target, typeConverter);

    RewritePatternSet patterns(context);

    target.addIllegalOp<AtenMaxPool2dOp>();
    patterns.add<ConvertAtenMaxPool2dOp>(typeConverter, context);
    target.addIllegalOp<AtenAvgPool2dOp>();
    patterns.add<ConvertAtenAvgPool2dOp>(typeConverter, context);

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

} // namespace

} // namespace torch_to_tf
} // namespace torch
} // namespace mlir

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
mlir::torch::createConvertTorchToTFPass() {
  return std::make_unique<mlir::torch::torch_to_tf::ConvertTorchToTF>();
}
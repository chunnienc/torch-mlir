//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "torch-mlir/Conversion/TorchToTF/TorchToTF.h"

#include "../PassDetail.h"
#include "../TorchToStablehlo/PopulatePatterns.h"
#include "LegalizeHloToTF.h"

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
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"
#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionDialect.h"
#include "torch-mlir/Dialect/TorchConversion/Transforms/BackendTypeConversion.h"

#include "stablehlo/dialect/ChloOps.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/xla/mlir_hlo/mhlo/IR/hlo_ops.h"
#include "tensorflow/compiler/xla/mlir_hlo/mhlo/transforms/passes.h"
#include "tensorflow/compiler/xla/mlir_hlo/mhlo/transforms/rewriters.h"
#include "tensorflow/compiler/xla/mlir_hlo/mhlo/utils/type_conversion.h"
namespace mlir::mhlo {

#define GEN_PASS_DEF_CHLOLEGALIZETOHLOPASS
#define GEN_PASS_DEF_STABLEHLOLEGALIZETOHLOPASS
#include "tensorflow/compiler/xla/mlir_hlo/mhlo/transforms/mhlo_passes.h.inc"

} // namespace mlir::mhlo

namespace {

using namespace mlir;
using namespace mlir::mhlo;
using namespace mlir::torch;
using namespace mlir::torch::Torch;

void populateStablehloPatterns(
    TypeConverter &typeConverter, RewritePatternSet &patterns,
    ConversionTarget &target, MLIRContext &context,
    torch_to_stablehlo::TorchToStablehloOptions options) {
  torch_to_stablehlo::populateBasicOpPatternsAndLegality(
      typeConverter, patterns, target, options);
  torch_to_stablehlo::populateViewLikeOpPatternsAndLegality(
      typeConverter, patterns, target, options);
  torch_to_stablehlo::populateGatherScatterOpPatternsAndLegality(
      typeConverter, patterns, target, options);
  torch_to_stablehlo::populateReductionOpPatternsAndLegality(
      typeConverter, patterns, target, options);
  torch_to_stablehlo::populateLinearOpPatternsAndLegality(
      typeConverter, patterns, target, options);
  torch_to_stablehlo::populatePoolingOpPatternsAndLegality(
      typeConverter, patterns, target, options);
}

void populateStablehloToHloPatterns(
    stablehlo::StablehloToHloTypeConverter &typeConverter,
    RewritePatternSet &patterns, ConversionTarget &target,
    MLIRContext &context) {
  stablehlo::populateStablehloToHloPatterns(&patterns, &typeConverter,
                                            &context);
  stablehlo::registerFuncOpsForTypeConversion(target, patterns, typeConverter);
}

class ConvertTorchToTF : public ConvertTorchToTFBase<ConvertTorchToTF> {
public:
  ConvertTorchToTF() = default;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<chlo::ChloDialect, stablehlo::StablehloDialect,
                tensor::TensorDialect, arith::ArithDialect, mhlo::MhloDialect,
                shape::ShapeDialect, scf::SCFDialect, TF::TensorFlowDialect>();

    TorchConversion::getBackendTypeConversionDependentDialects(registry);
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ConversionTarget target(*context);

    target.addLegalDialect<TF::TensorFlowDialect, func::FuncDialect>();

    target.addIllegalDialect<mhlo::MhloDialect, chlo::ChloDialect>();
    target.addIllegalDialect<stablehlo::StablehloDialect, tensor::TensorDialect,
                             arith::ArithDialect, shape::ShapeDialect,
                             scf::SCFDialect>();

    target.addLegalOp<chlo::MinimumBroadcastShapesOp>();

    // tf-legalize-hlo
    target.addLegalOp<func::CallOp, func::ConstantOp, arith::ConstantOp>();
    target.addLegalOp<mhlo::TupleOp>();

    TypeConverter typeConverter;
    typeConverter.addConversion([](Type type) { return type; });
    TorchConversion::setupBackendTypeConversion(target, typeConverter);

    RewritePatternSet patterns(context);

    bool enableStaticShape = true;
    bool enableI32Index = true;

    populateStablehloPatterns(typeConverter, patterns, target, getContext(),
                              torch_to_stablehlo::TorchToStablehloOptions{
                                  enableStaticShape,
                                  enableI32Index ? 32u : 64u,
                              });

    stablehlo::StablehloToHloTypeConverter stablehloToHloTypeConverter;
    populateStablehloToHloPatterns(stablehloToHloTypeConverter, patterns,
                                   target, getContext());

    bool legalize_broadcasts_ = true;
    bool expand_compositions_ = true;
    if (legalize_broadcasts_) {
      chlo::populateChloBroadcastingPatterns(&getContext(), &patterns);
    }

    if (expand_compositions_) {
      chlo::populateDecomposeChloPatterns(&getContext(), &patterns);
    } else {
      target.addLegalOp<chlo::NextAfterOp, chlo::PolygammaOp, chlo::ZetaOp>();
    }

    tf_to_torch::PopulateLegalizeHloToTfPatterns(&patterns, &getContext());

    // DenseSet<Operation *> legalizedOps;
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::torch::createConvertTorchToTFPass() {
  return std::make_unique<ConvertTorchToTF>();
}
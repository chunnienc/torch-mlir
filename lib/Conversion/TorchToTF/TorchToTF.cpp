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
using namespace mlir::torch;
using namespace mlir::torch::Torch;

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
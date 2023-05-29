//===- torch-mlir-opt.cpp - MLIR Optimizer Driver -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#ifndef TORCH_MLIR_ENABLE_STABLEHLO
#define TORCH_MLIR_ENABLE_STABLEHLO 1
#endif

#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "torch-mlir/InitAll.h"

// tensorflow includes
#include "tensorflow/compiler/mlir/tensorflow/dialect_registration.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/compiler/xla/mlir_hlo/lhlo/transforms/passes.h"
#include "tensorflow/compiler/xla/mlir_hlo/mhlo/transforms/passes.h"

#ifdef TORCH_MLIR_ENABLE_STABLEHLO
#include "mhlo/IR/hlo_ops.h"
#include "mhlo/transforms/passes.h"
#include "stablehlo/dialect/Register.h"
#endif

using namespace mlir;

int main(int argc, char **argv) {
  registerAllPasses();
  mlir::torch::registerAllPasses();

  DialectRegistry registry;
  registerAllDialects(registry);
  mlir::torch::registerAllDialects(registry);

#ifdef TORCH_MLIR_ENABLE_STABLEHLO
  mlir::stablehlo::registerAllDialects(registry);
  registry.insert<mlir::mhlo::MhloDialect>();
  mlir::mhlo::registerSymbolicShapeOptimizationPass();
  mlir::mhlo::registerStablehloLegalizeToHloPass();
  mlir::mhlo::registerChloLegalizeToHloPass();
  mlir::mhlo::registerHloLegalizeToLinalgPass();
  mlir::mhlo::registerTestUnfuseBatchNormPass();
#endif

  // Tensorflow dialects and passes
  mlir::RegisterAllTensorFlowDialects(registry);
  mlir::registerTransformsPasses();
  mlir::registerTensorFlowPasses();
  mlir::mhlo::registerAllMhloPasses();
  mlir::lmhlo::registerAllLmhloPasses();

  return mlir::asMainReturnCode(mlir::MlirOptMain(
      argc, argv, "MLIR modular optimizer driver\n", registry));
}

#!/bin/bash
set -euo pipefail

bazel run @torch-mlir//:torch-mlir-opt -- /models/resnet18_raw.mlir -o /models/test.mlir -pass-pipeline=" \
    builtin.module(torchscript-module-to-torch-backend-pipeline,torch-backend-to-stablehlo-backend-pipeline) \
    "
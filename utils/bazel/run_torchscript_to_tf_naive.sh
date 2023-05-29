#!/bin/bash
set -euo pipefail

PASS_PIPELINE="
    builtin.module(
        torchscript-module-to-torch-backend-pipeline,
        torch-backend-to-stablehlo-backend-pipeline,
        odml-print-op-stats,
        stablehlo-legalize-to-hlo,
        func.func(chlo-legalize-to-hlo),
        func.func(tf-legalize-hlo),
        odml-print-op-stats
    )"

bazel run @torch-mlir//:torch-mlir-opt -- \
    /models/resnet18_raw.mlir \
    -o /models/test.mlir \
    -pass-pipeline="$(echo $PASS_PIPELINE | sed 's/ //g')"
#!/bin/bash
set -euo pipefail

PASS_PIPELINE="
    builtin.module(
        torchscript-module-to-torch-backend-pipeline,
        torch-backend-to-stablehlo-backend-pipeline,
        # odml-print-op-stats,
        stablehlo-legalize-to-hlo,
        func.func(chlo-legalize-to-hlo),
        func.func(tf-legalize-hlo),
        print-op-stats,
    )"


# Remove comment lines
PASS_PIPELINE=$(sed -E 's/\#.*|(\/\/).*//g' <<< "$PASS_PIPELINE")
# Remove spaces
PASS_PIPELINE=$(echo $PASS_PIPELINE | sed 's/ //g')
# Remove trailing commas
PASS_PIPELINE=$(echo $PASS_PIPELINE | sed 's/,)/)/g')

bazel run @torch-mlir//:torch-mlir-opt -- \
    /models/resnet18_raw.mlir \
    -o /models/test.mlir \
    -pass-pipeline="$PASS_PIPELINE"
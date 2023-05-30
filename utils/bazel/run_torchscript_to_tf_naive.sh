#!/bin/bash
set -euo pipefail

# PASS_PIPELINE="
#     builtin.module(
#         torchscript-module-to-torch-backend-pipeline,
#         ###### Print torch ops ######
#         print-op-stats,
#         # func.func(convert-torch-to-stablehlo),
#         torch-backend-to-stablehlo-backend-pipeline,
#         stablehlo-legalize-to-hlo,
#         func.func(chlo-legalize-to-hlo),
#         func.func(tf-legalize-hlo),
#         ###### Print all converted ops ######
#         print-op-stats,
#     )"

PASS_PIPELINE="
    builtin.module(
        torchscript-module-to-torch-backend-pipeline,
        # torch-backend-to-stablehlo-backend-pipeline,
        func.func(torch-backend-to-tf),
        # func.func(tf-legalize-hlo),
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
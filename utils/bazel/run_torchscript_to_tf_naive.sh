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
        print-op-stats
    )"


PASS_PIPELINE_NO_COMMENTS=$(sed -E 's/\#.*|(\/\/).*//g' <<< "$PASS_PIPELINE")
PASS_PIPELINE_NO_COMMENTS_SPACES=$(echo $PASS_PIPELINE_NO_COMMENTS | sed 's/ //g')

bazel run @torch-mlir//:torch-mlir-opt -- \
    /models/resnet18_raw.mlir \
    -o /models/test.mlir \
    -pass-pipeline="$PASS_PIPELINE_NO_COMMENTS_SPACES"
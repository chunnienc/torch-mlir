#!/bin/bash
set -euo pipefail

OUTPUT_FILENAME=/models/test.mlir

OUTPUT_NAME=$(sed -E 's/(.mlir)?$//g' <<< $OUTPUT_FILENAME)
OUTPUT_FILENAME=$(sed -E 's/(.mlir)?$/.mlir/g' <<< $OUTPUT_FILENAME)

PASS_PIPELINE="
    builtin.module(
        torchscript-module-to-torch-backend-pipeline,
        snapshot-op-locations {tag=torch filename=${OUTPUT_NAME}_torch.mlir},
        ###### Print torch ops ######
        print-op-stats,
        torch-backend-to-stablehlo-backend-pipeline,
        snapshot-op-locations {tag=stablehlo filename=${OUTPUT_NAME}_stablehlo.mlir},
        stablehlo-legalize-to-hlo,
        func.func(chlo-legalize-to-hlo),
        snapshot-op-locations {tag=hlo filename=${OUTPUT_NAME}_hlo.mlir},
        func.func(tf-legalize-hlo),
        ###### Print TF/HLO ops ######
        print-op-stats,
    )"

PASS_PIPELINE_RAW=$PASS_PIPELINE
# Remove comment lines
PASS_PIPELINE=$(sed -E 's/\#.*|(\/\/).*//g' <<< $PASS_PIPELINE)
# Remove trailing commas
PASS_PIPELINE=$(echo $PASS_PIPELINE | sed -E 's/,\s+\)/)/g')

bazel run @torch-mlir//:torch-mlir-opt -- \
    /models/resnet18_raw.mlir \
    -o $OUTPUT_FILENAME \
    -pass-pipeline="$PASS_PIPELINE" \
    -mlir-print-debuginfo

echo "$PASS_PIPELINE_RAW"
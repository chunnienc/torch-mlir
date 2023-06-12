#!/bin/bash
set -euo pipefail

INPUT_FILENAME=/models/resnet18.raw.mlir
OUTPUT_FILENAME=/models/resnet18.output.mlir

if [[ ! "$INPUT_FILENAME" =~ ^.+\.raw\.mlir$ ]]; then
    echo "Input must have file extension '.raw.mlir', got '$INPUT_FILENAME'"
    exit 1
fi

INPUT_NAME=$(echo "$INPUT_FILENAME" | sed -E 's/\.raw\.mlir$//g')

PASS_PIPELINE="
    builtin.module(
        torchscript-module-to-torch-backend-pipeline,
        ###### Print torch ops ######
        snapshot-op-locations {tag=torch filename=${INPUT_NAME}.torch.mlir},
        print-op-stats,
        torch-backend-to-stablehlo-backend-pipeline,
        ###### Print stablehlo ops ######
        snapshot-op-locations {tag=stablehlo filename=${INPUT_NAME}.stablehlo.mlir},
        print-op-stats,
        canonicalize,
        odml-rename-entry-point-to-main,
        stablehlo-legalize-to-hlo,
        func.func(chlo-legalize-to-hlo),
        snapshot-op-locations {tag=hlo filename=${INPUT_NAME}.hlo.mlir},
        ###### Passes/Pipelines from tf_tfl_passes ######
        func.func(tfl-legalize-jax-random),
        canonicalize, cse, # Canonicalize, CSE etc.
        symbol-dce, # DCE for private symbols.
        tf-strip-noinline-attribute,
        mlir-inliner,
        func.func(mhlo-flatten-tuple),
        canonicalize, cse,
        func.func(tf-legalize-hlo),
        ###### Print TF/HLO ops ######
        snapshot-op-locations {tag=tf filename=${INPUT_NAME}.tf.mlir},
        print-op-stats,
    )"

PASS_PIPELINE="builtin.module(
    torchscript-module-to-torch-backend-pipeline,
    func.func(torch-backend-to-tf),
    print-op-stats,
)"

PASS_PIPELINE_RAW=$PASS_PIPELINE
# Remove comment lines
PASS_PIPELINE=$(sed -E 's/\#.*|(\/\/).*//g' <<< $PASS_PIPELINE)
# Remove trailing commas
PASS_PIPELINE=$(echo $PASS_PIPELINE | sed -E 's/,\s+\)/)/g')

bazel run @torch-mlir//:torch-mlir-opt --check_visibility=false -- \
    $INPUT_FILENAME \
    -o $OUTPUT_FILENAME \
    -pass-pipeline="$PASS_PIPELINE" \
    -mlir-print-debuginfo

echo "$PASS_PIPELINE_RAW"
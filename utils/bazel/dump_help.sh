#!/bin/bash
set -euo pipefail

bazel run @torch-mlir//:torch-mlir-opt --check_visibility=false -- -h | tee ./torch-mlir-opt-help.txt
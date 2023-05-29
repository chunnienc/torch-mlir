#!/bin/bash
set -euo pipefail

bazel run @torch-mlir//:torch-mlir-opt -- -h | tee ./torch-mlir-opt-help.txt
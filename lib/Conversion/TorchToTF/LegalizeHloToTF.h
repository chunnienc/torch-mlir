#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir::tf_to_torch {

// Addds the HLO to TF rewrite patterns to the specified pattern list.
void PopulateLegalizeHloToTfPatterns(RewritePatternSet *patterns,
                                     MLIRContext *context);

} // namespace mlir::TF::torch

#ifndef TORCHMLIR_LIB_CONVERSION_TORCHTOTF_POPULATEPATTERNS_H
#define TORCHMLIR_LIB_CONVERSION_TORCHTOTF_POPULATEPATTERNS_H

#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace torch {
namespace torch_to_tf {

struct TorchToTFOptions {};

template <typename AtenOpT>
class ConvertAtenOp : public OpConversionPattern<AtenOpT> {
public:
  using OpAdaptor = typename AtenOpT::Adaptor;
  ConvertAtenOp(TypeConverter &typeConverter, MLIRContext *context,
                const TorchToTFOptions &options)
      : OpConversionPattern<AtenOpT>(typeConverter, context) {
    this->options = options;
  }
  LogicalResult
  matchAndRewrite(AtenOpT op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    return rewriter.notifyMatchFailure(op, "haven't been implemented");
  }
  const TorchToTFOptions &getOptions() const { return options; }

private:
  TorchToTFOptions options;
};

void populatePoolingOpPatternsAndLegality(TypeConverter &typeConverter,
                                          RewritePatternSet &patterns,
                                          ConversionTarget &target,
                                          const TorchToTFOptions &options);

} // namespace torch_to_tf
} // namespace torch
} // namespace mlir

#endif // TORCHMLIR_LIB_CONVERSION_TORCHTOSTABLEHLO_POPULATEPATTERNS_H

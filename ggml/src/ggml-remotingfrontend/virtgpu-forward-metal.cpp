#include "virtgpu-forward-impl.h"

bool
apir_metal_get_device_context(struct virtgpu *gpu, struct ggml_backend_metal_device_context *metal_dev_ctx) {
  struct apir_encoder *encoder;
  struct apir_decoder *decoder;
  ApirForwardReturnCode ret;

  REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_METAL_GET_DEVICE_CONTEXT);

  REMOTE_CALL(gpu, encoder, decoder, ret);

  apir_decode_bool_t(decoder, &metal_dev_ctx->has_simdgroup_mm);
  apir_decode_bool_t(decoder, &metal_dev_ctx->has_simdgroup_reduction);
  apir_decode_bool_t(decoder, &metal_dev_ctx->has_bfloat);

  remote_call_finish(gpu, encoder, decoder);

  return true;
}

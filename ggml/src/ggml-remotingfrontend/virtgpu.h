#pragma once

#include <xf86drm.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdbool.h>
#include <threads.h>
#include <cstring>
#include <sys/stat.h>
#include <sys/sysmacros.h>

#include "virtgpu-forward.h"
#include "virtgpu-utils.h"
#include "../ggml-remotingbackend/shared/api_remoting.h"
#include "../ggml-remotingbackend/shared/venus_cs.h"

#include "virtgpu-shm.h"

#define VIRGL_RENDERER_UNSTABLE_APIS 1
#include "drm-uapi/virtgpu_drm.h"
#include "venus_hw.h"

// must match https://gitlab.freedesktop.org/kpouget/virglrenderer/-/blob/main/src/virglrenderer_hw.h?ref_type=heads
enum virgl_renderer_capset {
  VIRGL_RENDERER_CAPSET_VIRGL                   = 1,
  VIRGL_RENDERER_CAPSET_VIRGL2                  = 2,
  /* 3 is reserved for gfxstream */
  VIRGL_RENDERER_CAPSET_VENUS                   = 4,
  /* 5 is reserved for cross-domain */
  VIRGL_RENDERER_CAPSET_DRM                     = 6,
};

/* from src/virtio/vulkan/vn_renderer_virtgpu.c */
#define VIRTGPU_PCI_VENDOR_ID 0x1af4
#define VIRTGPU_PCI_DEVICE_ID 0x1050
#define VIRTGPU_BLOB_MEM_GUEST_VRAM 0x0004
#define VIRTGPU_PARAM_GUEST_VRAM 9

#define ARRAY_SIZE(x) (sizeof(x) / sizeof((x)[0]))

typedef enum virt_gpu_result_t {
  APIR_SUCCESS = 0,
  APIR_ERROR_INITIALIZATION_FAILED = -1,
} virt_gpu_result_t;


struct remoting_dev_instance {
  int yes;
};

#define PRINTFLIKE(f, a) __attribute__ ((format(__printf__, f, a)))

inline void
vn_log(struct remoting_dev_instance *instance, const char *format, ...)
  PRINTFLIKE(2, 3);


struct virtgpu {
  struct remoting_dev_instance *instance;

  int fd;

  struct {
    enum virgl_renderer_capset id;
    uint32_t version;
    struct virgl_renderer_capset_apir data;
  } capset;

  struct util_sparse_array shmem_array;

  /* APIR */
  struct vn_renderer_shmem *reply_shmem;
  struct vn_renderer_shmem *data_shmem;
};


static inline int
virtgpu_ioctl(struct virtgpu *gpu, unsigned long request, void *args)
{
  return drmIoctl(gpu->fd, request, args);
}

struct virtgpu *create_virtgpu();

struct vn_cs_encoder *remote_call_prepare(
  struct virtgpu *gpu,
  ApirCommandType apir_cmd_type,
  int32_t cmd_flags);

uint32_t remote_call(
  struct virtgpu *gpu,
  struct vn_cs_encoder *enc,
  struct vn_cs_decoder **dec,
  float max_wait_ms,
  long long *call_duration_ns
);

void remote_call_finish(
  struct virtgpu *gpu,
  struct vn_cs_encoder *enc,
  struct vn_cs_decoder *dec);

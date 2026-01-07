#pragma once

#include <cassert>
#include <cstdint>
#include <cstddef>
#include <atomic>
#include <sys/mman.h>

#include "virtgpu.h"
#include "virtgpu-utils.h"

int virtgpu_shmem_create(struct virtgpu *gpu, size_t size, struct virtgpu_shmem *shmem);
void virtgpu_shmem_destroy(struct virtgpu *gpu, struct virtgpu_shmem *shmem);

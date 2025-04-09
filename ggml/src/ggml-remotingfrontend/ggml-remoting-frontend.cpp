#include "ggml-remoting-frontend.h"

#include <ostream>
#include <iostream>
#include <mutex>
#include <memory>
#include <chrono>
#include <thread>

#include "ggml-impl.h"
#include "ggml-backend-impl.h"

#define RMT_LOG_DEBUG(msg) std::cerr << msg << std::endl

#define UNUSED GGML_UNUSED

int ggml_backend_remoting_get_device_count();
ggml_backend_buffer_type_t ggml_backend_remoting_host_buffer_type();

static void * const remoting_ptr_base = (void *)(uintptr_t) 0x1000;  // NOLINT


struct ggml_backend_remoting_buffer_type_context {
    std::string name;
};

struct remoting_context_struct {
   int i;
};
typedef std::shared_ptr<remoting_context_struct> remoting_context;
typedef std::weak_ptr<remoting_context_struct> remoting_context_ref;

static size_t ggml_backend_remoting_reg_get_device_count(ggml_backend_reg_t reg) {
    UNUSED(reg);
    return ggml_backend_remoting_get_device_count();
}

static const char * ggml_backend_remoting_reg_get_name(ggml_backend_reg_t reg) {
    UNUSED(reg);
    return GGML_REMOTING_NAME;
}

struct ggml_backend_remoting_device_context {
    size_t device;
    std::string name;
    std::string description;
};

static const char * ggml_backend_remoting_device_get_name(ggml_backend_dev_t dev) {
    UNUSED(dev);
    return "API Remoting";
}

static const char * ggml_backend_remoting_device_get_description(ggml_backend_dev_t dev) {
    UNUSED(dev);
    return "API Remoting device";
}

static enum ggml_backend_dev_type ggml_backend_remoting_device_get_type(ggml_backend_dev_t dev) {
    UNUSED(dev);
    return GGML_BACKEND_DEVICE_TYPE_GPU;
}

static void ggml_backend_remoting_device_get_memory(ggml_backend_dev_t device, size_t * free, size_t * total) {
    UNUSED(device);
    *total = 1024*1024*1024;
    *free = *total;
}

struct remoting_device_struct {
    std::mutex mutex;
};

struct remoting_device_struct;
typedef std::shared_ptr<remoting_device_struct> remoting_device;
typedef std::weak_ptr<remoting_device_struct> remoting_device_ref;

struct remoting_buffer_struct;
typedef std::shared_ptr<remoting_buffer_struct> remoting_buffer;
typedef std::weak_ptr<remoting_buffer_struct> remoting_buffer_ref;

// vk buffer type
static const char * ggml_backend_remoting_buffer_type_name(ggml_backend_buffer_type_t buft) {
    UNUSED(buft);

    return "Remoting buffer";
}

static void ggml_remoting_destroy_buffer(remoting_buffer& buf) {
    UNUSED(buf);
}


static void ggml_remoting_buffer_memset(remoting_buffer& dst, size_t offset, uint32_t c, size_t size) {
  UNUSED(dst);
  UNUSED(c);
  UNUSED(size);
  UNUSED(offset);
}

static void ggml_remoting_buffer_memset_async(remoting_context& ctx, remoting_buffer& dst, size_t offset, uint32_t c, size_t size) {
  UNUSED(ctx);
  UNUSED(dst);
  UNUSED(c);
  UNUSED(size);
  UNUSED(offset);
}


static uint64_t remoting_tensor_offset(const ggml_tensor * tensor) {
    if (tensor->view_src) {
        return (uint8_t *) tensor->view_src->data - (uint8_t *) remoting_ptr_base;
    }
    return (uint8_t *) tensor->data - (uint8_t *) remoting_ptr_base;
}

struct ggml_backend_remoting_buffer_context {
    remoting_device_ref device;
    remoting_buffer dev_buffer;
    std::string name;

    ggml_backend_remoting_buffer_context(remoting_device_ref device, remoting_buffer&& dev_buffer, std::string& name) :
        name(name) {
        UNUSED(device);
	UNUSED(dev_buffer);
    }

    ~ggml_backend_remoting_buffer_context() {
        ggml_remoting_destroy_buffer(dev_buffer);
    }
};

static void ggml_backend_remoting_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_backend_remoting_buffer_context * ctx = (ggml_backend_remoting_buffer_context *)buffer->context;
    ggml_remoting_destroy_buffer(ctx->dev_buffer);
    delete ctx;
}

static void * ggml_backend_remoting_buffer_get_base(ggml_backend_buffer_t buffer) {
    return (void *) 4096;

    UNUSED(buffer);
}

static enum ggml_status ggml_backend_remoting_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    if (tensor->view_src != nullptr) {
        GGML_ASSERT(tensor->view_src->buffer->buft == buffer->buft);
    }
    return GGML_STATUS_SUCCESS;
}

static void ggml_backend_remoting_buffer_memset_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
  UNUSED(buffer);
  UNUSED(tensor);
  UNUSED(value);
  UNUSED(offset);
  UNUSED(size);
}

static void ggml_remoting_buffer_write(remoting_buffer& dst, size_t offset, const void * src, size_t size) {
    UNUSED(dst);
    UNUSED(offset);
    UNUSED(src);
    UNUSED(size);
}

static void ggml_remoting_buffer_read(remoting_buffer& src, size_t offset, void * dst, size_t size) {
    UNUSED(src);
    UNUSED(offset);
    UNUSED(dst);
    UNUSED(size);
}

static void ggml_backend_remoting_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
#if 0
    ggml_backend_remoting_buffer_context * buf_ctx = (ggml_backend_remoting_buffer_context *)buffer->context;
    remoting_buffer buf = buf_ctx->dev_buffer;

    ggml_remoting_buffer_write(buf, remoting_tensor_offset(tensor) + tensor->view_offs + offset, data, size);
#else
    UNUSED(buffer);
    UNUSED(tensor);
    UNUSED(data);
    UNUSED(offset);
    UNUSED(size);
#endif
}

static void ggml_backend_remoting_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
#if 0
    ggml_backend_remoting_buffer_context * buf_ctx = (ggml_backend_remoting_buffer_context *)buffer->context;

    remoting_buffer buf = buf_ctx->dev_buffer;

    ggml_remoting_buffer_read(buf, remoting_tensor_offset(tensor) + tensor->view_offs + offset, data, size);
#else
    UNUSED(buffer);
    UNUSED(tensor);
    UNUSED(data);
    UNUSED(offset);
    UNUSED(size);
#endif
}

static void ggml_remoting_buffer_copy_async(remoting_context& ctx, remoting_buffer& dst, size_t dst_offset, remoting_buffer& src, size_t src_offset, size_t size) {
  UNUSED(ctx);
  UNUSED(dst);
  UNUSED(dst_offset);
  UNUSED(src);
  UNUSED(src_offset);
  UNUSED(size);
}

static bool ggml_backend_remoting_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * src, ggml_tensor * dst) {
  return true;

  UNUSED(buffer);
  UNUSED(src);
  UNUSED(dst);
}

static void ggml_backend_remoting_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    ggml_backend_remoting_buffer_context * ctx = (ggml_backend_remoting_buffer_context *)buffer->context;

    ggml_remoting_buffer_memset(ctx->dev_buffer, 0, value, buffer->size);
}

static ggml_backend_buffer_i ggml_backend_remoting_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_remoting_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_remoting_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_remoting_buffer_init_tensor,
    /* .memset_tensor   = */ ggml_backend_remoting_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_remoting_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_remoting_buffer_get_tensor,
    /* .cpy_tensor      = */ ggml_backend_remoting_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_remoting_buffer_clear,
    /* .reset           = */ NULL,
};

static ggml_backend_buffer_t ggml_backend_remoting_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    ggml_backend_remoting_buffer_type_context * ctx = (ggml_backend_remoting_buffer_type_context *) buft->context;


    return ggml_backend_buffer_init(buft, ggml_backend_remoting_buffer_interface, ctx, size);
}

static size_t ggml_backend_remoting_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    UNUSED(buft);
    return 4096;
}

static size_t ggml_backend_remoting_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {
    UNUSED(buft);
    return 40960;
}

static size_t ggml_backend_remoting_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    UNUSED(buft);
    UNUSED(tensor);
    return ggml_nbytes(tensor);
}

static ggml_backend_buffer_type_i ggml_backend_remoting_buffer_type_interface = {
    /* .get_name         = */ ggml_backend_remoting_buffer_type_name,
    /* .alloc_buffer     = */ ggml_backend_remoting_buffer_type_alloc_buffer,
    /* .get_alignment    = */ ggml_backend_remoting_buffer_type_get_alignment,
    /* .get_max_size     = */ ggml_backend_remoting_buffer_type_get_max_size,
    /* .get_alloc_size   = */ ggml_backend_remoting_buffer_type_get_alloc_size,
    /* .is_host          = */ NULL,
};

static ggml_backend_buffer_type_t ggml_backend_remoting_device_get_buffer_type(ggml_backend_dev_t dev) {

    static struct ggml_backend_buffer_type buft {
      /* .iface    = */ ggml_backend_remoting_buffer_type_interface,
      /* .device   = */ dev,
      /* .context  = */ new ggml_backend_remoting_buffer_type_context{ "device_name"},
    };

    return & buft;
}

static bool ggml_backend_remoting_device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
  UNUSED(dev);
  UNUSED(op);

  return true;
}

static bool ggml_backend_remoting_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    UNUSED(dev);
    UNUSED(buft);
    return true;
}


static bool ggml_backend_remoting_device_offload_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    const int min_batch_size = 32;

    return (op->ne[1] >= min_batch_size && op->op != GGML_OP_GET_ROWS) ||
           (op->ne[2] >= min_batch_size && op->op == GGML_OP_MUL_MAT_ID);

    UNUSED(dev);
}

static const char * ggml_backend_remoting_name(ggml_backend_t backend) {
    UNUSED(backend);

    return "API Remoting backend";
}

static void ggml_backend_remoting_free(ggml_backend_t backend) {
    UNUSED(backend);
}

static ggml_status ggml_backend_remoting_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    UNUSED(backend);
    UNUSED(cgraph);

    return GGML_STATUS_SUCCESS;
}

static ggml_backend_i ggml_backend_remoting_interface = {
    /* .get_name                = */ ggml_backend_remoting_name,
    /* .free                    = */ ggml_backend_remoting_free,
    /* .set_tensor_async        = */ NULL,  // ggml_backend_remoting_set_tensor_async,
    /* .get_tensor_async        = */ NULL,  // ggml_backend_remoting_get_tensor_async,
    /* .cpy_tensor_async        = */ NULL,  // ggml_backend_remoting_cpy_tensor_async,
    /* .synchronize             = */ NULL,  // ggml_backend_remoting_synchronize,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_remoting_graph_compute,
    /* .event_record            = */ NULL,
    /* .event_wait              = */ NULL,
};

static void ggml_backend_remoting_device_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props) {
    props->name        = ggml_backend_remoting_device_get_name(dev);
    props->description = ggml_backend_remoting_device_get_description(dev);
    props->type        = ggml_backend_remoting_device_get_type(dev);
    ggml_backend_remoting_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                 = */ false,
        /* .host_buffer           = */ true,
        /* .buffer_from_host_ptr  = */ false,
        /* .events                = */ false,
    };
}

static ggml_guid_t ggml_backend_remoting_guid() {
    static ggml_guid guid = { 0xb8, 0xf7, 0x4f, 0x86, 0x40, 0x3c, 0xe1, 0x02, 0x91, 0xc8, 0xdd, 0xe9, 0x02, 0x3f, 0xc0, 0x2b };
    return &guid;
}


static ggml_backend_t ggml_backend_remoting_device_init(ggml_backend_dev_t dev, const char * params) {
    UNUSED(params);
    ggml_backend_remoting_device_context * ctx = (ggml_backend_remoting_device_context *)dev->context;

    ggml_backend_t remoting_backend = new ggml_backend {
        /* .guid      = */ ggml_backend_remoting_guid(),
        /* .interface = */ ggml_backend_remoting_interface,
        /* .device    = */ ggml_backend_reg_dev_get(ggml_backend_remoting_reg(), ctx->device),
        /* .context   = */ ctx,
    };

    return remoting_backend;
}

// host buffer type

static const char * ggml_backend_remoting_host_buffer_type_name(ggml_backend_buffer_type_t buft) {
    return GGML_REMOTING_NAME "_Host";

    UNUSED(buft);
}

static void ggml_backend_remoting_host_buffer_free_buffer(ggml_backend_buffer_t buffer) {
# if 0
    ggml_remoting_host_free(remoting_instance.devices[0], buffer->context);
#endif
    UNUSED(buffer);
}

static ggml_backend_buffer_t ggml_backend_remoting_host_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {

    void *ptr = nullptr;
    ggml_backend_buffer_t buffer = ggml_backend_cpu_buffer_from_ptr(ptr, size);
    buffer->buft = buft;
    buffer->iface.free_buffer = ggml_backend_remoting_host_buffer_free_buffer;

    return buffer;
    UNUSED(buft);
}

static size_t ggml_backend_remoting_host_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
  UNUSED(buft);
  return 4096;
}

// Should be changed to return device-specific host buffer type
// but that probably requires changes in llama.cpp
ggml_backend_buffer_type_t ggml_backend_remoting_host_buffer_type() {
    static struct ggml_backend_buffer_type ggml_backend_remoting_buffer_type_host = {
        /* .iface    = */ {
            /* .get_name         = */ ggml_backend_remoting_host_buffer_type_name,
            /* .alloc_buffer     = */ ggml_backend_remoting_host_buffer_type_alloc_buffer,
            /* .get_alignment    = */ ggml_backend_remoting_host_buffer_type_get_alignment,
            /* .get_max_size     = */ NULL, // defaults to SIZE_MAX
            /* .get_alloc_size   = */ ggml_backend_cpu_buffer_type()->iface.get_alloc_size,
            /* .is_host          = */ ggml_backend_cpu_buffer_type()->iface.is_host,
        },
        /* .device   = */ ggml_backend_reg_dev_get(ggml_backend_remoting_reg(), 0),
        /* .context  = */ nullptr,
    };

    // Make sure device 0 is initialized
    //ggml_remoting_instance_init();
    //ggml_remoting_get_device(0);

    return &ggml_backend_remoting_buffer_type_host;
}

static ggml_backend_buffer_type_t ggml_backend_remoting_device_get_host_buffer_type(ggml_backend_dev_t dev) {
    UNUSED(dev);
    return ggml_backend_remoting_host_buffer_type();
}

static const struct ggml_backend_device_i ggml_backend_remoting_device_i = {
    /* .get_name             = */ ggml_backend_remoting_device_get_name,
    /* .get_description      = */ ggml_backend_remoting_device_get_description,
    /* .get_memory           = */ ggml_backend_remoting_device_get_memory,
    /* .get_type             = */ ggml_backend_remoting_device_get_type,
    /* .get_props            = */ ggml_backend_remoting_device_get_props,
    /* .init_backend         = */ ggml_backend_remoting_device_init,
    /* .get_buffer_type      = */ ggml_backend_remoting_device_get_buffer_type,
    /* .get_host_buffer_type = */ ggml_backend_remoting_device_get_host_buffer_type,
    /* .buffer_from_host_ptr = */ NULL,
    /* .supports_op          = */ ggml_backend_remoting_device_supports_op,
    /* .supports_buft        = */ ggml_backend_remoting_device_supports_buft,
    /* .offload_op           = */ ggml_backend_remoting_device_offload_op,
    /* .event_new            = */ NULL,
    /* .event_free           = */ NULL,
    /* .event_synchronize    = */ NULL,
};

static ggml_backend_dev_t ggml_backend_remoting_reg_get_device(ggml_backend_reg_t reg, size_t device) {
    static std::vector<ggml_backend_dev_t> devices;

    static bool initialized = false;

    {
        static std::mutex mutex;
        std::lock_guard<std::mutex> lock(mutex);
        if (!initialized) {
            for (size_t i = 0; i < ggml_backend_remoting_reg_get_device_count(reg); i++) {
                ggml_backend_remoting_device_context * ctx = new ggml_backend_remoting_device_context;
                char desc[256] = "API Remoting device";

                ctx->device = i;
                ctx->name = GGML_REMOTING_NAME + std::to_string(i);
                ctx->description = desc;
                devices.push_back(new ggml_backend_device {
                    /* .iface   = */ ggml_backend_remoting_device_i,
                    /* .reg     = */ reg,
                    /* .context = */ ctx,
                });
            }
            initialized = true;
        }
    }

    GGML_ASSERT(device < devices.size());
    return devices[device];
}

int ggml_backend_remoting_get_device_count() {
    return 1;
}

static const struct ggml_backend_reg_i ggml_backend_remoting_reg_i = {
    /* .get_name         = */ ggml_backend_remoting_reg_get_name,
    /* .get_device_count = */ ggml_backend_remoting_reg_get_device_count,
    /* .get_device       = */ ggml_backend_remoting_reg_get_device,
    /* .get_proc_address = */ NULL,
};

ggml_backend_reg_t ggml_backend_remoting_reg() {
    static ggml_backend_reg reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_remoting_reg_i,
        /* .context     = */ nullptr,
    };

    RMT_LOG_DEBUG("ggml_backend_remoting_frontend_reg() hello :wave:");
    return &reg;
}

#pragma once
#include <stdint.h>
#include "core/protocol.h"
#include "core/utils.h" // for ProfilerEntry

// Forward declaration - actual Transport comes from HAL
class Transport;

void proto_send_memprof(Transport& tr,
                        uint32_t seq,
                        uint32_t trained,
                        uint32_t tested,
                        float acc,
                        uint32_t free_heap,
                        uint32_t min_heap,
                        uint32_t max_alloc,
                        uint64_t util_cur,
                        uint64_t util_peak,
                        uint32_t alloc_count,
                        uint32_t free_count,
                        uint32_t active_alloc,
                        const ProfilerEntry* entries,
                        size_t entry_count);



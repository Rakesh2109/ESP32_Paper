#include "core/protocol_utils.h"
#include "config/bnn_config.h"
#include "hal/hal_platform.h"  // Get Transport definition
#include <string.h>

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
                        size_t entry_count){
  MemProfHeader hdr{};
  hdr.trained_le = trained;
  hdr.tested_le  = tested;
  hdr.acc_le     = acc;
  hdr.free_heap_le      = free_heap;
  hdr.min_heap_le       = min_heap;
  hdr.max_alloc_heap_le = max_alloc;
  hdr.util_cur_alloc_le    = util_cur;
  hdr.util_peak_alloc_le   = util_peak;
  hdr.util_alloc_count_le  = alloc_count;
  hdr.util_free_count_le   = free_count;
  hdr.util_active_allocs_le= active_alloc;
  hdr.total_entries_le = (uint32_t)entry_count;

  static uint8_t buf[MEMPROF_MAX_PAYLOAD];
  size_t off = 0;
  if (sizeof(MemProfHeader) > sizeof(buf)) return;
  memcpy(&buf[off], &hdr, sizeof(MemProfHeader));
  off += sizeof(MemProfHeader);

  uint32_t sent = 0;
  bool truncated = false;
  for (size_t i=0; i<entry_count; ++i){
    const ProfilerEntry& e = entries[i];
    const char* name = e.name ? e.name : "NULL";
    uint16_t nlen = (uint16_t)strnlen(name, 255);
    size_t need = 2 + (size_t)nlen + 8 + 4 + 8 + 8;
    if (off + need > MEMPROF_MAX_PAYLOAD){ truncated = true; break; }
    memcpy(&buf[off], &nlen, 2); off += 2;
    memcpy(&buf[off], name, nlen); off += nlen;
    memcpy(&buf[off], &e.total_us, 8); off += 8;
    memcpy(&buf[off], &e.count,    4); off += 4;
    memcpy(&buf[off], &e.max_us,   8); off += 8;
    memcpy(&buf[off], &e.last_us,  8); off += 8;
    sent++;
  }

  ((MemProfHeader*)buf)->sent_entries_le = sent;
  ((MemProfHeader*)buf)->truncated      = truncated ? 1 : 0;
  tr.sendRawFrame(FRAME_TYPE_MEMPROF, seq, buf, (uint16_t)off);
}



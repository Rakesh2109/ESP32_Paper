#include "core/transport.h"
#include "freertos/FreeRTOS.h"

#ifdef NATIVE_BUILD
#include <mutex>
#else
#include "freertos/semphr.h"
#endif

namespace {
#ifdef NATIVE_BUILD
std::mutex g_transport_tx_mutex;
#else
SemaphoreHandle_t g_transport_tx_mutex = nullptr;
portMUX_TYPE g_transport_init_mux = portMUX_INITIALIZER_UNLOCKED;
#endif
}

void Transport::lockTx() {
#ifdef NATIVE_BUILD
  g_transport_tx_mutex.lock();
#else
  if (!g_transport_tx_mutex) {
    taskENTER_CRITICAL(&g_transport_init_mux);
    if (!g_transport_tx_mutex) {
      g_transport_tx_mutex = xSemaphoreCreateMutex();
    }
    taskEXIT_CRITICAL(&g_transport_init_mux);
  }
  if (g_transport_tx_mutex) {
    xSemaphoreTake(g_transport_tx_mutex, portMAX_DELAY);
  }
#endif
}

void Transport::unlockTx() {
#ifdef NATIVE_BUILD
  g_transport_tx_mutex.unlock();
#else
  if (g_transport_tx_mutex) {
    xSemaphoreGive(g_transport_tx_mutex);
  }
#endif
}

bool Transport::readExact(uint8_t* dst, size_t len, unsigned long to_ms){
  const unsigned long start = millis();
  size_t got = 0;
  while (got < len){
    if (millis() - start > to_ms) {
      return false;
    }
    const int avail = serial.available();
    if (avail <= 0) {
      taskYIELD();
      continue;
    }
    const size_t avail_sz = static_cast<size_t>(avail);
    const size_t remain = len - got;
    const size_t want = (avail_sz < remain) ? avail_sz : remain;
    const size_t n = serial.readBytes(dst + got, want);
    if (n > 0) {
      got += n;
      continue;
    }
    taskYIELD();
  }
  return true;
}

bool Transport::syncMagic(unsigned long to_ms){
  unsigned long start = millis(); bool first=false;
  while (millis()-start <= to_ms){
    int c = serial.read();
    if (c < 0) { taskYIELD(); continue; }
    // Check for magic bytes: 0x5A (first byte) then 0xA5 (second byte)
    if (!first) first = (c==0x5A);
    else { if (c==0xA5) return true; first = (c==0x5A); }
  }
  return false;
}

bool Transport::readHeader(FrameHeader& h, bool& with_chk, uint16_t& chk_seed, unsigned long to_ms){
  if (!syncMagic(to_ms)) return false;
  uint8_t rest[8]; if (!readExact(rest, sizeof(rest), to_ms)) return false;
  h.magic = FRAME_MAGIC; h.type=rest[0]; h.flags=rest[1]; h.len = (uint16_t)(rest[2]|(rest[3]<<8));
  h.seq = (uint32_t)rest[4] | ((uint32_t)rest[5]<<8) | ((uint32_t)rest[6]<<16) | ((uint32_t)rest[7]<<24);
  with_chk = (h.flags & FLAG_FLETCHER) != 0;
  chk_seed = 0;
  if (with_chk){
    chk_seed = fletcher16_update(chk_seed, &h.type, 1);
    chk_seed = fletcher16_update(chk_seed, &h.flags, 1);
    chk_seed = fletcher16_update(chk_seed, (uint8_t*)&h.len, 2);
    chk_seed = fletcher16_update(chk_seed, (uint8_t*)&h.seq, 4);
  }
  return true;
}

void Transport::sendRawFrame(uint8_t type, uint32_t seq, const uint8_t* payload, uint16_t len){
  FrameHeader fh{FRAME_MAGIC,type,FLAG_FLETCHER,len,seq};
  uint16_t chk=0;
  chk = fletcher16_update(chk, &fh.type, 1);
  chk = fletcher16_update(chk, &fh.flags, 1);
  chk = fletcher16_update(chk, (uint8_t*)&fh.len, 2);
  chk = fletcher16_update(chk, (uint8_t*)&fh.seq, 4);
  if (payload && len) chk = fletcher16_update(chk, payload, len);
  lockTx();
  serial.write((uint8_t*)&fh, sizeof(fh));
  if (payload && len) serial.write(payload, len);
  serial.write((uint8_t*)&chk, 2);
  unlockTx();
}

void Transport::sendReady(uint8_t major, uint8_t minor, uint32_t trained, uint32_t tested){ ReadyPayload p{major,minor,0,trained,tested}; sendFramed(FRAME_TYPE_READY, 0, p); }
void Transport::sendAck(uint32_t cnt, uint32_t last_seq){ AckDonePayload p{cnt,last_seq}; sendFramed(FRAME_TYPE_ACK,last_seq,p); }
void Transport::sendDone(uint32_t cnt, uint32_t last_seq){ AckDonePayload p{cnt,last_seq}; sendFramed(FRAME_TYPE_DONE,last_seq,p); }
void Transport::sendError(uint8_t code, uint32_t seq){ struct { uint8_t c; } e{code}; sendFramed(FRAME_TYPE_ERROR,seq,e); }
void Transport::sendStat(uint32_t seq, uint32_t trained, uint32_t tested, float acc){ StatPayload sp{trained,tested,acc}; sendFramed(FRAME_TYPE_STAT, seq, sp); }
void Transport::sendMetric(uint32_t seq, const MetricPayload& mp){ sendFramed(FRAME_TYPE_METRIC, seq, mp); }
void Transport::sendLog(const char* msg, uint16_t len){ sendRawFrame(FRAME_TYPE_LOG, 0, (const uint8_t*)msg, len); }
void Transport::sendMemEfficiency(const MemEfficiencyPayload& payload){ sendFramed(FRAME_TYPE_MEM_EFFICIENCY, 0, payload); }
void Transport::sendPerfBench(const PerfBenchPayload& payload){ sendFramed(FRAME_TYPE_PERF_BENCH, 0, payload); }
void Transport::sendProtocolStats(const ProtocolStatsPayload& payload){ sendFramed(FRAME_TYPE_PROTOCOL_STATS, 0, payload); }



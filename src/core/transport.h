#pragma once
#include <Arduino.h>
#include <stdint.h>
#include "core/protocol.h"

class Transport {
public:
  //explicit Transport(HardwareSerial& ser) : serial(ser) {}
  explicit Transport(Stream& ser) : serial(ser) {}
  //explicit Transport(Adafruit_ST7789& tft) : tft(tft) {}
  bool readExact(uint8_t* dst, size_t len, unsigned long to_ms);
  bool syncMagic(unsigned long to_ms);
  bool readHeader(FrameHeader& h, bool& with_chk, uint16_t& chk_seed, unsigned long to_ms);

  template<class T>
  void sendFramed(uint8_t type, uint32_t seq, const T& payload){
    sendRawFrame(type, seq, reinterpret_cast<const uint8_t*>(&payload), static_cast<uint16_t>(sizeof(T)));
  }

  void sendRawFrame(uint8_t type, uint32_t seq, const uint8_t* payload, uint16_t len);

  void sendReady(uint8_t major, uint8_t minor, uint32_t trained, uint32_t tested);
  void sendAck(uint32_t cnt, uint32_t last_seq);
  void sendDone(uint32_t cnt, uint32_t last_seq);
  void sendError(uint8_t code, uint32_t seq);
  void sendStat(uint32_t seq, uint32_t trained, uint32_t tested, float acc);
  void sendMetric(uint32_t seq, const MetricPayload& mp);
  void sendLog(const char* msg, uint16_t len);
  void sendMemEfficiency(const MemEfficiencyPayload& payload);
  void sendPerfBench(const PerfBenchPayload& payload);
  void sendProtocolStats(const ProtocolStatsPayload& payload);

private:
  void lockTx();
  void unlockTx();
  //HardwareSerial& serial;
  Stream& serial;
};



#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void parser_unpack_bits_msb(const uint8_t* packed, int n, uint8_t* out);
void parser_parse_packed_header(uint32_t header_le, int* label, int* nfeat);

#ifdef __cplusplus
}
#endif



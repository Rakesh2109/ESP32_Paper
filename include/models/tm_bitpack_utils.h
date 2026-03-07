#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace tm_model::detail {

inline const std::uint8_t* reverse_lut() {
    static std::uint8_t lut[256];
    static bool init = false;
    if (!init) {
        for (unsigned i = 0; i < 256u; ++i) {
            std::uint8_t x = static_cast<std::uint8_t>(i);
            x = static_cast<std::uint8_t>((x >> 4) | (x << 4));
            x = static_cast<std::uint8_t>(((x & 0xCCu) >> 2) | ((x & 0x33u) << 2));
            x = static_cast<std::uint8_t>(((x & 0xAAu) >> 1) | ((x & 0x55u) << 1));
            lut[i] = x;
        }
        init = true;
    }
    return lut;
}

inline void pack_msb_bits_to_words(const std::uint8_t* packed_bits,
                                   std::size_t nfeat,
                                   std::size_t input_dim,
                                   std::vector<std::uint32_t>& packed_words) {
    const std::size_t nwords = (input_dim + 31u) >> 5;
    if (packed_words.size() != nwords) {
        packed_words.resize(nwords, 0u);
    }

    const std::size_t n = std::min(nfeat, input_dim);
    if (n == 0u || nwords == 0u) {
        std::fill(packed_words.begin(), packed_words.end(), 0u);
        return;
    }

    const std::uint8_t* rev = reverse_lut();
    const std::size_t nbytes = (n + 7u) >> 3;
    std::size_t wi = 0;
    std::size_t bi = 0;
    while (bi + 4u <= nbytes && wi < nwords) {
        const std::uint32_t w =
            static_cast<std::uint32_t>(rev[packed_bits[bi + 0u]]) |
            (static_cast<std::uint32_t>(rev[packed_bits[bi + 1u]]) << 8) |
            (static_cast<std::uint32_t>(rev[packed_bits[bi + 2u]]) << 16) |
            (static_cast<std::uint32_t>(rev[packed_bits[bi + 3u]]) << 24);
        packed_words[wi++] = w;
        bi += 4u;
    }

    if (bi < nbytes && wi < nwords) {
        std::uint32_t w = 0u;
        unsigned shift = 0u;
        while (bi < nbytes) {
            w |= (static_cast<std::uint32_t>(rev[packed_bits[bi++]]) << shift);
            shift += 8u;
        }
        packed_words[wi] = w;
    }

    const std::size_t tail = n & 31u;
    if (tail != 0u) {
        const std::size_t last_word = (n - 1u) >> 5;
        const std::uint32_t mask = (1u << tail) - 1u;
        packed_words[last_word] &= mask;
    }

    const std::size_t used_words = (n + 31u) >> 5;
    if (used_words < nwords) {
        std::fill(packed_words.begin() + used_words, packed_words.end(), 0u);
    }
}

}  // namespace tm_model::detail

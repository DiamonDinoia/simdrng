#include <random>
#include <cstring>

#include <catch2/catch_all.hpp>
#include <monocypher.h>

#include <simdrng/chacha.hpp>

static constexpr auto tests = 1 << 15;

TEST_CASE("ChaChaScalar", "[chacha]") {
    using ChaCha20 = simdrng::ChaCha<20>;

    auto seed = std::random_device{}();
    INFO("SEED: " << seed);
    std::mt19937 rng32(seed);
    std::mt19937_64 rng64(seed);
    ChaCha20::input_word counter = rng64(), nonce = rng64();
    std::array<ChaCha20::matrix_word, 8> key;
    for (int i = 0; i < 8; i++) {
        key[i] = rng32();
    }

    ChaCha20 rngChaCha(key, counter, nonce);
    for (int i = 0; i < tests; ++i) {
        // In a correct implementation, the internal state should neccsarily be
        // a validly arranged input for the algorithm and thus the reference impl.
        const auto input = rngChaCha.getState();
        uint8_t referenceOutput[64];

        // Assemble key (8 words -> 32 bytes) and nonce (2 words -> 8 bytes)
        uint8_t key[32];
        for (int k = 0; k < 8; ++k) {
            uint32_t w = input[4 + k];
            key[4*k + 0] = static_cast<uint8_t>(w & 0xFF);
            key[4*k + 1] = static_cast<uint8_t>((w >> 8) & 0xFF);
            key[4*k + 2] = static_cast<uint8_t>((w >> 16) & 0xFF);
            key[4*k + 3] = static_cast<uint8_t>((w >> 24) & 0xFF);
        }
        uint8_t nonce[8];
        uint32_t n0 = input[14];
        uint32_t n1 = input[15];
        nonce[0] = static_cast<uint8_t>(n0 & 0xFF);
        nonce[1] = static_cast<uint8_t>((n0 >> 8) & 0xFF);
        nonce[2] = static_cast<uint8_t>((n0 >> 16) & 0xFF);
        nonce[3] = static_cast<uint8_t>((n0 >> 24) & 0xFF);
        nonce[4] = static_cast<uint8_t>(n1 & 0xFF);
        nonce[5] = static_cast<uint8_t>((n1 >> 8) & 0xFF);
        nonce[6] = static_cast<uint8_t>((n1 >> 16) & 0xFF);
        nonce[7] = static_cast<uint8_t>((n1 >> 24) & 0xFF);
        uint64_t ctr = (static_cast<uint64_t>(input[13]) << 32) | static_cast<uint64_t>(input[12]);
        uint8_t zeros[64] = {0};
        // Use Monocypher's ChaCha20 DJB variant to produce the keystream block.
        crypto_chacha20_djb(referenceOutput, zeros, 64, key, nonce, ctr);

        const auto chachaOutput = rngChaCha.block();
        REQUIRE(std::memcmp(referenceOutput, chachaOutput.data(), 64) == 0);
    }
}

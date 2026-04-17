#include <random>

#include <catch2/catch_all.hpp>

#include <random/philox.hpp>

// Random123 Known Answer Test (KAT) vectors for Philox4x32-10
// Source: Random123 reference implementation (philox.h, kat_vectors)
TEST_CASE("Philox4x32-10 reference vectors", "[philox]") {
  using P = prng::Philox<4, 32, 10>;
  using ctr_t = P::counter_type;
  using key_t = P::key_type;

  auto check_block = [](key_t key, ctr_t ctr, ctr_t expected) {
    P rng(key, ctr);
    rng();
    auto output = std::bit_cast<ctr_t>(rng.result_cache());
    REQUIRE(output[0] == expected[0]);
    REQUIRE(output[1] == expected[1]);
    REQUIRE(output[2] == expected[2]);
    REQUIRE(output[3] == expected[3]);
  };

  SECTION("zeros") {
    check_block({0, 0}, {0, 0, 0, 0},
                {0x6627e8d5, 0xe169c58d, 0xbc57ac4c, 0x9b00dbd8});
  }

  SECTION("all ones") {
    check_block({0xffffffff, 0xffffffff},
                {0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff},
                {0x408f276d, 0x41c83b0e, 0xa20bc7c6, 0x6d5451fd});
  }

  SECTION("pi digits") {
    check_block({0xa4093822, 0x299f31d0},
                {0x243f6a88, 0x85a308d3, 0x13198a2e, 0x03707344},
                {0xd16cfe09, 0x94fdcceb, 0x5001e420, 0x24126ea1});
  }
}

// Random123 KAT vectors for Philox2x32-10
TEST_CASE("Philox2x32-10 reference vectors", "[philox]") {
  using P = prng::Philox<2, 32, 10>;
  using ctr_t = P::counter_type;
  using key_t = P::key_type;

  auto check_block = [](key_t key, ctr_t ctr, ctr_t expected) {
    P rng(key, ctr);
    rng();
    auto output = std::bit_cast<ctr_t>(rng.result_cache());
    REQUIRE(output[0] == expected[0]);
    REQUIRE(output[1] == expected[1]);
  };

  SECTION("zeros") {
    check_block({0}, {0, 0}, {0xff1dae59, 0x6cd10df2});
  }

  SECTION("all ones") {
    check_block({0xffffffff}, {0xffffffff, 0xffffffff},
                {0x2c3f628b, 0xab4fd7ad});
  }

  SECTION("pi digits") {
    check_block({0x13198a2e}, {0x243f6a88, 0x85a308d3},
                {0xdd7ce038, 0xf62a4c12});
  }
}

// Random123 KAT vectors for Philox4x64-10
TEST_CASE("Philox4x64-10 reference vectors", "[philox]") {
  using P = prng::Philox<4, 64, 10>;
  using ctr_t = P::counter_type;
  using key_t = P::key_type;

  SECTION("zeros") {
    P rng(key_t{0, 0}, ctr_t{0, 0, 0, 0});
    REQUIRE(rng() == 0x16554d9eca36314cULL);
    REQUIRE(rng() == 0xdb20fe9d672d0fdcULL);
    REQUIRE(rng() == 0xd7e772cee186176bULL);
    REQUIRE(rng() == 0x7e68b68aec7ba23bULL);
  }

  SECTION("all ones") {
    P rng(key_t{0xffffffffffffffffULL, 0xffffffffffffffffULL},
          ctr_t{0xffffffffffffffffULL, 0xffffffffffffffffULL,
                0xffffffffffffffffULL, 0xffffffffffffffffULL});
    REQUIRE(rng() == 0x87b092c3013fe90bULL);
    REQUIRE(rng() == 0x438c3c67be8d0224ULL);
    REQUIRE(rng() == 0x9cc7d7c69cd777b6ULL);
    REQUIRE(rng() == 0xa09caebf594f0ba0ULL);
  }
}

// Random123 KAT vectors for Philox2x64-10
TEST_CASE("Philox2x64-10 reference vectors", "[philox]") {
  using P = prng::Philox<2, 64, 10>;
  using ctr_t = P::counter_type;
  using key_t = P::key_type;

  SECTION("zeros") {
    P rng(key_t{0}, ctr_t{0, 0});
    REQUIRE(rng() == 0xca00a0459843d731ULL);
    REQUIRE(rng() == 0x66c24222c9a845b5ULL);
  }

  SECTION("all ones") {
    P rng(key_t{0xffffffffffffffffULL},
          ctr_t{0xffffffffffffffffULL, 0xffffffffffffffffULL});
    REQUIRE(rng() == 0x65b021d60cd8310fULL);
    REQUIRE(rng() == 0x4d02f3222f86df20ULL);
  }
}

// Verify sequential output: calling operator() returns elements from the same block
TEST_CASE("Philox4x32 sequential output matches block", "[philox]") {
  prng::Philox4x32 rng({0x12345678, 0x9abcdef0}, {0, 0, 0, 0});
  auto r0 = rng();
  auto r1 = rng();
  prng::Philox4x32 rng2({0x12345678, 0x9abcdef0}, {0, 0, 0, 0});
  rng2();
  auto cache = rng2.result_cache();
  REQUIRE(r0 == cache[0]);
  REQUIRE(r1 == cache[1]);
}

// Verify counter increment produces different output
TEST_CASE("Philox4x32 different counters produce different output", "[philox]") {
  prng::Philox4x32 rng1({0, 0}, {0, 0, 0, 0});
  prng::Philox4x32 rng2({0, 0}, {1, 0, 0, 0});
  REQUIRE(rng1() != rng2());
}

// Verify seed constructor produces deterministic output
TEST_CASE("Philox seed constructor is deterministic", "[philox]") {
  prng::Philox4x32 a(42);
  prng::Philox4x32 b(42);
  for (int i = 0; i < 100; ++i) {
    REQUIRE(a() == b());
  }
}

// Verify long sequence consistency across all four variants
TEMPLATE_TEST_CASE("Philox deterministic sequence", "[philox]",
                   prng::Philox4x32, prng::Philox2x32,
                   prng::Philox4x64, prng::Philox2x64) {
  auto seed = std::random_device{}();
  INFO("SEED: " << seed);
  TestType a(static_cast<uint64_t>(seed));
  TestType b(static_cast<uint64_t>(seed));
  for (int i = 0; i < 10000; ++i) {
    REQUIRE(a() == b());
  }
}

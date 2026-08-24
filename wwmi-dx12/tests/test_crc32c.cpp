// WWMI-DX12: CRC-32C unit tests.
//
// Vectors from RFC 3720 (iSCSI CRC-32C test vectors) and the standard
// CRC-32C check values. The bitwise reference implementation is deliberately
// independent of the production slicing-by-8 / SSE4.2 code paths.
#include "crc32c.hpp"
#include "test_framework.hpp"

#include <cstdint>
#include <cstring>

namespace
{

	// Naive bitwise CRC-32C (reflected poly 0x82F63B78, init/xorout ~seed),
	// matching crc32c_append's seed semantics.
	uint32_t ref_crc32c(uint32_t seed, const uint8_t *p, size_t n)
	{
		uint32_t crc = ~seed;
		for (size_t i = 0; i < n; ++i)
		{
			crc ^= p[i];
			for (int k = 0; k < 8; ++k)
				crc = (crc >> 1) ^ (0x82F63B78u & (0u - (crc & 1u)));
		}
		return ~crc;
	}

} // namespace

WWMI_TEST(crc32c_known_vectors)
{
	EXPECT_EQ(wwmi::crc32c_extend(0, "", 0), 0x00000000u);
	EXPECT_EQ(wwmi::crc32c_extend(0, "a", 1), 0xC1D04330u);
	EXPECT_EQ(wwmi::crc32c_extend(0, "abc", 3), 0x364B3FB7u);
	EXPECT_EQ(wwmi::crc32c_extend(0, "123456789", 9), 0xE3069283u);
	EXPECT_EQ(wwmi::crc32c_extend(0, "The quick brown fox jumps over the lazy dog", 43), 0x22620404u);

	uint8_t zeros[32] = {};
	uint8_t ones[32];
	std::memset(ones, 0xFF, sizeof(ones));
	uint8_t inc[32], dec[32];
	for (int i = 0; i < 32; ++i)
	{
		inc[i] = static_cast<uint8_t>(i);
		dec[i] = static_cast<uint8_t>(31 - i);
	}
	EXPECT_EQ(wwmi::crc32c_extend(0, zeros, 32), 0x8A9136AAu);
	EXPECT_EQ(wwmi::crc32c_extend(0, ones, 32), 0x62A8AB43u);
	EXPECT_EQ(wwmi::crc32c_extend(0, inc, 32), 0x46DD794Eu);
	EXPECT_EQ(wwmi::crc32c_extend(0, dec, 32), 0x113FDB5Cu);

	// Software fallback must produce identical results.
	EXPECT_EQ(wwmi::crc32c_extend_sw(0, "123456789", 9), 0xE3069283u);
	EXPECT_EQ(wwmi::crc32c_extend_sw(0, ones, 32), 0x62A8AB43u);
}

WWMI_TEST(crc32c_incremental_chaining)
{
	const char *full = "123456789";

	const uint32_t whole = wwmi::crc32c_extend(0, full, 9);

	uint32_t chained = wwmi::crc32c_extend(0, full, 4);
	chained = wwmi::crc32c_extend(chained, full + 4, 5);
	EXPECT_EQ(chained, whole);

	// Byte-at-a-time chaining.
	uint32_t byte_at_a_time = 0;
	for (size_t i = 0; i < 9; ++i)
		byte_at_a_time = wwmi::crc32c_extend(byte_at_a_time, full + i, 1);
	EXPECT_EQ(byte_at_a_time, whole);

	// Odd split point (1 + 8).
	uint32_t odd = wwmi::crc32c_extend(0, full, 1);
	odd = wwmi::crc32c_extend(odd, full + 1, 8);
	EXPECT_EQ(odd, whole);
}

WWMI_TEST(crc32c_matches_bitwise_reference)
{
	// Deterministic pseudo-random buffer; every length 0..64 and every
	// unaligned start offset 0..7 must match the reference implementation.
	uint8_t buf[64];
	uint32_t x = 0x12345678u;
	for (auto &b : buf)
	{
		x = x * 1664525u + 1013904223u;
		b = static_cast<uint8_t>(x >> 24);
	}

	for (size_t n = 0; n <= 64; ++n)
	{
		EXPECT_EQ(wwmi::crc32c_extend_sw(0, buf, n), ref_crc32c(0, buf, n));
		EXPECT_EQ(wwmi::crc32c_extend(0, buf, n), ref_crc32c(0, buf, n));
		if (wwmi::crc32c_sse42_supported())
			EXPECT_EQ(wwmi::crc32c_extend_hw(0, buf, n), ref_crc32c(0, buf, n));
	}

	for (size_t off = 0; off < 8; ++off)
	{
		const size_t n = 40 - off;
		EXPECT_EQ(wwmi::crc32c_extend_sw(0xDEADBEEFu, buf + off, n), ref_crc32c(0xDEADBEEFu, buf + off, n));
		EXPECT_EQ(wwmi::crc32c_extend(0xDEADBEEFu, buf + off, n), ref_crc32c(0xDEADBEEFu, buf + off, n));
		if (wwmi::crc32c_sse42_supported())
			EXPECT_EQ(wwmi::crc32c_extend_hw(0xDEADBEEFu, buf + off, n), ref_crc32c(0xDEADBEEFu, buf + off, n));
	}

	// Chained seeds must equal the reference's chained seeds.
	uint32_t a = wwmi::crc32c_extend(0, buf, 20);
	uint32_t b = wwmi::crc32c_extend(a, buf + 20, 20);
	EXPECT_EQ(b, ref_crc32c(ref_crc32c(0, buf, 20), buf + 20, 20));
	EXPECT_EQ(b, wwmi::crc32c_extend(0, buf, 40));
}

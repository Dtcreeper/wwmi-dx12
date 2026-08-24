// WWMI-DX12: CRC-32C implementation (SSE4.2 + slicing-by-8 software fallback).
//
// Semantics match crc32c-hw-1.0.5's crc32c_append() as used by 3DMigoto:
// both paths run the raw CRC over ~seed and XOR the result with 0xFFFFFFFF,
// which makes chained calls equivalent to hashing the concatenation.
#include "crc32c.hpp"

#include <algorithm>
#include <cstring>

#if defined(_M_X64) || defined(__x86_64__) || defined(_M_IX86) || defined(__i386__)
#define WWMI_X86 1
#include <intrin.h>
#if defined(_MSC_VER)
#include <nmmintrin.h>
#endif
#endif

namespace wwmi
{

	namespace
	{

		constexpr uint32_t kCrc32cPolyReflected = 0x82F63B78u; // CRC-32C, reflected

		// Slicing-by-8 tables (same construction as crc32c-hw's generated constants).
		struct Crc32cTables
		{
			uint32_t t[8][256];

			Crc32cTables()
			{
				for (uint32_t i = 0; i < 256; ++i)
				{
					uint32_t c = i;
					for (int k = 0; k < 8; ++k)
						c = (c & 1u) ? (kCrc32cPolyReflected ^ (c >> 1)) : (c >> 1);
					t[0][i] = c;
				}
				for (uint32_t i = 0; i < 256; ++i)
				{
					uint32_t c = t[0][i];
					for (int j = 1; j < 8; ++j)
					{
						c = t[0][c & 0xffu] ^ (c >> 8);
						t[j][i] = c;
					}
				}
			}
		};

		const Crc32cTables &crc32c_tables()
		{
			static const Crc32cTables tables;
			return tables;
		}

	} // namespace

	uint32_t crc32c_extend_sw(uint32_t seed, const void *data, size_t length)
	{
		const Crc32cTables &tables = crc32c_tables();
		const uint32_t(*t)[256] = tables.t;

		const uint8_t *next = static_cast<const uint8_t *>(data);
		uint64_t crc = static_cast<uint64_t>(seed) ^ 0xffffffffu;

		// Bring the pointer to an 8-byte boundary.
		while (length && (reinterpret_cast<uintptr_t>(next) & 7) != 0)
		{
			crc = t[0][(crc ^ *next++) & 0xffu] ^ (crc >> 8);
			--length;
		}

		while (length >= 8)
		{
			crc ^= *reinterpret_cast<const uint64_t *>(next);
			crc = t[7][crc & 0xffu]
				^ t[6][(crc >> 8) & 0xffu]
				^ t[5][(crc >> 16) & 0xffu]
				^ t[4][(crc >> 24) & 0xffu]
				^ t[3][(crc >> 32) & 0xffu]
				^ t[2][(crc >> 40) & 0xffu]
				^ t[1][(crc >> 48) & 0xffu]
				^ t[0][crc >> 56];
			next += 8;
			length -= 8;
		}

		while (length)
		{
			crc = t[0][(crc ^ *next++) & 0xffu] ^ (crc >> 8);
			--length;
		}

		return static_cast<uint32_t>(crc) ^ 0xffffffffu;
	}

#if WWMI_X86 && defined(_MSC_VER)

	uint32_t crc32c_extend_hw(uint32_t seed, const void *data, size_t length)
	{
		const uint8_t *next = static_cast<const uint8_t *>(data);
		uint64_t crc = static_cast<uint64_t>(seed) ^ 0xffffffffu;

		while (length >= 8)
		{
			crc = _mm_crc32_u64(static_cast<uint32_t>(crc), *reinterpret_cast<const uint64_t *>(next));
			next += 8;
			length -= 8;
		}
		if (length >= 4)
		{
			crc = _mm_crc32_u32(static_cast<uint32_t>(crc), *reinterpret_cast<const uint32_t *>(next));
			next += 4;
			length -= 4;
		}
		if (length >= 2)
		{
			crc = _mm_crc32_u16(static_cast<uint32_t>(crc), *reinterpret_cast<const uint16_t *>(next));
			next += 2;
			length -= 2;
		}
		if (length >= 1)
		{
			crc = _mm_crc32_u8(static_cast<uint32_t>(crc), *next);
		}

		return static_cast<uint32_t>(crc) ^ 0xffffffffu;
	}

	bool crc32c_sse42_supported()
	{
		static const bool supported = []() {
			int regs[4];
			__cpuid(regs, 1);
			return (regs[2] & (1 << 20)) != 0; // ECX bit 20: SSE4.2
		}();
		return supported;
	}

#else // Non-x86 or non-MSVC: no hardware path.

	uint32_t crc32c_extend_hw(uint32_t seed, const void *data, size_t length)
	{
		return crc32c_extend_sw(seed, data, length);
	}

	bool crc32c_sse42_supported()
	{
		return false;
	}

#endif

	uint32_t crc32c_extend(uint32_t seed, const void *data, size_t length)
	{
		if (crc32c_sse42_supported())
			return crc32c_extend_hw(seed, data, length);
		return crc32c_extend_sw(seed, data, length);
	}

} // namespace wwmi

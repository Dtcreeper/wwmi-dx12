#include "xxhash64.hpp"

#include <cstring>

namespace wwmi
{
	namespace
	{
		constexpr uint64_t PRIME1 = 0x9E3779B185EBCA87ULL;
		constexpr uint64_t PRIME2 = 0xC2B2AE3D27D4EB4FULL;
		constexpr uint64_t PRIME3 = 0x165667B19E3779F9ULL;
		constexpr uint64_t PRIME4 = 0x85EBCA77C2B2AE63ULL;
		constexpr uint64_t PRIME5 = 0x27D4EB2F165667C5ULL;

		inline uint64_t rotl64(uint64_t v, int s) { return (v << s) | (v >> (64 - s)); }
		inline uint64_t round64(uint64_t acc, uint64_t input)
		{
			acc += input * PRIME2;
			acc = rotl64(acc, 31);
			return acc * PRIME1;
		}
		inline uint64_t merge_round64(uint64_t acc, uint64_t val)
		{
			val = round64(0, val);
			acc ^= val;
			return acc * PRIME1 + PRIME4;
		}

		inline uint64_t read64(const uint8_t *p)
		{
			// XXH is little-endian; x64 target => direct unaligned load
			uint64_t v;
			std::memcpy(&v, p, 8);
			return v;
		}
		inline uint32_t read32(const uint8_t *p)
		{
			uint32_t v;
			std::memcpy(&v, p, 4);
			return v;
		}
	}

	uint64_t xxhash64(const void *data, size_t length, uint64_t seed)
	{
		const auto *p = static_cast<const uint8_t *>(data);
		const uint8_t *const end = p + length;

		uint64_t h;
		if (length >= 32)
		{
			uint64_t v1 = seed + PRIME1 + PRIME2;
			uint64_t v2 = seed + PRIME2;
			uint64_t v3 = seed;
			uint64_t v4 = seed - PRIME1;

			const uint8_t *const limit = end - 32;
			do
			{
				v1 = round64(v1, read64(p)); p += 8;
				v2 = round64(v2, read64(p)); p += 8;
				v3 = round64(v3, read64(p)); p += 8;
				v4 = round64(v4, read64(p)); p += 8;
			} while (p <= limit);

			h = rotl64(v1, 1) + rotl64(v2, 7) + rotl64(v3, 12) + rotl64(v4, 18);
			h = merge_round64(h, v1);
			h = merge_round64(h, v2);
			h = merge_round64(h, v3);
			h = merge_round64(h, v4);
		}
		else
		{
			h = seed + PRIME5;
		}

		h += static_cast<uint64_t>(length);

		while (p + 8 <= end)
		{
			h ^= round64(0, read64(p));
			h = rotl64(h, 27) * PRIME1 + PRIME4;
			p += 8;
		}
		if (p + 4 <= end)
		{
			h ^= static_cast<uint64_t>(read32(p)) * PRIME1;
			h = rotl64(h, 23) * PRIME2 + PRIME3;
			p += 4;
		}
		while (p < end)
		{
			h ^= (*p) * PRIME5;
			h = rotl64(h, 11) * PRIME1;
			++p;
		}

		h ^= h >> 33;
		h *= PRIME2;
		h ^= h >> 29;
		h *= PRIME3;
		h ^= h >> 32;
		return h;
	}
}

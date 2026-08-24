// WWMI-DX12: CRC-32C (Castagnoli).
//
// Drop-in equivalent of the crc32c-hw library 3DMigoto links against
// (crc32c_append): standard CRC-32C (poly 0x1EDC6F41 / reflected
// 0x82F63B78, init/xorout 0xFFFFFFFF) with incremental seeding:
//
//   crc32c_extend(0, data, n)          == CRC32C(data)
//   crc32c_extend(crc32c_extend(0,a),b) == CRC32C(a || b)
//
// SSE4.2 hardware path with a table-driven software fallback.
#pragma once

#include <cstddef>
#include <cstdint>

namespace wwmi
{

	uint32_t crc32c_extend(uint32_t seed, const void *data, size_t length);

	// True when the CPU supports the SSE4.2 CRC32 instruction.
	bool crc32c_sse42_supported();

	// Exposed for tests / diagnostics: force a specific implementation.
	uint32_t crc32c_extend_sw(uint32_t seed, const void *data, size_t length);
	uint32_t crc32c_extend_hw(uint32_t seed, const void *data, size_t length);

} // namespace wwmi

// WWMI-DX12: XXH64 (seed 0), software implementation.
//
// 3DMigoto hashes shader bytecode with XXH64 (src/d3d11/shader.cpp,
// XXHSUM64) and displays it as 16 hex digits -- e.g. the ShaderOverride
// 'hash = 525e619cd71fb4b0' entries WWMI mods carry. Texture hashes, by
// contrast, are CRC-32 (8 hex digits; see texture_hash.hpp).
//
// Self-verified against the canonical XXH64 test vectors:
//   xxhash64("", 0, 0)     == 0xEF46DB3751D8E999
//   xxhash64("a", 1, 0)    == 0xD24EC4F1A98C6E5B
//   xxhash64("abc", 3, 0)  == 0x44BC2CF5AD770999
#pragma once

#include <cstddef>
#include <cstdint>

namespace wwmi
{
	uint64_t xxhash64(const void *data, size_t length, uint64_t seed = 0);
}

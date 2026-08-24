// WWMI-DX12: 3DMigoto-compatible buffer (VB/IB) hashing (M2).
//
// 3DMigoto HackerDevice::CreateBuffer computes:
//
//   data_hash = crc32c(0, pInitialData->pSysMem, ByteWidth)  // 0 if absent
//   hash      = crc32c(data_hash, pDesc, sizeof(D3D11_BUFFER_DESC))
//
// The hash WWMI mods reference in TextureOverride sections is the full
// hash. UE's D3D11 RHI creates mesh VB/IB with initial data, so the
// data term is always present in the recorded hashes; the DX12 runtime
// must therefore learn buffer contents via GPU readback before hashing
// (BufferTracker, m2-4).
//
// DX12 differences handled here:
//  - D3D12_RESOURCE_DESC carries no D3D11 bind role, and ReShade's
//    resource_desc marks every usage; the role is only known when the
//    buffer is bound (bind_index_buffer / bind_vertex_buffers). The
//    caller passes it in.
//  - D3D12 Width (UINT64) maps to D3D11 ByteWidth (UINT32); mesh
//    buffers are far below 4GB.
//  - UE mesh buffers are D3D11_USAGE_DEFAULT with no CPU access, no
//    misc flags and no structured stride, so those desc fields are 0.
#pragma once

#include <cstddef>
#include <cstdint>

namespace wwmi
{
	// Buffer role discovered at bind time.
	enum class BufferRole : uint8_t
	{
		unknown,
		vertex, // D3D11_BIND_VERTEX_BUFFER
		index,  // D3D11_BIND_INDEX_BUFFER
	};

	// D3D11_BIND_* flag for a role (0 for unknown).
	uint32_t buffer_bind_flags(BufferRole role);

	// Byte-exact layout of D3D11_BUFFER_DESC: six UINTs, 24 bytes,
	// natural alignment, no padding. 3DMigoto hashes the raw struct
	// bytes, so the field order must not change.
	struct D3D11BufferDescBytes
	{
		uint32_t byte_width;            // D3D11_BUFFER_DESC::ByteWidth
		uint32_t usage;                 // D3D11_USAGE (DEFAULT = 0)
		uint32_t bind_flags;            // D3D11_BIND_*
		uint32_t cpu_access_flags;      // 0 for UE mesh buffers
		uint32_t misc_flags;            // 0
		uint32_t structure_byte_stride; // 0
	};
	static_assert(sizeof(D3D11BufferDescBytes) == 24, "D3D11_BUFFER_DESC layout");
	static_assert(offsetof(D3D11BufferDescBytes, byte_width) == 0, "D3D11_BUFFER_DESC layout");
	static_assert(offsetof(D3D11BufferDescBytes, usage) == 4, "D3D11_BUFFER_DESC layout");
	static_assert(offsetof(D3D11BufferDescBytes, bind_flags) == 8, "D3D11_BUFFER_DESC layout");
	static_assert(offsetof(D3D11BufferDescBytes, cpu_access_flags) == 12, "D3D11_BUFFER_DESC layout");
	static_assert(offsetof(D3D11BufferDescBytes, misc_flags) == 16, "D3D11_BUFFER_DESC layout");
	static_assert(offsetof(D3D11BufferDescBytes, structure_byte_stride) == 20, "D3D11_BUFFER_DESC layout");

	// Data term of the 3DMigoto buffer hash: CRC-32C over the buffer
	// contents (readback data), seeded from 0.
	uint32_t calc_buffer_data_hash(const void *data, size_t size);

	// Desc term: chains the 24 D3D11_BUFFER_DESC bytes onto seed.
	uint32_t calc_buffer_desc_hash(uint32_t seed, uint64_t byte_width, BufferRole role);

	// Full 3DMigoto buffer hash. Pass data_hash = 0 to reproduce the
	// no-initial-data path 3DMigoto uses for buffers created empty.
	uint32_t calc_buffer_hash(uint32_t data_hash, uint64_t byte_width, BufferRole role);
}

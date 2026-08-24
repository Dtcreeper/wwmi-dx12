// WWMI-DX12: 3DMigoto-compatible Texture2D hashing.
//
// Port of 3DMigoto's DirectX11/ResourceHash.cpp texture hash so hashes
// computed at runtime under DX12 match the hashes used by existing mods:
//
//   data_hash = calc_texture2d_data_hash(desc, data)   // first subresource only
//   hash      = calc_texture2d_desc_hash(data_hash, desc) // CRC over desc bytes
//
// Tex2DDesc is layout-identical to D3D11_TEXTURE2D_DESC (44 bytes) so the
// descriptor hash matches 3DMigoto byte-for-byte. The DirectXTK
// LoaderHelpers::GetSurfaceInfo / BitsPerPixel ports below are copied from
// 3DMigoto's copy of DirectXTK to keep the row pitch math identical.
#pragma once

#include <cstddef>
#include <cstdint>

namespace wwmi
{

	// Layout-identical to D3D11_TEXTURE2D_DESC. Field values are the
	// DXGI_FORMAT / D3D11_USAGE / D3D11_BIND_FLAG etc. enums (plain uint32_t
	// here so this header stays free of D3D includes).
	struct Tex2DDesc
	{
		uint32_t width;
		uint32_t height;
		uint32_t mip_levels;
		uint32_t array_size;
		uint32_t format;         // DXGI_FORMAT
		uint32_t sample_count;   // DXGI_SAMPLE_DESC.Count
		uint32_t sample_quality; // DXGI_SAMPLE_DESC.Quality
		uint32_t usage;          // D3D11_USAGE
		uint32_t bind_flags;     // D3D11_BIND_FLAG
		uint32_t cpu_access_flags;
		uint32_t misc_flags;
	};
	static_assert(sizeof(Tex2DDesc) == 44, "must match D3D11_TEXTURE2D_DESC layout");

	// Mirrors D3D11_SUBRESOURCE_DATA / D3D11_MAPPED_SUBRESOURCE.
	struct SubresourceData
	{
		const void *sys_mem = nullptr;
		uint32_t sys_mem_pitch = 0;
		uint32_t sys_mem_slice_pitch = 0;
	};

	// ---- DirectXTK LoaderHelpers ports ----

	// Bits per pixel for a DXGI format; 0 for unknown/typeless-without-size.
	size_t bits_per_pixel(uint32_t fmt);

	struct SurfaceInfo
	{
		size_t num_bytes = 0;
		size_t row_bytes = 0;
		size_t num_rows = 0;
	};
	bool get_surface_info(size_t width, size_t height, uint32_t fmt, SurfaceInfo *out);

	// 3DMigoto CompressedFormatBlockSize: bytes per 4x4 block; 0 = uncompressed.
	size_t compressed_format_block_size(uint32_t fmt);

	// 3DMigoto Texture2DLength: byte size of a 2D texture's first mip level.
	size_t texture2d_length(const Tex2DDesc &desc, uint32_t sys_mem_pitch, uint32_t level);

	// 3DMigoto hash_tex2d_data: row-wise hashing with optional padding handling
	// (zero_padding replaces row padding with zeroes, skip_padding drops it).
	uint32_t hash_tex2d_data(uint32_t hash, const void *data, size_t length,
		const Tex2DDesc &desc, bool zero_padding,
		bool skip_padding, uint32_t mapped_row_pitch);

	// 3DMigoto CalcTexture2DDataHash (default v1.2.1-compatible path with the
	// v1.2.11+ skip-padding fallback for buffers the v1.2.1 length overflows).
	uint32_t calc_texture2d_data_hash(const Tex2DDesc &desc, const SubresourceData &data,
		bool zero_padding = false);

	// 3DMigoto CalcTexture2DDescHash: CRC-32C over the desc struct bytes,
	// seeded with the data hash.
	uint32_t calc_texture2d_desc_hash(uint32_t initial_hash, const Tex2DDesc &desc);

	// Screen-resolution hash override port (3DMigoto AdjustForConstResolution).
	// Inactive by default, matching 3DMigoto's default configuration.
	struct HashResolutionOverride
	{
		bool active = false;
		uint32_t width = 0;
		uint32_t height = 0;
	};
	HashResolutionOverride &hash_resolution_override();

	// Readback compaction: D3D12 returns GPU texture data with a row pitch
	// aligned to D3D12_TEXTURE_DATA_PITCH_ALIGNMENT (256), while 3DMigoto's
	// hashes were computed over the game's upload layout (usually tight).
	// Copies the tight row_bytes of each of num_rows rows into dst and
	// returns the compacted byte count (0 on bad arguments / overflow).
	size_t compact_rows(const void *src, size_t src_row_pitch, size_t num_rows,
		size_t row_bytes, void *dst, size_t dst_capacity);

} // namespace wwmi

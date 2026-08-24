// WWMI-DX12: 3DMigoto-compatible Texture2D hashing (see texture_hash.hpp).
//
// Ported from 3DMigoto DirectX11/ResourceHash.cpp (BSD-2) and the DirectXTK
// LoaderHelpers code copied into it. Keep the quirks: the v1.2.1 byte-length
// (Width * Height * ArraySize with 32-bit wrap), first-subresource-only
// hashing, and the row/padding semantics of hash_tex2d_data.
#include "texture_hash.hpp"

#include "crc32c.hpp"

#include <dxgiformat.h>

#include <algorithm>
#include <cstring>
#include <vector>

namespace wwmi
{

	// ---- DirectXTK LoaderHelpers::BitsPerPixel port ----

	size_t bits_per_pixel(uint32_t fmt)
	{
		switch (static_cast<DXGI_FORMAT>(fmt))
		{
		case DXGI_FORMAT_R32G32B32A32_TYPELESS:
		case DXGI_FORMAT_R32G32B32A32_FLOAT:
		case DXGI_FORMAT_R32G32B32A32_UINT:
		case DXGI_FORMAT_R32G32B32A32_SINT:
			return 128;

		case DXGI_FORMAT_R32G32B32_TYPELESS:
		case DXGI_FORMAT_R32G32B32_FLOAT:
		case DXGI_FORMAT_R32G32B32_UINT:
		case DXGI_FORMAT_R32G32B32_SINT:
			return 96;

		case DXGI_FORMAT_R16G16B16A16_TYPELESS:
		case DXGI_FORMAT_R16G16B16A16_FLOAT:
		case DXGI_FORMAT_R16G16B16A16_UNORM:
		case DXGI_FORMAT_R16G16B16A16_UINT:
		case DXGI_FORMAT_R16G16B16A16_SNORM:
		case DXGI_FORMAT_R16G16B16A16_SINT:
		case DXGI_FORMAT_R32G32_TYPELESS:
		case DXGI_FORMAT_R32G32_FLOAT:
		case DXGI_FORMAT_R32G32_UINT:
		case DXGI_FORMAT_R32G32_SINT:
		case DXGI_FORMAT_R32G8X24_TYPELESS:
		case DXGI_FORMAT_D32_FLOAT_S8X24_UINT:
		case DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS:
		case DXGI_FORMAT_X32_TYPELESS_G8X24_UINT:
		case DXGI_FORMAT_Y416:
		case DXGI_FORMAT_Y210:
		case DXGI_FORMAT_Y216:
			return 64;

		case DXGI_FORMAT_R10G10B10A2_TYPELESS:
		case DXGI_FORMAT_R10G10B10A2_UNORM:
		case DXGI_FORMAT_R10G10B10A2_UINT:
		case DXGI_FORMAT_R11G11B10_FLOAT:
		case DXGI_FORMAT_R8G8B8A8_TYPELESS:
		case DXGI_FORMAT_R8G8B8A8_UNORM:
		case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB:
		case DXGI_FORMAT_R8G8B8A8_UINT:
		case DXGI_FORMAT_R8G8B8A8_SNORM:
		case DXGI_FORMAT_R8G8B8A8_SINT:
		case DXGI_FORMAT_R16G16_TYPELESS:
		case DXGI_FORMAT_R16G16_FLOAT:
		case DXGI_FORMAT_R16G16_UNORM:
		case DXGI_FORMAT_R16G16_UINT:
		case DXGI_FORMAT_R16G16_SNORM:
		case DXGI_FORMAT_R16G16_SINT:
		case DXGI_FORMAT_R32_TYPELESS:
		case DXGI_FORMAT_D32_FLOAT:
		case DXGI_FORMAT_R32_FLOAT:
		case DXGI_FORMAT_R32_UINT:
		case DXGI_FORMAT_R32_SINT:
		case DXGI_FORMAT_R24G8_TYPELESS:
		case DXGI_FORMAT_D24_UNORM_S8_UINT:
		case DXGI_FORMAT_R24_UNORM_X8_TYPELESS:
		case DXGI_FORMAT_X24_TYPELESS_G8_UINT:
		case DXGI_FORMAT_R9G9B9E5_SHAREDEXP:
		case DXGI_FORMAT_R8G8_B8G8_UNORM:
		case DXGI_FORMAT_G8R8_G8B8_UNORM:
		case DXGI_FORMAT_B8G8R8A8_UNORM:
		case DXGI_FORMAT_B8G8R8X8_UNORM:
		case DXGI_FORMAT_R10G10B10_XR_BIAS_A2_UNORM:
		case DXGI_FORMAT_B8G8R8A8_TYPELESS:
		case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB:
		case DXGI_FORMAT_B8G8R8X8_TYPELESS:
		case DXGI_FORMAT_B8G8R8X8_UNORM_SRGB:
		case DXGI_FORMAT_AYUV:
		case DXGI_FORMAT_Y410:
		case DXGI_FORMAT_YUY2:
			return 32;

		case DXGI_FORMAT_P010:
		case DXGI_FORMAT_P016:
		case DXGI_FORMAT_V408:
			return 24;

		case DXGI_FORMAT_R8G8_TYPELESS:
		case DXGI_FORMAT_R8G8_UNORM:
		case DXGI_FORMAT_R8G8_UINT:
		case DXGI_FORMAT_R8G8_SNORM:
		case DXGI_FORMAT_R8G8_SINT:
		case DXGI_FORMAT_R16_TYPELESS:
		case DXGI_FORMAT_R16_FLOAT:
		case DXGI_FORMAT_D16_UNORM:
		case DXGI_FORMAT_R16_UNORM:
		case DXGI_FORMAT_R16_UINT:
		case DXGI_FORMAT_R16_SNORM:
		case DXGI_FORMAT_R16_SINT:
		case DXGI_FORMAT_B5G6R5_UNORM:
		case DXGI_FORMAT_B5G5R5A1_UNORM:
		case DXGI_FORMAT_A8P8:
		case DXGI_FORMAT_B4G4R4A4_UNORM:
		case DXGI_FORMAT_P208:
		case DXGI_FORMAT_V208:
			return 16;

		case DXGI_FORMAT_NV12:
		case DXGI_FORMAT_420_OPAQUE:
		case DXGI_FORMAT_NV11:
			return 12;

		case DXGI_FORMAT_R8_TYPELESS:
		case DXGI_FORMAT_R8_UNORM:
		case DXGI_FORMAT_R8_UINT:
		case DXGI_FORMAT_R8_SNORM:
		case DXGI_FORMAT_R8_SINT:
		case DXGI_FORMAT_A8_UNORM:
		case DXGI_FORMAT_BC2_TYPELESS:
		case DXGI_FORMAT_BC2_UNORM:
		case DXGI_FORMAT_BC2_UNORM_SRGB:
		case DXGI_FORMAT_BC3_TYPELESS:
		case DXGI_FORMAT_BC3_UNORM:
		case DXGI_FORMAT_BC3_UNORM_SRGB:
		case DXGI_FORMAT_BC5_TYPELESS:
		case DXGI_FORMAT_BC5_UNORM:
		case DXGI_FORMAT_BC5_SNORM:
		case DXGI_FORMAT_BC6H_TYPELESS:
		case DXGI_FORMAT_BC6H_UF16:
		case DXGI_FORMAT_BC6H_SF16:
		case DXGI_FORMAT_BC7_TYPELESS:
		case DXGI_FORMAT_BC7_UNORM:
		case DXGI_FORMAT_BC7_UNORM_SRGB:
		case DXGI_FORMAT_AI44:
		case DXGI_FORMAT_IA44:
		case DXGI_FORMAT_P8:
			return 8;

		case DXGI_FORMAT_R1_UNORM:
			return 1;

		case DXGI_FORMAT_BC1_TYPELESS:
		case DXGI_FORMAT_BC1_UNORM:
		case DXGI_FORMAT_BC1_UNORM_SRGB:
		case DXGI_FORMAT_BC4_TYPELESS:
		case DXGI_FORMAT_BC4_UNORM:
		case DXGI_FORMAT_BC4_SNORM:
			return 4;

		case DXGI_FORMAT_UNKNOWN:
		case DXGI_FORMAT_FORCE_UINT:
		default:
			return 0;
		}
	}

	// ---- DirectXTK LoaderHelpers::GetSurfaceInfo port ----

	bool get_surface_info(size_t width, size_t height, uint32_t fmt, SurfaceInfo *out)
	{
		const DXGI_FORMAT f = static_cast<DXGI_FORMAT>(fmt);

		uint64_t num_bytes = 0;
		uint64_t row_bytes = 0;
		uint64_t num_rows = 0;

		bool bc = false;
		bool packed = false;
		bool planar = false;
		size_t bpe = 0;
		switch (f)
		{
		case DXGI_FORMAT_UNKNOWN:
			return false;

		case DXGI_FORMAT_BC1_TYPELESS:
		case DXGI_FORMAT_BC1_UNORM:
		case DXGI_FORMAT_BC1_UNORM_SRGB:
		case DXGI_FORMAT_BC4_TYPELESS:
		case DXGI_FORMAT_BC4_UNORM:
		case DXGI_FORMAT_BC4_SNORM:
			bc = true;
			bpe = 8;
			break;

		case DXGI_FORMAT_BC2_TYPELESS:
		case DXGI_FORMAT_BC2_UNORM:
		case DXGI_FORMAT_BC2_UNORM_SRGB:
		case DXGI_FORMAT_BC3_TYPELESS:
		case DXGI_FORMAT_BC3_UNORM:
		case DXGI_FORMAT_BC3_UNORM_SRGB:
		case DXGI_FORMAT_BC5_TYPELESS:
		case DXGI_FORMAT_BC5_UNORM:
		case DXGI_FORMAT_BC5_SNORM:
		case DXGI_FORMAT_BC6H_TYPELESS:
		case DXGI_FORMAT_BC6H_UF16:
		case DXGI_FORMAT_BC6H_SF16:
		case DXGI_FORMAT_BC7_TYPELESS:
		case DXGI_FORMAT_BC7_UNORM:
		case DXGI_FORMAT_BC7_UNORM_SRGB:
			bc = true;
			bpe = 16;
			break;

		case DXGI_FORMAT_R8G8_B8G8_UNORM:
		case DXGI_FORMAT_G8R8_G8B8_UNORM:
		case DXGI_FORMAT_YUY2:
			packed = true;
			bpe = 4;
			break;

		case DXGI_FORMAT_Y210:
		case DXGI_FORMAT_Y216:
			packed = true;
			bpe = 8;
			break;

		case DXGI_FORMAT_NV12:
		case DXGI_FORMAT_420_OPAQUE:
			if ((height % 2) != 0)
				return false; // requires a height alignment of 2
			planar = true;
			bpe = 2;
			break;

		case DXGI_FORMAT_P208:
			planar = true;
			bpe = 2;
			break;

		case DXGI_FORMAT_P010:
		case DXGI_FORMAT_P016:
			if ((height % 2) != 0)
				return false; // requires a height alignment of 2
			planar = true;
			bpe = 4;
			break;

		default:
			break;
		}

		if (bc)
		{
			uint64_t num_blocks_wide = 0;
			if (width > 0)
				num_blocks_wide = std::max<uint64_t>(1u, (static_cast<uint64_t>(width) + 3u) / 4u);
			uint64_t num_blocks_high = 0;
			if (height > 0)
				num_blocks_high = std::max<uint64_t>(1u, (static_cast<uint64_t>(height) + 3u) / 4u);
			row_bytes = num_blocks_wide * bpe;
			num_rows = num_blocks_high;
			num_bytes = row_bytes * num_blocks_high;
		}
		else if (packed)
		{
			row_bytes = ((static_cast<uint64_t>(width) + 1u) >> 1) * bpe;
			num_rows = static_cast<uint64_t>(height);
			num_bytes = row_bytes * height;
		}
		else if (f == DXGI_FORMAT_NV11)
		{
			row_bytes = ((static_cast<uint64_t>(width) + 3u) >> 2) * 4u;
			num_rows = static_cast<uint64_t>(height) * 2u; // Direct3D makes this simplifying assumption
			num_bytes = row_bytes * num_rows;
		}
		else if (planar)
		{
			row_bytes = ((static_cast<uint64_t>(width) + 1u) >> 1) * bpe;
			num_bytes = (row_bytes * static_cast<uint64_t>(height)) + ((row_bytes * static_cast<uint64_t>(height) + 1u) >> 1);
			num_rows = height + ((static_cast<uint64_t>(height) + 1u) >> 1);
		}
		else
		{
			const size_t bpp = bits_per_pixel(fmt);
			if (!bpp)
				return false;

			row_bytes = (static_cast<uint64_t>(width) * bpp + 7u) / 8u; // round up to nearest byte
			num_rows = static_cast<uint64_t>(height);
			num_bytes = row_bytes * height;
		}

		if (out)
		{
			out->num_bytes = static_cast<size_t>(num_bytes);
			out->row_bytes = static_cast<size_t>(row_bytes);
			out->num_rows = static_cast<size_t>(num_rows);
		}
		return true;
	}

	// ---- 3DMigoto ResourceHash.cpp ports ----

	size_t compressed_format_block_size(uint32_t fmt)
	{
		switch (static_cast<DXGI_FORMAT>(fmt))
		{
		case DXGI_FORMAT_BC1_TYPELESS:
		case DXGI_FORMAT_BC1_UNORM:
		case DXGI_FORMAT_BC1_UNORM_SRGB:
		case DXGI_FORMAT_BC4_TYPELESS:
		case DXGI_FORMAT_BC4_UNORM:
		case DXGI_FORMAT_BC4_SNORM:
			return 8;

		case DXGI_FORMAT_BC2_TYPELESS:
		case DXGI_FORMAT_BC2_UNORM:
		case DXGI_FORMAT_BC2_UNORM_SRGB:
		case DXGI_FORMAT_BC3_TYPELESS:
		case DXGI_FORMAT_BC3_UNORM:
		case DXGI_FORMAT_BC3_UNORM_SRGB:
		case DXGI_FORMAT_BC5_TYPELESS:
		case DXGI_FORMAT_BC5_UNORM:
		case DXGI_FORMAT_BC5_SNORM:
		case DXGI_FORMAT_BC6H_TYPELESS:
		case DXGI_FORMAT_BC6H_UF16:
		case DXGI_FORMAT_BC6H_SF16:
		case DXGI_FORMAT_BC7_TYPELESS:
		case DXGI_FORMAT_BC7_UNORM:
		case DXGI_FORMAT_BC7_UNORM_SRGB:
			return 16;

		default:
			return 0;
		}
	}

	size_t texture2d_length(const Tex2DDesc &desc, uint32_t sys_mem_pitch, uint32_t level)
	{
		// At the moment we are only using the first mip-map level, but this
		// should work if we wanted to use another:
		const uint32_t mip_width = std::max(desc.width >> level, 1u);
		const uint32_t mip_height = std::max(desc.height >> level, 1u);

		const size_t block_size = compressed_format_block_size(desc.format);

		if (!block_size)
		{
			// Uncompressed texture - use the SysMemPitch to get the width
			// (including any padding) in bytes.
			return static_cast<size_t>(sys_mem_pitch) * mip_height;
		}

		// Compressed textures: can't rely on SysMemPitch ("lines" are
		// meaningless before decompression). Use the mip width + height padded
		// to a multiple of 4 with the 4x4 block size.
		const uint32_t padded_width = (mip_width + 3u) & ~0x3u;
		const uint32_t padded_height = (mip_height + 3u) & ~0x3u;

		return static_cast<size_t>(padded_width) * padded_height / 16 * block_size;
	}

	uint32_t hash_tex2d_data(uint32_t hash, const void *data, size_t length,
		const Tex2DDesc &desc, bool zero_padding,
		bool skip_padding, uint32_t mapped_row_pitch)
	{
		// Each row in a 2D texture has some alignment constraint, and the
		// unused bytes at the end of each row can be garbage, interfering with
		// the hash calculation. See 3DMigoto's notes in ResourceHash.cpp for
		// the history: zero_padding replaces the padding with zeroes,
		// skip_padding leaves it out of the hash entirely.

		if (!zero_padding && !skip_padding)
			return crc32c_extend(hash, data, length);

		SurfaceInfo info;
		if (!get_surface_info(desc.width, desc.height, desc.format, &info))
		{
			// 3DMigoto would proceed with uninitialized pitch values here;
			// fall back to hashing the raw buffer instead of UB.
			return crc32c_extend(hash, data, length);
		}

		const uint8_t *sptr = static_cast<const uint8_t *>(data);
		const size_t msize = std::min(info.row_bytes, static_cast<size_t>(mapped_row_pitch));

		const int64_t padding = static_cast<int64_t>(mapped_row_pitch) - static_cast<int64_t>(info.row_bytes);
		std::vector<uint8_t> zeroes;
		const uint8_t *zero_ptr = nullptr;
		if (zero_padding && padding > 0)
		{
			zeroes.assign(static_cast<size_t>(padding), 0);
			zero_ptr = zeroes.data();
		}

		int64_t remaining = static_cast<int64_t>(length);
		for (size_t h = 0; h < info.num_rows && remaining > 0; h++)
		{
			// 3DMigoto: min(msize, (unsigned)remaining)
			const uint64_t capped = static_cast<uint64_t>(static_cast<uint32_t>(remaining));
			const size_t chunk = static_cast<uint64_t>(msize) < capped ? msize : static_cast<size_t>(capped);
			hash = crc32c_extend(hash, sptr, chunk);
			sptr += mapped_row_pitch;
			remaining -= static_cast<int64_t>(msize);

			if (zero_ptr && remaining > 0)
			{
				const size_t zchunk = static_cast<size_t>(std::min(padding, remaining));
				hash = crc32c_extend(hash, zero_ptr, zchunk);
				remaining -= padding;
			}
		}

		return hash;
	}

	uint32_t calc_texture2d_data_hash(const Tex2DDesc &desc, const SubresourceData &data,
		bool zero_padding)
	{
		if (data.sys_mem == nullptr)
			return 0;

		// 3DMigoto v1.2.1 "length": Width * Height * ArraySize as a 32-bit
		// product (texels misinterpreted as bytes; wraps on overflow). Only
		// the first subresource participates in the hash (3DMigoto >= 1.2.11
		// semantics).
		const uint32_t length_v12 = desc.width * desc.height * desc.array_size;
		const size_t length = texture2d_length(desc, data.sys_mem_pitch, 0);

		if (static_cast<size_t>(length_v12) <= length)
		{
			// v1.2.1 compatible path: hash the first length_v12 bytes as-is.
			return hash_tex2d_data(0, data.sys_mem, length_v12, desc,
				zero_padding, false, data.sys_mem_pitch);
		}

		// The v1.2.1 length overflowed the buffer: hash the real first-mip
		// length with row padding skipped.
		return hash_tex2d_data(0, data.sys_mem, length, desc,
			false, true, data.sys_mem_pitch);
	}

	namespace
	{

		// 3DMigoto AdjustForConstResolution: replaces well-known screen
		// resolution dimensions with constant tags so hashes stay stable
		// across resolution changes. Tags are the multi-char constants
		// 'SRES' / 'SR*2' / 'SR*4' / 'SR*8' / 'SR/2'.
		void adjust_for_const_resolution(uint32_t *hash_width, uint32_t *hash_height)
		{
			const HashResolutionOverride &r = hash_resolution_override();
			if (!r.active)
				return;

			if (*hash_width == r.width && *hash_height == r.height)
			{
				*hash_width = 0x53524553u;  // 'SRES'
				*hash_height = 0x53524553u;
			}
			else if (*hash_width == r.width * 2 && *hash_height == r.height * 2)
			{
				*hash_width = 0x53522A32u;  // 'SR*2'
				*hash_height = 0x53522A32u;
			}
			else if (*hash_width == r.width * 4 && *hash_height == r.height * 4)
			{
				*hash_width = 0x53522A34u;  // 'SR*4'
				*hash_height = 0x53522A34u;
			}
			else if (*hash_width == r.width * 8 && *hash_height == r.height * 8)
			{
				*hash_width = 0x53522A38u;  // 'SR*8'
				*hash_height = 0x53522A38u;
			}
			else if (*hash_width == r.width / 2 && *hash_height == r.height / 2)
			{
				*hash_width = 0x53522F32u;  // 'SR/2'
				*hash_height = 0x53522F32u;
			}
		}

	} // namespace

	uint32_t calc_texture2d_desc_hash(uint32_t initial_hash, const Tex2DDesc &desc_in)
	{
		Tex2DDesc desc = desc_in;
		adjust_for_const_resolution(&desc.width, &desc.height);
		return crc32c_extend(initial_hash, &desc, sizeof(desc));
	}

	HashResolutionOverride &hash_resolution_override()
	{
		static HashResolutionOverride override;
		return override;
	}

	size_t compact_rows(const void *src, size_t src_row_pitch, size_t num_rows,
		size_t row_bytes, void *dst, size_t dst_capacity)
	{
		if (!src || !dst || !num_rows)
			return 0;
		if (row_bytes > src_row_pitch)
			return 0;
		if (num_rows > dst_capacity / (row_bytes ? row_bytes : 1))
			return 0;

		const uint8_t *s = static_cast<const uint8_t *>(src);
		uint8_t *d = static_cast<uint8_t *>(dst);
		for (size_t r = 0; r < num_rows; ++r)
		{
			std::memcpy(d, s, row_bytes);
			s += src_row_pitch;
			d += row_bytes;
		}
		return num_rows * row_bytes;
	}

} // namespace wwmi

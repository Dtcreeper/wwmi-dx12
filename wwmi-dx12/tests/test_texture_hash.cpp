// WWMI-DX12: 3DMigoto texture hash port tests.
//
// Expected values are built with an independent bitwise CRC-32C and manual
// byte layouts so the tests validate the ported algorithm structure (v1.2.1
// length quirk, padding semantics, desc hash composition), not just that the
// code agrees with itself.
#include "crc32c.hpp"
#include "test_framework.hpp"
#include "texture_hash.hpp"

#include <dxgiformat.h>

#include <cstdint>
#include <cstring>
#include <vector>

namespace
{

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

	wwmi::Tex2DDesc rgba8_desc(uint32_t w, uint32_t h)
	{
		return wwmi::Tex2DDesc{
			w, h, 1, 1,
			DXGI_FORMAT_R8G8B8A8_UNORM,
			1, 0,                      // sample count/quality
			0,                         // D3D11_USAGE_DEFAULT
			0x8,                       // D3D11_BIND_SHADER_RESOURCE
			0, 0,
		};
	}

	wwmi::Tex2DDesc bc_desc(uint32_t w, uint32_t h, uint32_t fmt)
	{
		return wwmi::Tex2DDesc{
			w, h, 1, 1,
			fmt,
			1, 0,
			0,
			0x8,
			0, 0,
		};
	}

	// Serialize a Tex2DDesc exactly as calc_texture2d_desc_hash sees it.
	std::vector<uint8_t> desc_bytes(const wwmi::Tex2DDesc &d)
	{
		std::vector<uint8_t> b(sizeof(d));
		std::memcpy(b.data(), &d, sizeof(d));
		return b;
	}

	std::vector<uint8_t> pattern(size_t n, uint8_t seed)
	{
		std::vector<uint8_t> v(n);
		uint32_t x = seed * 2654435761u + 1u;
		for (auto &b : v)
		{
			x = x * 1664525u + 1013904223u;
			b = static_cast<uint8_t>(x >> 24);
		}
		return v;
	}

} // namespace

// ---- DirectXTK ports ----

WWMI_TEST(bits_per_pixel_common_formats)
{
	EXPECT_EQ(wwmi::bits_per_pixel(DXGI_FORMAT_R8G8B8A8_UNORM), 32u);
	EXPECT_EQ(wwmi::bits_per_pixel(DXGI_FORMAT_R8G8B8A8_UNORM_SRGB), 32u);
	EXPECT_EQ(wwmi::bits_per_pixel(DXGI_FORMAT_B8G8R8A8_UNORM), 32u);
	EXPECT_EQ(wwmi::bits_per_pixel(DXGI_FORMAT_R16G16B16A16_FLOAT), 64u);
	EXPECT_EQ(wwmi::bits_per_pixel(DXGI_FORMAT_R32G32B32A32_FLOAT), 128u);
	EXPECT_EQ(wwmi::bits_per_pixel(DXGI_FORMAT_R10G10B10A2_UNORM), 32u);
	EXPECT_EQ(wwmi::bits_per_pixel(DXGI_FORMAT_R11G11B10_FLOAT), 32u);
	EXPECT_EQ(wwmi::bits_per_pixel(DXGI_FORMAT_R16G16_FLOAT), 32u);
	EXPECT_EQ(wwmi::bits_per_pixel(DXGI_FORMAT_R8_UNORM), 8u);
	EXPECT_EQ(wwmi::bits_per_pixel(DXGI_FORMAT_R16_FLOAT), 16u);
	EXPECT_EQ(wwmi::bits_per_pixel(DXGI_FORMAT_R9G9B9E5_SHAREDEXP), 32u);
	// BC formats report their *per-pixel average* bpp in DirectXTK:
	EXPECT_EQ(wwmi::bits_per_pixel(DXGI_FORMAT_BC1_UNORM), 4u);
	EXPECT_EQ(wwmi::bits_per_pixel(DXGI_FORMAT_BC4_UNORM), 4u);
	EXPECT_EQ(wwmi::bits_per_pixel(DXGI_FORMAT_BC3_UNORM), 8u);
	EXPECT_EQ(wwmi::bits_per_pixel(DXGI_FORMAT_BC5_UNORM), 8u);
	EXPECT_EQ(wwmi::bits_per_pixel(DXGI_FORMAT_BC7_UNORM), 8u);
	EXPECT_EQ(wwmi::bits_per_pixel(DXGI_FORMAT_UNKNOWN), 0u);
}

WWMI_TEST(surface_info_uncompressed)
{
	wwmi::SurfaceInfo si{};

	EXPECT(wwmi::get_surface_info(8, 8, DXGI_FORMAT_R8G8B8A8_UNORM, &si));
	EXPECT_EQ(si.row_bytes, 32u);
	EXPECT_EQ(si.num_rows, 8u);
	EXPECT_EQ(si.num_bytes, 256u);

	// 9x9: row bytes round up per-row, not per-texture.
	EXPECT(wwmi::get_surface_info(9, 9, DXGI_FORMAT_R8G8B8A8_UNORM, &si));
	EXPECT_EQ(si.row_bytes, 36u);
	EXPECT_EQ(si.num_rows, 9u);
	EXPECT_EQ(si.num_bytes, 324u);

	// 1x1 RGBA8 = 4 bytes.
	EXPECT(wwmi::get_surface_info(1, 1, DXGI_FORMAT_R8G8B8A8_UNORM, &si));
	EXPECT_EQ(si.row_bytes, 4u);
	EXPECT_EQ(si.num_rows, 1u);
	EXPECT_EQ(si.num_bytes, 4u);

	EXPECT(!wwmi::get_surface_info(8, 8, DXGI_FORMAT_UNKNOWN, &si));
}

WWMI_TEST(surface_info_block_compressed)
{
	wwmi::SurfaceInfo si{};

	// 8x8 BC1: 2x2 blocks of 8 bytes.
	EXPECT(wwmi::get_surface_info(8, 8, DXGI_FORMAT_BC1_UNORM, &si));
	EXPECT_EQ(si.row_bytes, 16u);
	EXPECT_EQ(si.num_rows, 2u);
	EXPECT_EQ(si.num_bytes, 32u);

	// 9x9 BC1: blocks round up to 3x3.
	EXPECT(wwmi::get_surface_info(9, 9, DXGI_FORMAT_BC1_UNORM, &si));
	EXPECT_EQ(si.row_bytes, 24u);
	EXPECT_EQ(si.num_rows, 3u);
	EXPECT_EQ(si.num_bytes, 72u);

	// 4x4 BC7: single 16-byte block.
	EXPECT(wwmi::get_surface_info(4, 4, DXGI_FORMAT_BC7_UNORM, &si));
	EXPECT_EQ(si.row_bytes, 16u);
	EXPECT_EQ(si.num_rows, 1u);
	EXPECT_EQ(si.num_bytes, 16u);

	// 1024x1024 BC7 (typical mod texture): 256 block rows x 4096 bytes.
	EXPECT(wwmi::get_surface_info(1024, 1024, DXGI_FORMAT_BC7_UNORM, &si));
	EXPECT_EQ(si.row_bytes, 4096u);
	EXPECT_EQ(si.num_rows, 256u);
	EXPECT_EQ(si.num_bytes, 1048576u);
}

// ---- 3DMigoto ports ----

WWMI_TEST(texture2d_length_paths)
{
	// Uncompressed: SysMemPitch * height.
	EXPECT_EQ(wwmi::texture2d_length(rgba8_desc(8, 8), 32, 0), 256u);
	EXPECT_EQ(wwmi::texture2d_length(rgba8_desc(16, 16), 64, 0), 1024u);

	// BC1: 8x8 -> 32 bytes, 9x9 -> pads to 12x12 texels -> 72 bytes.
	EXPECT_EQ(wwmi::texture2d_length(bc_desc(8, 8, DXGI_FORMAT_BC1_UNORM), 16, 0), 32u);
	EXPECT_EQ(wwmi::texture2d_length(bc_desc(9, 9, DXGI_FORMAT_BC1_UNORM), 16, 0), 72u);

	// BC7: 4x4 -> 16 bytes.
	EXPECT_EQ(wwmi::texture2d_length(bc_desc(4, 4, DXGI_FORMAT_BC7_UNORM), 16, 0), 16u);

	// BC3 9x9 -> 144 bytes.
	EXPECT_EQ(wwmi::texture2d_length(bc_desc(9, 9, DXGI_FORMAT_BC3_UNORM), 16, 0), 144u);
}

WWMI_TEST(data_hash_v121_compat_path_hashes_v12_length_only)
{
	// 16x16 RGBA8, real buffer 1024 bytes. The v1.2.1 length is
	// 16*16*1 = 256 bytes, so only the first 256 bytes get hashed.
	const std::vector<uint8_t> data = pattern(1024, 7);

	wwmi::SubresourceData sd;
	sd.sys_mem = data.data();
	sd.sys_mem_pitch = 64;

	const uint32_t h = wwmi::calc_texture2d_data_hash(rgba8_desc(16, 16), sd);
	EXPECT_EQ(h, ref_crc32c(0, data.data(), 256));

	// Bytes past the v1.2.1 length must not affect the hash.
	std::vector<uint8_t> mutated = data;
	std::memset(mutated.data() + 256, 0xAB, 1024 - 256);
	sd.sys_mem = mutated.data();
	EXPECT_EQ(wwmi::calc_texture2d_data_hash(rgba8_desc(16, 16), sd), h);
}

WWMI_TEST(data_hash_bc1_takes_v1211_skip_padding_path)
{
	// BC1 is 4bpp, so Width*Height (v1.2.1 "length" = 64) overflows the real
	// buffer (32 bytes) -> the v1.2.11+ path hashes the real length with
	// skip_padding. With SysMemPitch equal to the block row pitch there is no
	// padding to skip, so this is a plain CRC over the 32 bytes.
	const std::vector<uint8_t> data = pattern(32, 3);

	wwmi::SubresourceData sd;
	sd.sys_mem = data.data();
	sd.sys_mem_pitch = 16; // 2 blocks * 8 bytes

	const uint32_t h = wwmi::calc_texture2d_data_hash(bc_desc(8, 8, DXGI_FORMAT_BC1_UNORM), sd);
	EXPECT_EQ(h, ref_crc32c(0, data.data(), 32));
}

WWMI_TEST(data_hash_bc7_equal_lengths_use_compat_path)
{
	// BC7 is 8bpp: for multiples of 4, Width*Height == real byte size, so the
	// v1.2.1 path is taken with the full buffer hashed.
	const std::vector<uint8_t> data = pattern(64, 5);

	wwmi::SubresourceData sd;
	sd.sys_mem = data.data();
	sd.sys_mem_pitch = 32; // 2 blocks * 16 bytes

	const uint32_t h = wwmi::calc_texture2d_data_hash(bc_desc(8, 8, DXGI_FORMAT_BC7_UNORM), sd);
	EXPECT_EQ(h, ref_crc32c(0, data.data(), 64));
}

WWMI_TEST(data_hash_null_data_returns_zero)
{
	wwmi::SubresourceData sd;
	sd.sys_mem = nullptr;
	EXPECT_EQ(wwmi::calc_texture2d_data_hash(rgba8_desc(8, 8), sd), 0u);
}

WWMI_TEST(hash_tex2d_data_skip_padding_drops_row_padding)
{
	// 8x8 RGBA8 with a mapped row pitch of 40 (8 bytes of alignment padding
	// per row). skip_padding must hash the 32 real bytes of each row only.
	const size_t row_bytes = 32;
	const size_t mapped_pitch = 40;
	const size_t rows = 8;
	std::vector<uint8_t> data(mapped_pitch * rows);
	std::memcpy(data.data(), pattern(row_bytes * rows, 11).data(), row_bytes * rows);

	wwmi::Tex2DDesc desc = rgba8_desc(8, 8);
	const uint32_t h = wwmi::hash_tex2d_data(0, data.data(), row_bytes * rows,
		desc, false, true, static_cast<uint32_t>(mapped_pitch));

	// Expected: chain the CRC over each 32-byte row.
	uint32_t expected = 0;
	for (size_t r = 0; r < rows; ++r)
		expected = ref_crc32c(expected, data.data() + r * mapped_pitch, row_bytes);
	EXPECT_EQ(h, expected);

	// Padding bytes must not affect the hash.
	std::vector<uint8_t> mutated = data;
	for (size_t r = 0; r < rows; ++r)
		std::memset(mutated.data() + r * mapped_pitch + row_bytes, 0xCD, mapped_pitch - row_bytes);
	const uint32_t h2 = wwmi::hash_tex2d_data(0, mutated.data(), row_bytes * rows,
		desc, false, true, static_cast<uint32_t>(mapped_pitch));
	EXPECT_EQ(h2, h);
}

WWMI_TEST(hash_tex2d_data_zero_padding_hashes_zeroes)
{
	// Same layout, zero_padding=true: each row is hashed as 32 real bytes
	// followed by 8 zero bytes, so the hash equals a CRC over the packed
	// [row || 8 zeroes] buffer.
	const size_t row_bytes = 32;
	const size_t mapped_pitch = 40;
	const size_t rows = 8;
	std::vector<uint8_t> data(mapped_pitch * rows);
	std::memcpy(data.data(), pattern(row_bytes * rows, 13).data(), row_bytes * rows);

	wwmi::Tex2DDesc desc = rgba8_desc(8, 8);
	const uint32_t h = wwmi::hash_tex2d_data(0, data.data(), row_bytes * rows,
		desc, true, false, static_cast<uint32_t>(mapped_pitch));

	std::vector<uint8_t> packed((row_bytes + 8) * rows);
	for (size_t r = 0; r < rows; ++r)
	{
		std::memcpy(packed.data() + r * (row_bytes + 8), data.data() + r * mapped_pitch, row_bytes);
		// remaining 8 bytes of each packed row stay zero
	}
	// The length budget (row_bytes * rows = 256) is spent on padding too, so
	// the hash covers 6 full padded rows (240 bytes) plus the first 16 bytes
	// of row 6 before running out.
	EXPECT_EQ(h, ref_crc32c(0, packed.data(), row_bytes * rows));
}

WWMI_TEST(hash_tex2d_data_length_budget_quirk)
{
	// 3DMigoto's hash_tex2d_data spends the *length* budget on padding bytes
	// too: with length=64 (two rows), it hashes row0[0:32], 8 zero padding
	// bytes, then only row1[0:24] before the budget runs out. The port must
	// reproduce this exactly.
	const size_t row_bytes = 32;
	const size_t mapped_pitch = 40;
	std::vector<uint8_t> data(mapped_pitch * 8);
	std::memcpy(data.data(), pattern(row_bytes * 2, 17).data(), row_bytes * 2);

	wwmi::Tex2DDesc desc = rgba8_desc(8, 8);
	const uint32_t h = wwmi::hash_tex2d_data(0, data.data(), 64,
		desc, true, false, static_cast<uint32_t>(mapped_pitch));

	std::vector<uint8_t> expected;
	expected.insert(expected.end(), data.begin(), data.begin() + 32);        // row0
	expected.insert(expected.end(), 8, 0);                                    // padding
	expected.insert(expected.end(), data.begin() + mapped_pitch, data.begin() + mapped_pitch + 24); // row1[0:24]
	EXPECT_EQ(expected.size(), 64u);
	EXPECT_EQ(h, ref_crc32c(0, expected.data(), expected.size()));
}

WWMI_TEST(desc_hash_composes_over_desc_bytes)
{
	// Full pipeline: data hash then desc hash must equal the manual
	// composition over the 44-byte desc layout.
	const std::vector<uint8_t> data = pattern(16, 23);
	wwmi::Tex2DDesc desc = rgba8_desc(4, 4);

	wwmi::SubresourceData sd;
	sd.sys_mem = data.data();
	sd.sys_mem_pitch = 16;

	const uint32_t data_hash = wwmi::calc_texture2d_data_hash(desc, sd);
	EXPECT_EQ(data_hash, ref_crc32c(0, data.data(), 16));

	const uint32_t full = wwmi::calc_texture2d_desc_hash(data_hash, desc);
	const std::vector<uint8_t> bytes = desc_bytes(desc);
	EXPECT_EQ(full, wwmi::crc32c_extend(data_hash, bytes.data(), 44));

	// Chaining property: desc hash == CRC over data || desc concatenated.
	std::vector<uint8_t> chained;
	chained.insert(chained.end(), data.begin(), data.end());
	chained.insert(chained.end(), bytes.begin(), bytes.end());
	EXPECT_EQ(full, ref_crc32c(0, chained.data(), chained.size()));

	// Any desc field change must change the hash.
	wwmi::Tex2DDesc other = desc;
	other.array_size = 2;
	EXPECT_NE(wwmi::calc_texture2d_desc_hash(data_hash, other), full);
	other = desc;
	other.format = DXGI_FORMAT_R8G8B8A8_UNORM_SRGB;
	EXPECT_NE(wwmi::calc_texture2d_desc_hash(data_hash, other), full);
	other = desc;
	other.sample_count = 4;
	EXPECT_NE(wwmi::calc_texture2d_desc_hash(data_hash, other), full);
}

WWMI_TEST(desc_hash_resolution_override)
{
	wwmi::HashResolutionOverride &ro = wwmi::hash_resolution_override();
	ro.active = false;

	const uint32_t data_hash = 0x11223344u;
	wwmi::Tex2DDesc desc = rgba8_desc(1920, 1080);
	const uint32_t plain = wwmi::calc_texture2d_desc_hash(data_hash, desc);

	// Override active with matching screen dimensions -> 'SRES' tags.
	ro.active = true;
	ro.width = 1920;
	ro.height = 1080;

	wwmi::Tex2DDesc tagged = desc;
	tagged.width = 0x53524553u;  // 'SRES'
	tagged.height = 0x53524553u;
	EXPECT_EQ(wwmi::calc_texture2d_desc_hash(data_hash, desc),
		ref_crc32c(data_hash, desc_bytes(tagged).data(), 44));
	EXPECT_NE(wwmi::calc_texture2d_desc_hash(data_hash, desc), plain);

	// 2x screen resolution -> 'SR*2'.
	wwmi::Tex2DDesc big = rgba8_desc(3840, 2160);
	wwmi::Tex2DDesc big_tagged = big;
	big_tagged.width = 0x53522A32u;  // 'SR*2'
	big_tagged.height = 0x53522A32u;
	EXPECT_EQ(wwmi::calc_texture2d_desc_hash(data_hash, big),
		ref_crc32c(data_hash, desc_bytes(big_tagged).data(), 44));

	// Non-matching dimensions are untouched.
	wwmi::Tex2DDesc small = rgba8_desc(512, 512);
	EXPECT_EQ(wwmi::calc_texture2d_desc_hash(data_hash, small),
		ref_crc32c(data_hash, desc_bytes(small).data(), 44));

	ro.active = false; // restore default for other tests
}

WWMI_TEST(hash_pipeline_bc7_mod_texture)
{
	// End-to-end sanity on a typical mod texture layout: 256x256 BC7 with a
	// matching block row pitch. BC7 is 8bpp so v1.2.1 length == real length
	// and the full 64KiB gets hashed, then the desc hash composes on top.
	const size_t size = 256 / 4 * 256 / 4 * 16; // 65536 bytes
	const std::vector<uint8_t> data = pattern(size, 29);

	wwmi::Tex2DDesc desc = bc_desc(256, 256, DXGI_FORMAT_BC7_UNORM);
	wwmi::SubresourceData sd;
	sd.sys_mem = data.data();
	sd.sys_mem_pitch = 256 / 4 * 16; // 1024

	const uint32_t data_hash = wwmi::calc_texture2d_data_hash(desc, sd);
	EXPECT_EQ(data_hash, ref_crc32c(0, data.data(), size));

	const uint32_t full = wwmi::calc_texture2d_desc_hash(data_hash, desc);
	const std::vector<uint8_t> bytes = desc_bytes(desc);
	EXPECT_EQ(full, ref_crc32c(data_hash, bytes.data(), 44));
}

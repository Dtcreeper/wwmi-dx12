// WWMI-DX12: DDS loader unit tests.
//
// DDS blobs are constructed byte-by-byte so the tests validate real file
// layouts: legacy FourCC headers, legacy RGBA mask headers, DX10 extended
// headers, mip chains, cubemaps and arrays.
#include "dds_loader.hpp"
#include "test_framework.hpp"

#include <dxgiformat.h>

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace
{

	constexpr uint32_t DDSD_CAPS = 0x1;
	constexpr uint32_t DDSD_HEIGHT = 0x2;
	constexpr uint32_t DDSD_WIDTH = 0x4;
	constexpr uint32_t DDSD_PIXELFORMAT = 0x1000;
	constexpr uint32_t DDSD_MIPMAPCOUNT = 0x20000;
	constexpr uint32_t DDSD_LINEARSIZE = 0x80000;

	constexpr uint32_t DDPF_ALPHAPIXELS = 0x1;
	constexpr uint32_t DDPF_FOURCC = 0x4;
	constexpr uint32_t DDPF_RGB = 0x40;

	constexpr uint32_t DDSCAPS_TEXTURE = 0x1000;
	constexpr uint32_t DDSCAPS_COMPLEX = 0x8;
	constexpr uint32_t DDSCAPS_MIPMAP = 0x400000;
	constexpr uint32_t DDSCAPS2_CUBEMAP = 0x200;
	constexpr uint32_t CUBE_FACES = 0xFE00; // all six face flags

	struct Blob
	{
		std::vector<uint8_t> bytes;

		void u32(uint32_t v)
		{
			for (int i = 0; i < 4; ++i)
				bytes.push_back(static_cast<uint8_t>(v >> (8 * i)));
		}
		void pad(size_t n) { bytes.insert(bytes.end(), n, 0); }
	};

	// Builds a DDS header; data_bytes is appended after (header + optional
	// DX10 block) filled with an offset-derived pattern so data pointers can
	// be verified.
	Blob make_dds(uint32_t width, uint32_t height, uint32_t mips,
		uint32_t flags_extra, uint32_t pf_flags, uint32_t four_cc,
		uint32_t bit_count, uint32_t rm, uint32_t gm, uint32_t bm, uint32_t am,
		uint32_t caps2, size_t data_bytes, uint32_t dx10_format = 0,
		uint32_t dx10_array = 1, uint32_t dx10_misc = 0, bool use_dx10 = false)
	{
		Blob b;
		b.u32(0x20534444); // 'DDS '
		b.u32(124);        // header size
		uint32_t flags = DDSD_CAPS | DDSD_HEIGHT | DDSD_WIDTH | DDSD_PIXELFORMAT | flags_extra;
		if (mips > 1)
			flags |= DDSD_MIPMAPCOUNT;
		b.u32(flags);
		b.u32(height);
		b.u32(width);
		b.u32(0); // pitch
		b.u32(0); // depth
		b.u32(mips > 1 ? mips : 0);
		b.pad(44); // reserved1[11]
		b.u32(32); // pixel format size
		b.u32(pf_flags);
		b.u32(four_cc);
		b.u32(bit_count);
		b.u32(rm);
		b.u32(gm);
		b.u32(bm);
		b.u32(am);
		b.u32(DDSCAPS_TEXTURE | (mips > 1 ? (DDSCAPS_COMPLEX | DDSCAPS_MIPMAP) : 0));
		b.u32(caps2);
		b.u32(0); // caps3
		b.u32(0); // caps4
		b.u32(0); // reserved2

		if (use_dx10)
		{
			b.u32(dx10_format);
			b.u32(3); // DDS_DIMENSION_TEXTURE2D
			b.u32(dx10_misc);
			b.u32(dx10_array);
			b.u32(0); // miscFlags2
		}

		// Pixel data pattern: byte at file offset o == (o * 7 + 3) & 0xFF.
		const size_t base = b.bytes.size();
		for (size_t i = 0; i < data_bytes; ++i)
		{
			const size_t file_off = base + i;
			b.bytes.push_back(static_cast<uint8_t>((file_off * 7 + 3) & 0xFF));
		}
		return b;
	}

	Blob make_bc_dds(uint32_t four_cc, uint32_t width, uint32_t height, uint32_t mips,
		size_t data_bytes, uint32_t caps2 = 0)
	{
		return make_dds(width, height, mips, DDSD_LINEARSIZE, DDPF_FOURCC, four_cc,
			0, 0, 0, 0, 0, caps2, data_bytes);
	}

	Blob make_dx10_dds(uint32_t dxgi_format, uint32_t width, uint32_t height, uint32_t mips,
		uint32_t array_size, uint32_t misc, size_t data_bytes)
	{
		return make_dds(width, height, mips, DDSD_LINEARSIZE, DDPF_FOURCC, 0x30315844 /*'DX10'*/,
			0, 0, 0, 0, 0, 0, data_bytes, dxgi_format, array_size, misc, true);
	}

	Blob make_rgba_dds(uint32_t width, uint32_t height, uint32_t mips, size_t data_bytes)
	{
		// Legacy A8R8G8B8 mask layout.
		return make_dds(width, height, mips, 0, DDPF_RGB | DDPF_ALPHAPIXELS, 0, 32,
			0x00FF0000, 0x0000FF00, 0x000000FF, 0xFF000000, 0, data_bytes);
	}

	uint8_t pattern_byte(size_t file_off)
	{
		return static_cast<uint8_t>((file_off * 7 + 3) & 0xFF);
	}

} // namespace

WWMI_TEST(dds_rejects_bad_magic_and_short_buffer)
{
	wwmi::DdsTexture tex;
	std::string err;

	std::vector<uint8_t> not_dds = {'D', 'D', 'S', 'X', 0};
	EXPECT(!wwmi::parse_dds(not_dds.data(), not_dds.size(), &tex, &err));
	EXPECT(!err.empty());

	// Short buffer (< 128 bytes).
	std::vector<uint8_t> short_buf(64, 0);
	EXPECT(!wwmi::parse_dds(short_buf.data(), short_buf.size(), &tex, &err));

	// Null blob.
	EXPECT(!wwmi::parse_dds(nullptr, 128, &tex, &err));
}

WWMI_TEST(dds_rejects_bad_header_sizes)
{
	wwmi::DdsTexture tex;
	std::string err;

	// Valid magic + correct structure but wrong header size.
	Blob b = make_bc_dds(0x31545844, 8, 8, 1, 32);
	b.bytes[4] = 100; // corrupt dwSize
	EXPECT(!wwmi::parse_dds(b.bytes.data(), b.bytes.size(), &tex, &err));

	// Wrong pixel format size.
	Blob c = make_bc_dds(0x31545844, 8, 8, 1, 32);
	c.bytes[4 + 72] = 16; // corrupt pixel format dwSize (offset: 4 + 7*4 + 44)
	EXPECT(!wwmi::parse_dds(c.bytes.data(), c.bytes.size(), &tex, &err));
}

WWMI_TEST(dds_legacy_dxt1)
{
	// 8x8 BC1: one mip, 32 bytes of data starting at offset 128.
	Blob b = make_bc_dds(0x31545844 /*'DXT1'*/, 8, 8, 1, 32);
	wwmi::DdsTexture tex;
	std::string err;
	EXPECT(wwmi::parse_dds(b.bytes.data(), b.bytes.size(), &tex, &err));

	EXPECT_EQ(tex.width, 8u);
	EXPECT_EQ(tex.height, 8u);
	EXPECT_EQ(tex.mip_levels, 1u);
	EXPECT_EQ(tex.array_size, 1u);
	EXPECT_EQ(tex.format, static_cast<uint32_t>(DXGI_FORMAT_BC1_UNORM));
	EXPECT(!tex.is_cubemap);
	EXPECT_EQ(tex.subresource_count(), 1u);

	const wwmi::DdsSubresource *s = tex.subresource(0, 0);
	EXPECT(s != nullptr);
	EXPECT_EQ(s->offset, 128u);
	EXPECT_EQ(s->row_pitch, 16u); // 2 blocks * 8 bytes
	EXPECT_EQ(s->slice_pitch, 32u);
	EXPECT_EQ(tex.data, b.bytes.data() + 128);
	EXPECT_EQ(tex.data_size, 32u);

	// Data pattern check: first pixel byte matches the file offset pattern.
	EXPECT_EQ(tex.data[0], pattern_byte(128));
	EXPECT_EQ(tex.data[31], pattern_byte(159));
}

WWMI_TEST(dds_legacy_dxt3_dxt5)
{
	wwmi::DdsTexture tex;
	std::string err;

	Blob b3 = make_bc_dds(0x33545844 /*'DXT3'*/, 8, 8, 1, 64);
	EXPECT(wwmi::parse_dds(b3.bytes.data(), b3.bytes.size(), &tex, &err));
	EXPECT_EQ(tex.format, static_cast<uint32_t>(DXGI_FORMAT_BC2_UNORM));

	Blob b5 = make_bc_dds(0x35545844 /*'DXT5'*/, 8, 8, 1, 64);
	EXPECT(wwmi::parse_dds(b5.bytes.data(), b5.bytes.size(), &tex, &err));
	EXPECT_EQ(tex.format, static_cast<uint32_t>(DXGI_FORMAT_BC3_UNORM));

	// DXT2/DXT4 (premultiplied alpha variants) also map to BC2/BC3.
	Blob b2 = make_bc_dds(0x32545844, 8, 8, 1, 64);
	EXPECT(wwmi::parse_dds(b2.bytes.data(), b2.bytes.size(), &tex, &err));
	EXPECT_EQ(tex.format, static_cast<uint32_t>(DXGI_FORMAT_BC2_UNORM));

	Blob b4 = make_bc_dds(0x34545844, 8, 8, 1, 64);
	EXPECT(wwmi::parse_dds(b4.bytes.data(), b4.bytes.size(), &tex, &err));
	EXPECT_EQ(tex.format, static_cast<uint32_t>(DXGI_FORMAT_BC3_UNORM));

	// ATI1/ATI2 (BC4/BC5): BC4 blocks are 8 bytes, BC5 blocks are 16.
	Blob ati1 = make_bc_dds(0x31495441, 8, 8, 1, 32);
	EXPECT(wwmi::parse_dds(ati1.bytes.data(), ati1.bytes.size(), &tex, &err));
	EXPECT_EQ(tex.format, static_cast<uint32_t>(DXGI_FORMAT_BC4_UNORM));

	Blob ati2 = make_bc_dds(0x32495441, 8, 8, 1, 64);
	EXPECT(wwmi::parse_dds(ati2.bytes.data(), ati2.bytes.size(), &tex, &err));
	EXPECT_EQ(tex.format, static_cast<uint32_t>(DXGI_FORMAT_BC5_UNORM));
}

WWMI_TEST(dds_dx10_bc7_with_mips)
{
	// 8x8 BC7 (16 bytes/block), 3 mips: 64 (8x8), 16 (4x4), 16 (2x2 padded).
	const size_t total = 64 + 16 + 16;
	Blob b = make_dx10_dds(DXGI_FORMAT_BC7_UNORM, 8, 8, 3, 1, 0, total);
	wwmi::DdsTexture tex;
	std::string err;
	EXPECT(wwmi::parse_dds(b.bytes.data(), b.bytes.size(), &tex, &err));

	EXPECT_EQ(tex.format, static_cast<uint32_t>(DXGI_FORMAT_BC7_UNORM));
	EXPECT_EQ(tex.mip_levels, 3u);
	EXPECT_EQ(tex.subresource_count(), 3u);

	// Data starts after the 20-byte DX10 header.
	const wwmi::DdsSubresource *m0 = tex.subresource(0, 0);
	const wwmi::DdsSubresource *m1 = tex.subresource(0, 1);
	const wwmi::DdsSubresource *m2 = tex.subresource(0, 2);
	EXPECT(m0 != nullptr);
	EXPECT(m1 != nullptr);
	EXPECT(m2 != nullptr);
	if (!m0 || !m1 || !m2 || !tex.data)
		return; // never dereference a failed parse below

	EXPECT_EQ(m0->offset, 148u);
	EXPECT_EQ(m0->slice_pitch, 64u);
	EXPECT_EQ(m1->offset, 212u);
	EXPECT_EQ(m1->slice_pitch, 16u);
	EXPECT_EQ(m2->offset, 228u);
	EXPECT_EQ(m2->slice_pitch, 16u);
	EXPECT_EQ(tex.data_size, total);
	EXPECT_EQ(tex.data[0], pattern_byte(148));
}

WWMI_TEST(dds_dx10_srgb_variant)
{
	Blob b = make_dx10_dds(DXGI_FORMAT_BC3_UNORM_SRGB, 8, 8, 1, 1, 0, 64);
	wwmi::DdsTexture tex;
	std::string err;
	EXPECT(wwmi::parse_dds(b.bytes.data(), b.bytes.size(), &tex, &err));
	EXPECT_EQ(tex.format, static_cast<uint32_t>(DXGI_FORMAT_BC3_UNORM_SRGB));

	// Unsupported DX10 format (UNKNOWN) rejected.
	Blob bad = make_dx10_dds(DXGI_FORMAT_UNKNOWN, 8, 8, 1, 1, 0, 32);
	EXPECT(!wwmi::parse_dds(bad.bytes.data(), bad.bytes.size(), &tex, &err));
	EXPECT(!err.empty());
}

WWMI_TEST(dds_legacy_rgba_masks)
{
	// 4x4 A8R8G8B8: 64 bytes, row pitch 16.
	Blob b = make_rgba_dds(4, 4, 1, 64);
	wwmi::DdsTexture tex;
	std::string err;
	EXPECT(wwmi::parse_dds(b.bytes.data(), b.bytes.size(), &tex, &err));

	EXPECT_EQ(tex.format, static_cast<uint32_t>(DXGI_FORMAT_B8G8R8A8_UNORM));
	const wwmi::DdsSubresource *s = tex.subresource(0, 0);
	EXPECT_EQ(s->offset, 128u);
	EXPECT_EQ(s->row_pitch, 16u);
	EXPECT_EQ(s->slice_pitch, 64u);

	// R8G8B8A8 byte-order masks.
	Blob b2 = make_dds(4, 4, 1, 0, DDPF_RGB | DDPF_ALPHAPIXELS, 0, 32,
		0x000000FF, 0x0000FF00, 0x00FF0000, 0xFF000000, 0, 64);
	EXPECT(wwmi::parse_dds(b2.bytes.data(), b2.bytes.size(), &tex, &err));
	EXPECT_EQ(tex.format, static_cast<uint32_t>(DXGI_FORMAT_R8G8B8A8_UNORM));

	// X8R8G8B8 (no alpha) -> B8G8R8X8.
	Blob b3 = make_dds(4, 4, 1, 0, DDPF_RGB, 0, 32,
		0x00FF0000, 0x0000FF00, 0x000000FF, 0, 0, 64);
	EXPECT(wwmi::parse_dds(b3.bytes.data(), b3.bytes.size(), &tex, &err));
	EXPECT_EQ(tex.format, static_cast<uint32_t>(DXGI_FORMAT_B8G8R8X8_UNORM));

	// 24bpp RGB is not supported.
	Blob b24 = make_dds(4, 4, 1, 0, DDPF_RGB, 0, 24,
		0xFF0000, 0x00FF00, 0x0000FF, 0, 0, 48);
	EXPECT(!wwmi::parse_dds(b24.bytes.data(), b24.bytes.size(), &tex, &err));
}

WWMI_TEST(dds_legacy_16bpp_masks)
{
	wwmi::DdsTexture tex;
	std::string err;

	Blob b565 = make_dds(4, 4, 1, 0, DDPF_RGB, 0, 16, 0xF800, 0x07E0, 0x001F, 0, 0, 32);
	EXPECT(wwmi::parse_dds(b565.bytes.data(), b565.bytes.size(), &tex, &err));
	EXPECT_EQ(tex.format, static_cast<uint32_t>(DXGI_FORMAT_B5G6R5_UNORM));

	Blob b5551 = make_dds(4, 4, 1, 0, DDPF_RGB | DDPF_ALPHAPIXELS, 0, 16,
		0x7C00, 0x03E0, 0x001F, 0x8000, 0, 32);
	EXPECT(wwmi::parse_dds(b5551.bytes.data(), b5551.bytes.size(), &tex, &err));
	EXPECT_EQ(tex.format, static_cast<uint32_t>(DXGI_FORMAT_B5G5R5A1_UNORM));

	Blob b4444 = make_dds(4, 4, 1, 0, DDPF_RGB | DDPF_ALPHAPIXELS, 0, 16,
		0x0F00, 0x00F0, 0x000F, 0xF000, 0, 32);
	EXPECT(wwmi::parse_dds(b4444.bytes.data(), b4444.bytes.size(), &tex, &err));
	EXPECT_EQ(tex.format, static_cast<uint32_t>(DXGI_FORMAT_B4G4R4A4_UNORM));
}

WWMI_TEST(dds_mip_chain_offsets_rgba)
{
	// 8x8 RGBA, 4 mips: sizes 256, 64, 16, 4.
	const size_t total = 256 + 64 + 16 + 4;
	Blob b = make_rgba_dds(8, 8, 4, total);
	wwmi::DdsTexture tex;
	std::string err;
	EXPECT(wwmi::parse_dds(b.bytes.data(), b.bytes.size(), &tex, &err));

	EXPECT_EQ(tex.mip_levels, 4u);
	EXPECT_EQ(tex.subresource_count(), 4u);
	EXPECT_EQ(tex.data_size, total);

	const size_t expect_offset[4] = {128, 384, 448, 464};
	const size_t expect_size[4] = {256, 64, 16, 4};
	for (uint32_t m = 0; m < 4; ++m)
	{
		const wwmi::DdsSubresource *s = tex.subresource(0, m);
		EXPECT(s != nullptr);
		EXPECT_EQ(s->offset, expect_offset[m]);
		EXPECT_EQ(s->slice_pitch, expect_size[m]);
		// Data pointer must alias the blob at the right file offset.
		EXPECT_EQ(*(tex.data + (s->offset - 128)), pattern_byte(s->offset));
	}

	// Out-of-range accessors are safe.
	EXPECT(tex.subresource(0, 4) == nullptr);
	EXPECT(tex.subresource(1, 0) == nullptr);
}

WWMI_TEST(dds_legacy_cubemap)
{
	// 8x8 BC1 cubemap: 6 faces, 32 bytes each.
	Blob b = make_bc_dds(0x31545844, 8, 8, 1, 32 * 6, CUBE_FACES);
	wwmi::DdsTexture tex;
	std::string err;
	EXPECT(wwmi::parse_dds(b.bytes.data(), b.bytes.size(), &tex, &err));

	EXPECT(tex.is_cubemap);
	EXPECT_EQ(tex.array_size, 6u);
	EXPECT_EQ(tex.subresource_count(), 6u);
	for (uint32_t f = 0; f < 6; ++f)
	{
		const wwmi::DdsSubresource *s = tex.subresource(f, 0);
		EXPECT(s != nullptr);
		EXPECT_EQ(s->offset, 128u + 32u * f);
		EXPECT_EQ(s->slice_pitch, 32u);
	}
	EXPECT_EQ(tex.data_size, 192u);
}

WWMI_TEST(dds_dx10_cubemap_and_array)
{
	wwmi::DdsTexture tex;
	std::string err;

	// DX10 cubemap: array_size must be a multiple of 6.
	Blob bad = make_dx10_dds(DXGI_FORMAT_BC1_UNORM, 8, 8, 1, 3, 0x4, 32 * 3);
	EXPECT(!wwmi::parse_dds(bad.bytes.data(), bad.bytes.size(), &tex, &err));

	// Valid DX10 cubemap with mips: 8x8 BC1, 2 mips (32 + 8 bytes per face).
	Blob cube = make_dx10_dds(DXGI_FORMAT_BC1_UNORM, 8, 8, 2, 6, 0x4, (32 + 8) * 6);
	EXPECT(wwmi::parse_dds(cube.bytes.data(), cube.bytes.size(), &tex, &err));
	EXPECT(tex.is_cubemap);
	EXPECT_EQ(tex.array_size, 6u);
	EXPECT_EQ(tex.mip_levels, 2u);
	EXPECT_EQ(tex.subresource_count(), 12u);
	// Face 1 mip 0 comes after all mips of face 0.
	EXPECT(tex.subresource(1, 0) != nullptr);
	EXPECT_EQ(tex.subresource(1, 0)->offset, 148u + 40u);
	EXPECT_EQ(tex.subresource(1, 0)->slice_pitch, 32u);

	// DX10 array (not a cubemap): 3 slices, 2 mips, BC1 8x8 (32 + 8 each).
	Blob arr = make_dx10_dds(DXGI_FORMAT_BC1_UNORM, 8, 8, 2, 3, 0, (32 + 8) * 3);
	EXPECT(wwmi::parse_dds(arr.bytes.data(), arr.bytes.size(), &tex, &err));
	EXPECT(!tex.is_cubemap);
	EXPECT_EQ(tex.array_size, 3u);
	EXPECT_EQ(tex.subresource_count(), 6u);
	// Slice-major ordering: [s0m0, s0m1, s1m0, s1m1, s2m0, s2m1].
	EXPECT(tex.subresource(0, 0) != nullptr);
	EXPECT_EQ(tex.subresource(0, 0)->offset, 148u);
	EXPECT_EQ(tex.subresource(0, 1)->offset, 180u);
	EXPECT_EQ(tex.subresource(1, 0)->offset, 188u);
	EXPECT_EQ(tex.subresource(1, 1)->offset, 220u);
	EXPECT_EQ(tex.subresource(2, 0)->offset, 228u);
	EXPECT_EQ(tex.subresource(2, 1)->offset, 260u);
}

WWMI_TEST(dds_rejects_truncated_data)
{
	wwmi::DdsTexture tex;
	std::string err;

	// 8x8 BC1 needs 32 bytes; provide 31.
	Blob b = make_bc_dds(0x31545844, 8, 8, 1, 31);
	EXPECT(!wwmi::parse_dds(b.bytes.data(), b.bytes.size(), &tex, &err));
	EXPECT(!err.empty());

	// Mip chain declared but data ends early: 8x8 RGBA 2 mips = 256 + 64.
	Blob c = make_rgba_dds(8, 8, 2, 256 + 63);
	EXPECT(!wwmi::parse_dds(c.bytes.data(), c.bytes.size(), &tex, &err));

	// Extra trailing bytes are fine (some tools pad files).
	Blob d = make_bc_dds(0x31545844, 8, 8, 1, 32 + 16);
	EXPECT(wwmi::parse_dds(d.bytes.data(), d.bytes.size(), &tex, &err));
	EXPECT_EQ(tex.data_size, 32u);
}

WWMI_TEST(dds_rejects_volume_and_zero_dims)
{
	wwmi::DdsTexture tex;
	std::string err;

	// Volume flag set -> rejected.
	Blob vol = make_bc_dds(0x31545844, 8, 8, 1, 32, 0x200000);
	EXPECT(!wwmi::parse_dds(vol.bytes.data(), vol.bytes.size(), &tex, &err));

	// Zero width.
	Blob zw = make_bc_dds(0x31545844, 0, 8, 1, 32);
	EXPECT(!wwmi::parse_dds(zw.bytes.data(), zw.bytes.size(), &tex, &err));

	// DX10 3D dimension.
	Blob d3 = make_dx10_dds(DXGI_FORMAT_R8G8B8A8_UNORM, 8, 8, 1, 1, 0, 256);
	// Patch dimension field (offset: 4 + 124 + 4).
	d3.bytes[132] = 4;
	EXPECT(!wwmi::parse_dds(d3.bytes.data(), d3.bytes.size(), &tex, &err));
}

WWMI_TEST(dds_non_multiple_of_block_sizes)
{
	// 6x6 BC1: pads to 8x8 texels = 2x2 blocks = 32 bytes.
	Blob b = make_bc_dds(0x31545844, 6, 6, 1, 32);
	wwmi::DdsTexture tex;
	std::string err;
	EXPECT(wwmi::parse_dds(b.bytes.data(), b.bytes.size(), &tex, &err));
	EXPECT_EQ(tex.width, 6u);
	EXPECT_EQ(tex.subresource(0, 0)->slice_pitch, 32u);

	// 1x1 RGBA: 4 bytes.
	Blob c = make_rgba_dds(1, 1, 1, 4);
	EXPECT(wwmi::parse_dds(c.bytes.data(), c.bytes.size(), &tex, &err));
	EXPECT_EQ(tex.subresource(0, 0)->row_pitch, 4u);
	EXPECT_EQ(tex.subresource(0, 0)->slice_pitch, 4u);
}

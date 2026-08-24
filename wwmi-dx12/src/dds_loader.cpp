// WWMI-DX12: DDS texture loader (see dds_loader.hpp).
//
// Layout math uses the same DirectXTK-derived get_surface_info() as the
// texture hash port so pitches and sizes agree with 3DMigoto's expectations.
#include "dds_loader.hpp"

#include "texture_hash.hpp"

#include <dxgiformat.h>

#include <algorithm>
#include <cstdio>
#include <cstring>

namespace wwmi
{

	namespace
	{

#pragma pack(push, 4)

		struct DdsPixelFormat
		{
			uint32_t size;
			uint32_t flags;
			uint32_t four_cc;
			uint32_t rgb_bit_count;
			uint32_t r_bit_mask;
			uint32_t g_bit_mask;
			uint32_t b_bit_mask;
			uint32_t a_bit_mask;
		};
		static_assert(sizeof(DdsPixelFormat) == 32);

		struct DdsHeader
		{
			uint32_t size;
			uint32_t flags;
			uint32_t height;
			uint32_t width;
			uint32_t pitch_or_linear_size;
			uint32_t depth;
			uint32_t mip_map_count;
			uint32_t reserved1[11];
			DdsPixelFormat pixel_format;
			uint32_t caps;
			uint32_t caps2;
			uint32_t caps3;
			uint32_t caps4;
			uint32_t reserved2;
		};
		static_assert(sizeof(DdsHeader) == 124);

		struct DdsHeaderDx10
		{
			uint32_t dxgi_format;
			uint32_t resource_dimension;
			uint32_t misc_flag;
			uint32_t array_size;
			uint32_t misc_flags2;
		};
		static_assert(sizeof(DdsHeaderDx10) == 20);

#pragma pack(pop)

		constexpr uint32_t kDdsMagic = 0x20534444; // 'DDS '

		constexpr uint32_t kDdSDCaps = 0x1;
		constexpr uint32_t kDdSDHeight = 0x2;
		constexpr uint32_t kDdSDWidth = 0x4;
		constexpr uint32_t kDdSDPixelFormat = 0x1000;
		constexpr uint32_t kDdSDMipMapCount = 0x20000;
		constexpr uint32_t kDdSDDepth = 0x800000;

		constexpr uint32_t kDdPfAlphaPixels = 0x1;
		constexpr uint32_t kDdPfAlpha = 0x2;
		constexpr uint32_t kDdPfFourCC = 0x4;
		constexpr uint32_t kDdPfRgb = 0x40;
		constexpr uint32_t kDdPfLuminance = 0x20000;

		constexpr uint32_t kDdsCaps2CubeMap = 0x200;
		constexpr uint32_t kDdsCaps2Volume = 0x200000;

		// DX10 extended header values.
		constexpr uint32_t kDdsDimensionTexture1D = 2;
		constexpr uint32_t kDdsDimensionTexture2D = 3;
		constexpr uint32_t kDdsDimensionTexture3D = 4;
		constexpr uint32_t kDdsResourceMiscTextureCube = 0x4;

		constexpr uint32_t kFourCC_DXT1 = 0x31545844; // 'DXT1'
		constexpr uint32_t kFourCC_DXT2 = 0x32545844;
		constexpr uint32_t kFourCC_DXT3 = 0x33545844; // 'DXT3'
		constexpr uint32_t kFourCC_DXT4 = 0x34545844;
		constexpr uint32_t kFourCC_DXT5 = 0x35545844; // 'DXT5'
		constexpr uint32_t kFourCC_ATI1 = 0x31495441; // 'ATI1' (BC4)
		constexpr uint32_t kFourCC_BC4U = 0x55344342; // 'BC4U'
		constexpr uint32_t kFourCC_BC4S = 0x53344342; // 'BC4S'
		constexpr uint32_t kFourCC_ATI2 = 0x32495441; // 'ATI2' (BC5)
		constexpr uint32_t kFourCC_BC5U = 0x55354342; // 'BC5U'
		constexpr uint32_t kFourCC_BC5S = 0x53354342; // 'BC5S'
		constexpr uint32_t kFourCC_DX10 = 0x30315844; // 'DX10'

		bool format_supported_for_mods(uint32_t fmt)
		{
			switch (static_cast<DXGI_FORMAT>(fmt))
			{
			case DXGI_FORMAT_BC1_TYPELESS:
			case DXGI_FORMAT_BC1_UNORM:
			case DXGI_FORMAT_BC1_UNORM_SRGB:
			case DXGI_FORMAT_BC2_TYPELESS:
			case DXGI_FORMAT_BC2_UNORM:
			case DXGI_FORMAT_BC2_UNORM_SRGB:
			case DXGI_FORMAT_BC3_TYPELESS:
			case DXGI_FORMAT_BC3_UNORM:
			case DXGI_FORMAT_BC3_UNORM_SRGB:
			case DXGI_FORMAT_BC4_TYPELESS:
			case DXGI_FORMAT_BC4_UNORM:
			case DXGI_FORMAT_BC4_SNORM:
			case DXGI_FORMAT_BC5_TYPELESS:
			case DXGI_FORMAT_BC5_UNORM:
			case DXGI_FORMAT_BC5_SNORM:
			case DXGI_FORMAT_BC6H_TYPELESS:
			case DXGI_FORMAT_BC6H_UF16:
			case DXGI_FORMAT_BC6H_SF16:
			case DXGI_FORMAT_BC7_TYPELESS:
			case DXGI_FORMAT_BC7_UNORM:
			case DXGI_FORMAT_BC7_UNORM_SRGB:
			case DXGI_FORMAT_R8G8B8A8_TYPELESS:
			case DXGI_FORMAT_R8G8B8A8_UNORM:
			case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB:
			case DXGI_FORMAT_R8G8B8A8_SNORM:
			case DXGI_FORMAT_B8G8R8A8_TYPELESS:
			case DXGI_FORMAT_B8G8R8A8_UNORM:
			case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB:
			case DXGI_FORMAT_B8G8R8X8_UNORM:
			case DXGI_FORMAT_R16G16B16A16_TYPELESS:
			case DXGI_FORMAT_R16G16B16A16_FLOAT:
			case DXGI_FORMAT_R16G16B16A16_UNORM:
			case DXGI_FORMAT_R10G10B10A2_TYPELESS:
			case DXGI_FORMAT_R10G10B10A2_UNORM:
			case DXGI_FORMAT_R11G11B10_FLOAT:
			case DXGI_FORMAT_R8G8_TYPELESS:
			case DXGI_FORMAT_R8G8_UNORM:
			case DXGI_FORMAT_R8G8_SNORM:
			case DXGI_FORMAT_R8_TYPELESS:
			case DXGI_FORMAT_R8_UNORM:
			case DXGI_FORMAT_R8_SNORM:
			case DXGI_FORMAT_A8_UNORM:
			case DXGI_FORMAT_B5G6R5_UNORM:
			case DXGI_FORMAT_B5G5R5A1_UNORM:
			case DXGI_FORMAT_B4G4R4A4_UNORM:
				return true;
			default:
				return false;
			}
		}

		// Legacy DDSPixelFormat mask layout -> DXGI format. Covers the
		// combinations that appear in mod textures.
		bool format_from_masks(const DdsPixelFormat &pf, uint32_t *out_fmt)
		{
			const uint32_t bpp = pf.rgb_bit_count;

			if (pf.flags & kDdPfFourCC)
				return false; // handled by the caller

			if (pf.flags & kDdPfRgb)
			{
				if (bpp == 32)
				{
					const uint32_t a = (pf.flags & kDdPfAlphaPixels) ? pf.a_bit_mask : 0;
					if (a == 0xFF000000u && pf.r_bit_mask == 0x00FF0000u && pf.g_bit_mask == 0x0000FF00u && pf.b_bit_mask == 0x000000FFu)
					{
						*out_fmt = DXGI_FORMAT_B8G8R8A8_UNORM; // legacy A8R8G8B8
						return true;
					}
					if (a == 0xFF000000u && pf.r_bit_mask == 0x000000FFu && pf.g_bit_mask == 0x0000FF00u && pf.b_bit_mask == 0x00FF0000u)
					{
						*out_fmt = DXGI_FORMAT_R8G8B8A8_UNORM;
						return true;
					}
					// X8R8G8B8 (no alpha): accepted as B8G8R8X8.
					if (a == 0 && pf.r_bit_mask == 0x00FF0000u && pf.g_bit_mask == 0x0000FF00u && pf.b_bit_mask == 0x000000FFu)
					{
						*out_fmt = DXGI_FORMAT_B8G8R8X8_UNORM;
						return true;
					}
					return false;
				}
				if (bpp == 16)
				{
					if (pf.r_bit_mask == 0xF800u && pf.g_bit_mask == 0x07E0u && pf.b_bit_mask == 0x001Fu)
					{
						*out_fmt = DXGI_FORMAT_B5G6R5_UNORM;
						return true;
					}
					if (pf.flags & kDdPfAlphaPixels && pf.a_bit_mask == 0x8000u &&
						pf.r_bit_mask == 0x7C00u && pf.g_bit_mask == 0x03E0u && pf.b_bit_mask == 0x001Fu)
					{
						*out_fmt = DXGI_FORMAT_B5G5R5A1_UNORM;
						return true;
					}
					if (pf.flags & kDdPfAlphaPixels && pf.a_bit_mask == 0xF000u &&
						pf.r_bit_mask == 0x0F00u && pf.g_bit_mask == 0x00F0u && pf.b_bit_mask == 0x000Fu)
					{
						*out_fmt = DXGI_FORMAT_B4G4R4A4_UNORM;
						return true;
					}
					return false;
				}
				return false;
			}

			if (pf.flags & kDdPfAlpha && bpp == 8)
			{
				*out_fmt = DXGI_FORMAT_A8_UNORM;
				return true;
			}

			if (pf.flags & kDdPfLuminance)
			{
				if (bpp == 8)
				{
					*out_fmt = DXGI_FORMAT_R8_UNORM;
					return true;
				}
				if (bpp == 16 && pf.a_bit_mask == 0xFF00u)
				{
					*out_fmt = DXGI_FORMAT_R8G8_UNORM;
					return true;
				}
			}

			return false;
		}

		bool fourcc_to_format(uint32_t four_cc, uint32_t *out_fmt)
		{
			switch (four_cc)
			{
			case kFourCC_DXT1: *out_fmt = DXGI_FORMAT_BC1_UNORM; return true;
			case kFourCC_DXT2:
			case kFourCC_DXT3: *out_fmt = DXGI_FORMAT_BC2_UNORM; return true;
			case kFourCC_DXT4:
			case kFourCC_DXT5: *out_fmt = DXGI_FORMAT_BC3_UNORM; return true;
			case kFourCC_ATI1:
			case kFourCC_BC4U: *out_fmt = DXGI_FORMAT_BC4_UNORM; return true;
			case kFourCC_BC4S: *out_fmt = DXGI_FORMAT_BC4_SNORM; return true;
			case kFourCC_ATI2:
			case kFourCC_BC5U: *out_fmt = DXGI_FORMAT_BC5_UNORM; return true;
			case kFourCC_BC5S: *out_fmt = DXGI_FORMAT_BC5_SNORM; return true;
			default: return false;
			}
		}

		struct MipInfo
		{
			size_t row_pitch;
			size_t num_rows;
			size_t total_size;
		};

		bool mip_info(uint32_t fmt, uint32_t width, uint32_t height, MipInfo *out)
		{
			SurfaceInfo si;
			if (!get_surface_info(width, height, fmt, &si))
				return false;
			out->row_pitch = si.row_bytes;
			out->num_rows = si.num_rows;
			out->total_size = si.num_bytes;
			return true;
		}

		void set_error(std::string *error, const char *msg)
		{
			if (error)
				*error = msg;
		}

	} // namespace

	bool parse_dds(const void *blob, size_t size, DdsTexture *out, std::string *error)
	{
		if (error)
			error->clear();
		if (!blob || !out)
		{
			set_error(error, "null arguments");
			return false;
		}
		std::memset(out, 0, sizeof(*out));
		out->mip_levels = 1;
		out->array_size = 1;

		const uint8_t *bytes = static_cast<const uint8_t *>(blob);

		if (size < sizeof(uint32_t) + sizeof(DdsHeader))
		{
			set_error(error, "buffer too small for DDS header");
			return false;
		}

		uint32_t magic;
		std::memcpy(&magic, bytes, sizeof(magic));
		if (magic != kDdsMagic)
		{
			set_error(error, "bad DDS magic");
			return false;
		}

		DdsHeader header;
		std::memcpy(&header, bytes + 4, sizeof(header));
		if (header.size != sizeof(DdsHeader))
		{
			set_error(error, "bad DDS header size");
			return false;
		}
		if (header.pixel_format.size != sizeof(DdsPixelFormat))
		{
			set_error(error, "bad DDS pixel format size");
			return false;
		}
		if (!(header.flags & kDdSDWidth) || !(header.flags & kDdSDHeight))
		{
			set_error(error, "DDS header missing width/height");
			return false;
		}
		if (header.width == 0 || header.height == 0)
		{
			set_error(error, "DDS texture has zero dimension");
			return false;
		}
		if (header.caps2 & kDdsCaps2Volume)
		{
			set_error(error, "volume (3D) textures are not supported");
			return false;
		}

		uint32_t mips = (header.flags & kDdSDMipMapCount) ? header.mip_map_count : 1;
		if (mips == 0)
			mips = 1;
		if (mips > 32)
		{
			set_error(error, "unreasonable mip count");
			return false;
		}

		uint32_t array_size = 1;
		bool is_cubemap = false;
		uint32_t format = DXGI_FORMAT_UNKNOWN;
		size_t header_bytes = 4 + sizeof(DdsHeader);

		if (header.pixel_format.flags & kDdPfFourCC && header.pixel_format.four_cc == kFourCC_DX10)
		{
			if (size < header_bytes + sizeof(DdsHeaderDx10))
			{
				set_error(error, "truncated DX10 header");
				return false;
			}
			DdsHeaderDx10 dx10;
			std::memcpy(&dx10, bytes + header_bytes, sizeof(dx10));
			header_bytes += sizeof(DdsHeaderDx10);

			format = dx10.dxgi_format;
			if (dx10.resource_dimension == kDdsDimensionTexture1D ||
				dx10.resource_dimension == kDdsDimensionTexture3D)
			{
				set_error(error, "only 2D DDS textures are supported");
				return false;
			}
			if (dx10.resource_dimension != kDdsDimensionTexture2D)
			{
				set_error(error, "bad DX10 resource dimension");
				return false;
			}

			is_cubemap = (dx10.misc_flag & kDdsResourceMiscTextureCube) != 0;
			array_size = dx10.array_size;
			if (array_size == 0)
				array_size = 1;
			if (is_cubemap && array_size % 6 != 0)
			{
				set_error(error, "cubemap array size must be a multiple of 6");
				return false;
			}
		}
		else if (header.pixel_format.flags & kDdPfFourCC)
		{
			if (!fourcc_to_format(header.pixel_format.four_cc, &format))
			{
				set_error(error, "unsupported DDS FourCC");
				return false;
			}
			if (header.caps2 & kDdsCaps2CubeMap)
			{
				is_cubemap = true;
				array_size = 6;
			}
		}
		else
		{
			if (!format_from_masks(header.pixel_format, &format))
			{
				set_error(error, "unsupported DDS pixel format (masks)");
				return false;
			}
			if (header.caps2 & kDdsCaps2CubeMap)
			{
				is_cubemap = true;
				array_size = 6;
			}
		}

		if (!format_supported_for_mods(format))
		{
			set_error(error, "unsupported DXGI format for mods");
			return false;
		}
		// Block compressed textures must be at least one full block; the
		// surface math already pads sub-block mips, so no extra check needed.

		// Build the subresource layout: slice-major, then mip.
		const size_t subresource_count = static_cast<size_t>(array_size) * mips;
		out->subresources.resize(subresource_count);

		const uint8_t *pixel_data = bytes + header_bytes;
		const size_t available = size - header_bytes;

		uint64_t offset = 0;
		for (uint32_t slice = 0; slice < array_size; ++slice)
		{
			for (uint32_t mip = 0; mip < mips; ++mip)
			{
				const uint32_t w = std::max(header.width >> mip, 1u);
				const uint32_t h = std::max(header.height >> mip, 1u);

				MipInfo mi;
				if (!mip_info(format, w, h, &mi))
				{
					set_error(error, "format has no surface math");
					return false;
				}

				DdsSubresource &sub = out->subresources[static_cast<size_t>(slice) * mips + mip];
				sub.offset = header_bytes + static_cast<size_t>(offset);
				sub.row_pitch = mi.row_pitch;
				sub.slice_pitch = mi.total_size;
				offset += mi.total_size;
			}
		}

		if (offset > available)
		{
			set_error(error, "DDS pixel data truncated");
			return false;
		}

		out->width = header.width;
		out->height = header.height;
		out->mip_levels = mips;
		out->array_size = array_size;
		out->format = format;
		out->is_cubemap = is_cubemap;
		out->data = pixel_data;
		out->data_size = static_cast<size_t>(offset);
		return true;
	}

	bool load_dds_file(const char *path, std::vector<uint8_t> *blob, DdsTexture *out,
		std::string *error)
	{
		if (error)
			error->clear();
		if (!path || !blob || !out)
		{
			set_error(error, "null arguments");
			return false;
		}

		FILE *f = nullptr;
		if (fopen_s(&f, path, "rb") != 0 || !f)
		{
			set_error(error, "cannot open file");
			return false;
		}

		blob->clear();
		uint8_t chunk[65536];
		size_t n;
		while ((n = std::fread(chunk, 1, sizeof(chunk), f)) > 0)
			blob->insert(blob->end(), chunk, chunk + n);
		const bool read_error = std::ferror(f) != 0;
		std::fclose(f);

		if (read_error)
		{
			set_error(error, "file read error");
			return false;
		}

		return parse_dds(blob->data(), blob->size(), out, error);
	}

} // namespace wwmi

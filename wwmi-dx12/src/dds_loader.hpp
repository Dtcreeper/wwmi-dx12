// WWMI-DX12: DDS texture loader.
//
// Parses the .dds files shipped with mods (BC1/2/3/4/5/6/7 via FourCC or the
// DX10 extended header, plus common uncompressed RGBA formats). Zero-copy:
// the parsed DdsTexture points into the caller's blob; subresource
// offsets/pitches describe the tightly-packed DDS file layout
// (array-slice-major, then mip level, per the DDS convention).
//
// Scope is deliberately mod-oriented: 2D textures, cubemaps and arrays are
// supported; 1D/3D/planar-video formats are rejected with a clear error.
#pragma once

#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>

namespace wwmi
{

	struct DdsSubresource
	{
		size_t offset = 0;      // byte offset from the start of the DDS blob (incl. headers)
		size_t row_pitch = 0;   // bytes per row (tight, as stored in the file)
		size_t slice_pitch = 0; // bytes per 2D slice (== total size for 2D)
	};

	struct DdsTexture
	{
		uint32_t width = 0;
		uint32_t height = 0;
		uint32_t mip_levels = 1;
		uint32_t array_size = 1; // includes the 6 faces of a cubemap
		uint32_t format = 0;     // DXGI_FORMAT
		bool is_cubemap = false;

		const uint8_t *data = nullptr; // first pixel byte, points into the blob
		size_t data_size = 0;          // total pixel data bytes

		// [slice * mip_levels + mip]
		std::vector<DdsSubresource> subresources;

		size_t subresource_count() const { return subresources.size(); }
		const DdsSubresource *subresource(uint32_t slice, uint32_t mip) const
		{
			const size_t idx = static_cast<size_t>(slice) * mip_levels + mip;
			return idx < subresources.size() ? &subresources[idx] : nullptr;
		}
	};

	// Parse a DDS blob. Returns false and sets *error (when provided) on
	// malformed input, unsupported formats, or truncated data.
	bool parse_dds(const void *blob, size_t size, DdsTexture *out, std::string *error = nullptr);

	// Convenience: read a .dds file from disk, then parse it.
	bool load_dds_file(const char *path, std::vector<uint8_t> *blob, DdsTexture *out,
		std::string *error = nullptr);

} // namespace wwmi

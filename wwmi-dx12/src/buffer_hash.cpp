#include "buffer_hash.hpp"

#include "crc32c.hpp"

namespace wwmi
{
	uint32_t buffer_bind_flags(BufferRole role)
	{
		switch (role)
		{
		case BufferRole::vertex:
			return 0x1; // D3D11_BIND_VERTEX_BUFFER
		case BufferRole::index:
			return 0x2; // D3D11_BIND_INDEX_BUFFER
		default:
			return 0;
		}
	}

	uint32_t calc_buffer_data_hash(const void *data, size_t size)
	{
		return crc32c_extend(0, data, size);
	}

	uint32_t calc_buffer_desc_hash(uint32_t seed, uint64_t byte_width, BufferRole role)
	{
		D3D11BufferDescBytes desc{};
		desc.byte_width = static_cast<uint32_t>(byte_width);
		desc.bind_flags = buffer_bind_flags(role);
		// usage/cpu_access/misc/stride stay 0 (UE mesh buffers are
		// D3D11_USAGE_DEFAULT with no extra flags).
		return crc32c_extend(seed, &desc, sizeof(desc));
	}

	uint32_t calc_buffer_hash(uint32_t data_hash, uint64_t byte_width, BufferRole role)
	{
		return calc_buffer_desc_hash(data_hash, byte_width, role);
	}
}

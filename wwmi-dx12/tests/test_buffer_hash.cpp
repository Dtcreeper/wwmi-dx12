#include "buffer_hash.hpp"
#include "crc32c.hpp"
#include "test_framework.hpp"

#include <cstring>

using namespace wwmi;

// The desc term must hash exactly the 24 documented D3D11_BUFFER_DESC
// bytes: width LE, usage=0, bind, and zeroed tail fields.
WWMI_TEST(buffer_desc_hash_byte_layout)
{
	const uint8_t expected[24] = {
		0x64, 0x00, 0x00, 0x00, // ByteWidth = 100
		0x00, 0x00, 0x00, 0x00, // Usage = DEFAULT
		0x02, 0x00, 0x00, 0x00, // BindFlags = D3D11_BIND_INDEX_BUFFER
		0x00, 0x00, 0x00, 0x00, // CPUAccessFlags
		0x00, 0x00, 0x00, 0x00, // MiscFlags
		0x00, 0x00, 0x00, 0x00, // StructureByteStride
	};

	const uint32_t want = crc32c_extend(0, expected, sizeof(expected));
	EXPECT_EQ(calc_buffer_desc_hash(0, 100, BufferRole::index), want);
	EXPECT_NE(calc_buffer_desc_hash(0, 100, BufferRole::vertex), want);
	EXPECT_NE(calc_buffer_desc_hash(0, 101, BufferRole::index), want);

	// Role mapping mirrors the D3D11 bind flags.
	EXPECT_EQ(buffer_bind_flags(BufferRole::vertex), 0x1u);
	EXPECT_EQ(buffer_bind_flags(BufferRole::index), 0x2u);
	EXPECT_EQ(buffer_bind_flags(BufferRole::unknown), 0u);
}

// Full formula matches the 3DMigoto chain
// crc32c(crc32c(0, data, ByteWidth), desc, 24).
WWMI_TEST(buffer_hash_matches_3dmigoto_chain)
{
	const uint8_t data[16] = {
		0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
		0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f,
	};

	const uint32_t data_hash = calc_buffer_data_hash(data, sizeof(data));
	EXPECT_EQ(data_hash, crc32c_extend(0, data, sizeof(data)));

	D3D11BufferDescBytes desc{};
	desc.byte_width = sizeof(data);
	desc.bind_flags = buffer_bind_flags(BufferRole::vertex);
	const uint32_t want = crc32c_extend(data_hash, &desc, sizeof(desc));

	EXPECT_EQ(calc_buffer_hash(data_hash, sizeof(data), BufferRole::vertex), want);

	// Chaining is associative with the manual two-step call.
	EXPECT_EQ(calc_buffer_hash(data_hash, sizeof(data), BufferRole::vertex),
		calc_buffer_desc_hash(calc_buffer_data_hash(data, sizeof(data)), sizeof(data), BufferRole::vertex));
}

// 3DMigoto's no-initial-data path: hash = crc32c(0, desc) with the
// data term starting from seed 0.
WWMI_TEST(buffer_hash_no_data_path)
{
	const uint32_t empty = calc_buffer_hash(0, 4096, BufferRole::index);
	EXPECT_EQ(empty, calc_buffer_desc_hash(0, 4096, BufferRole::index));

	D3D11BufferDescBytes desc{};
	desc.byte_width = 4096;
	desc.bind_flags = 0x2;
	EXPECT_EQ(empty, crc32c_extend(0, &desc, sizeof(desc)));

	// Empty data hashes to 0 only for zero-length input; real readback
	// data always contributes.
	EXPECT_EQ(calc_buffer_data_hash(nullptr, 0), 0u);
	EXPECT_EQ(calc_buffer_data_hash("", 0), 0u);
}

// Same contents in different roles (VB vs IB) produce different hashes
// because BindFlags participates in the desc term.
WWMI_TEST(buffer_hash_role_disambiguates)
{
	const uint32_t vb = calc_buffer_hash(0x12345678u, 2048, BufferRole::vertex);
	const uint32_t ib = calc_buffer_hash(0x12345678u, 2048, BufferRole::index);
	const uint32_t unk = calc_buffer_hash(0x12345678u, 2048, BufferRole::unknown);
	EXPECT_NE(vb, ib);
	EXPECT_NE(vb, unk);
	EXPECT_NE(ib, unk);
}

// Widths above 4GB are truncated the way a D3D11 app could never pass
// them (defensive; mesh buffers never reach this size).
WWMI_TEST(buffer_hash_width_truncation_is_documented)
{
	const uint64_t huge = 0x100000000ULL + 16;
	EXPECT_EQ(calc_buffer_hash(0, huge, BufferRole::index),
		calc_buffer_hash(0, 16, BufferRole::index));
}

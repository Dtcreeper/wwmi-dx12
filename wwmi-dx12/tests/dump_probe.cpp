// WWMI-DX12 diagnostic: brute-force 3DMigoto buffer-hash hypotheses
// over a dumped index-view region (the .bin files the addon writes to
// <data-root>/Dumps on the first mismatch per view).
//
//   dump-probe <region.bin> <want-hash-hex> [index_size]
//
// 3DMigoto's hash is crc32c(crc32c(data, W), &D3D11_BUFFER_DESC, 24)
// where W is the ORIGINAL DX11 buffer's ByteWidth -- which the mod
// rules only imply (window span), never state. For every prefix
// length W (step = index_size, default 4) this tool chains the 24
// desc bytes with Usage in {DEFAULT, IMMUTABLE, DYNAMIC, STAGING} and
// BindFlags = D3D11_BIND_INDEX_BUFFER, and reports every (W, Usage)
// whose full hash equals <want-hash-hex>.
//
// Output reading:
//   MATCH W=... usage=...   -> the desc/width hypothesis that fits;
//                              feed it back into the probe formula.
//   no match                -> the region bytes themselves differ from
//                              the DX11 buffer (wrong base offset,
//                              16<->32 bit width, or re-indexed data).
#include "crc32c.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <vector>

namespace
{
	// Byte-exact D3D11_BUFFER_DESC layout 3DMigoto hashes (24 bytes).
	struct DescBytes
	{
		uint32_t byte_width;
		uint32_t usage;
		uint32_t bind_flags;
		uint32_t cpu_access_flags;
		uint32_t misc_flags;
		uint32_t structure_byte_stride;
	};
	static_assert(sizeof(DescBytes) == 24, "D3D11_BUFFER_DESC layout");
}

int main(int argc, char **argv)
{
	if (argc < 3)
	{
		std::fprintf(stderr, "usage: dump-probe <region.bin> <want-hex> [index_size]\n");
		return 1;
	}

	std::ifstream f(argv[1], std::ios::binary);
	if (!f)
	{
		std::fprintf(stderr, "cannot open %s\n", argv[1]);
		return 1;
	}
	f.seekg(0, std::ios::end);
	const std::streamoff len_off = f.tellg();
	f.seekg(0, std::ios::beg);
	std::vector<unsigned char> bytes(static_cast<size_t>(len_off));
	if (!bytes.empty())
		f.read(reinterpret_cast<char *>(bytes.data()), bytes.size());
	if (!f && !bytes.empty())
	{
		std::fprintf(stderr, "short read on %s\n", argv[1]);
		return 1;
	}

	const uint32_t want = static_cast<uint32_t>(
		std::strtoull(argv[2], nullptr, 16));
	unsigned step = 4;
	if (argc > 3)
	{
		step = static_cast<unsigned>(std::strtoul(argv[3], nullptr, 0));
		if (step == 0 || step > 16)
		{
			std::fprintf(stderr, "bad index_size %u\n", step);
			return 1;
		}
	}

	// Incremental prefix CRCs: prefix[i] = CRC32C(bytes[0 .. (i+1)*step)).
	// One pass, O(n): every candidate W costs one desc-size extend.
	std::vector<uint32_t> prefix;
	prefix.reserve(bytes.size() / step + 1);
	uint32_t crc = 0;
	for (size_t off = 0; off + step <= bytes.size(); off += step)
	{
		crc = wwmi::crc32c_extend(crc, bytes.data() + off, step);
		prefix.push_back(crc);
	}

	unsigned matches = 0;
	for (size_t i = 0; i < prefix.size(); ++i)
	{
		const uint32_t w = static_cast<uint32_t>((i + 1) * step);
		for (uint32_t usage = 0; usage < 4; ++usage)
		{
			const DescBytes d{ w, usage,
				2u /* D3D11_BIND_INDEX_BUFFER */, 0, 0, 0 };

			uint32_t full = wwmi::crc32c_extend(prefix[i], &d, sizeof d);
			if (full == want)
			{
				std::printf("MATCH W=%u usage=%u\n", w, usage);
				++matches;
			}

			// Buffer created without initial data (data term = 0).
			full = wwmi::crc32c_extend(0, &d, sizeof d);
			if (full == want)
			{
				std::printf("MATCH W=%u usage=%u (no initial data)\n",
					w, usage);
				++matches;
			}
		}
	}

	std::printf("%u match(es); W range [%u, %llu] step %u; want=%08x\n",
		matches, step,
		static_cast<unsigned long long>(
			static_cast<uint64_t>(prefix.size()) * step),
		step, want);
	return matches == 0 ? 2 : 0;
}

// WWMI-DX12: CPU-side skeleton chain (M9). See skeleton_chain.hpp.
#include "skeleton_chain.hpp"

#include <algorithm>
#include <cstring>

namespace wwmi
{
	void SkeletonChain::finalize_config()
	{
		for (uint32_t r = 0; r < k_remap_tables; ++r)
			_remap_bones[r] = 0;
		for (const auto &kv : rules)
		{
			const Component &c = kv.second;
			if (c.remap_id >= 0 && c.remap_id < static_cast<int32_t>(k_remap_tables))
				_remap_bones[c.remap_id] = std::max(_remap_bones[c.remap_id], c.remap_count);
		}
	}

	void SkeletonChain::merge_one(const float *game, float *out,
		uint32_t *written) const
	{
		// out covers k_merged_bones bones x 3 float4 rows (12 floats each),
		// already zero-filled by the caller. Bone vg of the game CB is its
		// rows vg*3..vg*3+2; the merged slot is (vg_offset+vg)*3 rows.
		uint32_t bones = 0;
		for (const auto &kv : rules)
		{
			const Component &c = kv.second;
			for (uint32_t vg = 0; vg < c.vg_count; ++vg)
			{
				// Presence heuristic, replicated from SkeletonMerger.hlsl:
				// any non-zero float in rows vg, vg+1, vg+2 (un-multiplied
				// index) keeps the bone alive.
				bool alive = false;
				for (uint32_t r = 0; r < 3 && !alive; ++r)
				{
					const uint32_t row = vg + r; // float4 row in the game CB
					if (row >= k_game_rows)
						break;
					const float *f = game + row * 4;
					for (uint32_t e = 0; e < 4; ++e)
						if (f[e] != 0.0f)
						{
							alive = true;
							break;
						}
				}
				if (!alive)
					continue;

				const uint32_t src_row = vg * 3; // float4 rows
				const uint32_t dst_slot = c.vg_offset + vg;
				if (src_row + 2 >= k_game_rows || dst_slot >= k_merged_bones)
					continue;

				std::memcpy(out + static_cast<size_t>(dst_slot) * 12,
					game + static_cast<size_t>(src_row) * 4, 12 * sizeof(float));
				++bones;
			}
		}
		*written = bones;
	}

	void SkeletonChain::merge(const float *game_main, const float *game_extra)
	{
		_merged.assign(static_cast<size_t>(k_merged_bones) * 12, 0.0f);
		_extra.assign(static_cast<size_t>(k_merged_bones) * 12, 0.0f);
		_merged_written = 0;
		_extra_written = 0;

		merge_one(game_main, _merged.data(), &_merged_written);
		merge_one(game_extra, _extra.data(), &_extra_written);

		// SkeletonRemapper.hlsl: remapped[local*3+k] =
		//   merged[forward[remap_id*512 + local]*3 + k]
		// Regions are padded to the full game-CB row count (768 float4s)
		// with zeros: the game VS declares Skeleton[768] and D3D12 root
		// CBVs carry no size, so any read past the live bones must land
		// on zeroed rows instead of the neighboring upload regions.
		for (uint32_t r = 0; r < k_remap_tables; ++r)
		{
			if (_remap_bones[r] == 0)
				continue;
			const uint32_t count = std::min(_remap_bones[r], k_max_remap_bones);
			_remapped[r].assign(static_cast<size_t>(k_game_rows) * 12, 0.0f);
			_extra_remapped[r].assign(static_cast<size_t>(k_game_rows) * 12, 0.0f);
			for (uint32_t local = 0; local < count; ++local)
			{
				const size_t map_index = static_cast<size_t>(r) * k_remap_stride + local;
				if (map_index >= forward.size())
					break;
				const uint32_t g = forward[map_index];
				if (g >= k_merged_bones)
					continue;
				const float *src_m = _merged.data() + static_cast<size_t>(g) * 12;
				const float *src_e = _extra.data() + static_cast<size_t>(g) * 12;
				std::memcpy(_remapped[r].data() + static_cast<size_t>(local) * 12,
					src_m, 12 * sizeof(float));
				std::memcpy(_extra_remapped[r].data() + static_cast<size_t>(local) * 12,
					src_e, 12 * sizeof(float));
			}
		}
	}

	const float *SkeletonChain::remapped(uint32_t remap_id) const
	{
		if (remap_id >= k_remap_tables || _remapped[remap_id].empty())
			return nullptr;
		return _remapped[remap_id].data();
	}

	const float *SkeletonChain::extra_remapped(uint32_t remap_id) const
	{
		if (remap_id >= k_remap_tables || _extra_remapped[remap_id].empty())
			return nullptr;
		return _extra_remapped[remap_id].data();
	}
}

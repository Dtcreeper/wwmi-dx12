// WWMI-DX12: CPU-side skeleton chain (M9).
//
// Replicates the WWMIv1 compute-shader pipeline that the DX12 port had
// to strip from the mod (SkeletonMerger + SkeletonRemapper):
//
//   game vs-cb4 (main skeleton)  --merge--> MergedSkeleton      [512 bones]
//   game vs-cb3 (extra skeleton) --merge--> ExtraMergedSkeleton [512 bones]
//   MergedSkeleton      --forward remap--> per-component RemappedSkeleton
//   ExtraMergedSkeleton --forward remap--> per-component ExtraRemappedSkeleton
//
// Semantics mirrored from the HLSL (external/wwmi WWMI/Core Shaders):
//  - SkeletonMerger: for every component window (vg_offset, vg_count),
//    game bone vg's 3 float4 rows land at merged slot vg_offset+vg, but
//    only when the presence heuristic sees any non-zero float in rows
//    vg..vg+2 of the game CB (un-multiplied index -- replicated as-is).
//  - SkeletonRemapper: remapped[local] = merged[forward[remap_id*512+local]]
//    (forward = BlendRemapForward.buf, R16, 4 tables x 512 entries).
//
// The merge inputs are the game's two 12KB skeleton CBs (768 float4s =
// 256 bones x 3 rows); outputs are bound as root CBVs at the mod's
// re-issued draws (cb4 slot <- merged, cb3 slot <- extra). Like the
// WWMI "Merged" skeleton mode the data is one frame behind the game
// (CPU readback at present).
#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace wwmi
{
	class SkeletonChain
	{
	public:
		// Game skeleton CB layout: Skeleton[768] float4s (WWMI merger cbuffer).
		static constexpr uint32_t k_game_rows = 768;
		// Merged skeleton: 1536 float4s = 512 bones x 3 rows.
		static constexpr uint32_t k_merged_bones = 512;
		// Forward remap table: 4 tables x 512 R16 entries.
		static constexpr uint32_t k_remap_tables = 4;
		static constexpr uint32_t k_remap_stride = 512;
		static constexpr uint32_t k_max_remap_bones = 512;

		// One [SkeletonChain] "component = <rule>, vg_offset, vg_count[,
		// remap_id, remap_count]" declaration.
		struct Component
		{
			uint32_t vg_offset = 0;
			uint32_t vg_count = 0;
			int32_t remap_id = -1;    // -1: bind the full merged skeleton
			uint32_t remap_count = 0; // valid when remap_id >= 0
		};

		// ---- configuration (filled by load_mods, immutable afterwards) ----
		// Rule section name -> component parameters.
		std::unordered_map<std::string, Component> rules;
		// BlendRemapForward table (R16 little endian), remap_id*512+local ->
		// merged bone slot (the mod's global VG id).
		std::vector<uint16_t> forward;

		bool active() const { return !rules.empty(); }
		bool has_remap() const { return !forward.empty(); }

		// Largest remap_count registered per table id (0 = table unused).
		uint32_t remap_bones(uint32_t remap_id) const { return _remap_bones[remap_id]; }

		// ---- per-frame merge (present thread only) ----
		// game_main / game_extra: k_game_rows float4s (k_game_rows*4 floats).
		void merge(const float *game_main, const float *game_extra);

		const float *merged() const { return _merged.data(); } // 512*3 float4
		const float *extra() const { return _extra.data(); }   // 512*3 float4
		// remap_count*3 float4s, or nullptr when the table is unused.
		const float *remapped(uint32_t remap_id) const;
		const float *extra_remapped(uint32_t remap_id) const;

		static uint64_t bone_rows_bytes(uint32_t bones)
		{
			return static_cast<uint64_t>(bones) * 3 * 16;
		}
		uint64_t merged_bytes() const { return bone_rows_bytes(k_merged_bones); }

		// Diagnostics: merged bone slots written this frame.
		uint32_t merged_bones_written() const { return _merged_written; }
		uint32_t extra_bones_written() const { return _extra_written; }

		void finalize_config(); // computes _remap_bones from rules

	private:
		void merge_one(const float *game, float *out, uint32_t *written) const;

		uint32_t _remap_bones[k_remap_tables] = {};
		std::vector<float> _merged; // k_merged_bones*3 float4s
		std::vector<float> _extra;  // k_merged_bones*3 float4s
		std::vector<float> _remapped[k_remap_tables];
		std::vector<float> _extra_remapped[k_remap_tables];
		uint32_t _merged_written = 0;
		uint32_t _extra_written = 0;
	};
}

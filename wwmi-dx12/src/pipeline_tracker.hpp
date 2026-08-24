// WWMI-DX12: PSO / shader-hash tracking + subobject cache (M4).
//
// 3DMigoto's ShaderOverride keys on the XXH64 of the shader bytecode.
// In D3D12 the bytecode is embedded in the PSO, so ReShade's
// init_pipeline event is the single place both are visible:
//
//   init_pipeline -> hash every shader subobject (vs/ps/cs/...)
//                 -> optionally deep-copy the subobject array for
//                    later PSO cloning (blend override, m4-3)
//   bind_pipeline -> per command-list "current pipeline" (addon_main)
//   draw          -> current pipeline's stage hashes -> ShaderOverride
//                    rule match (m4-2)
//
// Subobject cache notes:
//  - small arrays (input_layout, render_target_formats, dynamic states,
//    stream output) are deep-copied; the shader bytecode pointer is
//    *borrowed* on the assumption the game's shader archive outlives
//    the PSO (UE keeps shader code resident).
//  - entries are LRU-capped (default 8192) to bound memory; eviction
//    only drops OUR clone bookkeeping, never a game object.
//  - the cache can be disabled entirely (no mod needs it) in which
//    case only the hash table is maintained.
#pragma once

#include "reshade_api_pipeline.hpp"

#include <cstddef>
#include <cstdint>
#include <list>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace wwmi
{
	using reshade::api::pipeline_subobject;
	using reshade::api::pipeline_subobject_type;

	// Per-stage shader hashes of one pipeline.
	struct PipelineShaders
	{
		static constexpr uint32_t stage_count = 6; // vs ps gs ds hs cs

		uint64_t vs = 0, ps = 0, gs = 0, ds = 0, hs = 0, cs = 0;
		bool has_vs = false, has_ps = false, has_gs = false;
		bool has_ds = false, has_hs = false, has_cs = false;
	};

	class PipelineTracker
	{
	public:
		// max_cached: LRU cap for the clone-source cache. 0 disables the
		// subobject cache (hashes are still tracked).
		explicit PipelineTracker(size_t max_cached = 8192)
			: _max_cached(max_cached) {}

		~PipelineTracker() = default;
		PipelineTracker(const PipelineTracker &) = delete;
		PipelineTracker &operator=(const PipelineTracker &) = delete;

		// ReShade addon_event::init_pipeline. Reads (and optionally
		// deep-copies) the subobjects. Thread-safe.
		void on_init_pipeline(uint64_t layout, uint32_t count,
			const pipeline_subobject *subobjects, uint64_t pipeline);

		void on_destroy_pipeline(uint64_t pipeline);

		const PipelineShaders *find(uint64_t pipeline) const;

		// Clone source access: returns the cached subobject array (owned
		// by the tracker) or nullptr when not cached / evicted.
		const std::vector<pipeline_subobject> *clone_source(uint64_t pipeline);
		uint64_t clone_source_layout(uint64_t pipeline) const;

		size_t cached_count() const { return _by_handle.size(); }
		size_t hash_count() const { return _hashes.size(); }

	private:
		struct Entry
		{
			PipelineShaders shaders;
			std::vector<pipeline_subobject> subobjects; // deep-copied
			uint64_t layout = 0;
		};

		void touch(uint64_t pipeline);

		const size_t _max_cached;
		mutable std::mutex _lock;
		mutable std::list<uint64_t> _lru; // front = most recent
		std::unordered_map<uint64_t, std::list<uint64_t>::iterator> _lru_pos;
		std::unordered_map<uint64_t, Entry> _by_handle;   // cached (clone-able)
		std::unordered_map<uint64_t, PipelineShaders> _hashes; // every PSO

		// Frees the arrays the subobject copies point into. The entry in
		// _by_handle must be erased by the caller.
		static void free_subobject_storage(pipeline_subobject *subs, uint32_t count);
	};
}

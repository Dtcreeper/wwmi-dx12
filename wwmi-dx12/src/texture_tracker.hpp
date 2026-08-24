// WWMI-DX12: runtime texture tracking + prefilter + learned hashes (m1-8).
//
// Core bookkeeping for the hunting/matching pipeline, deliberately free of
// any ReShade dependency so it is unit-testable:
//
//   TextureTracker  pool of candidate 2D textures with hash state machine
//                   (pending -> learning -> done/unsupported), LRU-bounded.
//   HashIndex       3DMigoto texture hash -> TextureOverride rule lookup.
//   SessionCache    learned DX12-hash -> mod-hash pairings, persisted as a
//                   small JSON file so hunting survives restarts.
//
// The addon bridge (addon_main.cpp) feeds ReShade events into these and
// implements the GPU readback used to learn hashes.
#pragma once

#include "mod_rules.hpp"

#include <cstdint>
#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

namespace wwmi
{

	struct TrackerConfig
	{
		// Prefilter: appearance mods ship with reasonably sized diffuse /
		// normal / detail textures; tiny textures are UI noise and hashing
		// them wastes readback bandwidth.
		uint32_t min_width = 256;
		uint32_t min_height = 256;
		uint32_t max_dimension = 16384;

		// Pool bound: evict least-recently-used unhashed entries beyond this.
		size_t max_tracked = 8192;

		// How many textures per frame the readback hasher may process.
		uint32_t hash_budget_per_frame = 2;
	};

	enum class HashState : uint8_t
	{
		pending,     // tracked, hash not learned yet
		learning,    // readback in flight (reserved by the bridge)
		done,        // data+desc hash known
		unsupported, // readback/hash failed; do not retry
	};

	struct TrackedTexture
	{
		uint64_t handle = 0;   // ReShade api::resource handle
		uint32_t width = 0;
		uint32_t height = 0;
		uint32_t format = 0;   // DXGI_FORMAT
		uint16_t levels = 1;
		uint16_t layers = 1;   // depth_or_layers (cubemap = 6)

		HashState hash_state = HashState::pending;
		uint32_t data_hash = 0; // 3DMigoto CalcTexture2DDataHash
		uint32_t desc_hash = 0; // 3DMigoto texture hash (what mods match on)

		uint64_t frame_seen = 0; // LRU tick / stats
	};

	class TextureTracker
	{
	public:
		explicit TextureTracker(TrackerConfig cfg = {}) : _cfg(cfg) {}

		const TrackerConfig &config() const { return _cfg; }

		// Prefilter shared by the bridge: must be a 2D, gpu-only, sampleable
		// texture within the configured size window.
		bool should_track(uint32_t width, uint32_t height) const;

		// Registers a texture. Returns nullptr when the pool is full and the
		// entry cannot be admitted (all entries are more valuable), or when
		// the handle already exists (then find() has it).
		TrackedTexture *track(uint64_t handle, uint32_t width, uint32_t height,
			uint32_t format, uint16_t levels, uint16_t layers, uint64_t frame);

		void untrack(uint64_t handle);

		TrackedTexture *find(uint64_t handle);
		const TrackedTexture *find(uint64_t handle) const;

		// Records a learned hash; transitions state to done.
		void set_hash(uint64_t handle, uint32_t data_hash, uint32_t desc_hash);

		// Marks a texture as unsupported (readback failed etc.).
		void set_unsupported(uint64_t handle);

		// Picks up to <budget> pending textures, oldest first, transitioning
		// them to 'learning' so the bridge can hash exactly these. Returns
		// SNAPSHOTS (by value): the bridge does the GPU readback outside the
		// runtime lock, where tracker entries may be evicted or untracked.
		std::vector<TrackedTexture> pick_pending(size_t budget);

		// Returns a 'learning' entry to 'pending' with a fresh LRU tick.
		// Used when the bridge defers a readback (e.g. unsafe present-time
		// resource state) and wants to retry later, not fail permanently.
		void requeue(uint64_t handle);

		// Un-learns everything but keeps tracking (used when the hash
		// pipeline is reset, e.g. device lost).
		void reset_hashes();

		void clear();

		size_t size() const { return _by_handle.size(); }

		// Live entries per hash state (overlay snapshot; O(n), n <= max_tracked).
		struct StateCounts
		{
			size_t pending = 0;
			size_t learning = 0;
			size_t done = 0;
			size_t unsupported = 0;
		};
		StateCounts count_by_state() const;

		struct Stats
		{
			uint64_t tracked = 0;    // admitted total
			uint64_t rejected = 0;   // failed prefilter
			uint64_t evicted = 0;    // dropped by LRU
			uint64_t hashed = 0;     // completed hashes
			uint64_t unsupported = 0;
		};
		const Stats &stats() const { return _stats; }

	private:
		// Evicts the oldest entry without a learned hash. Returns false when
		// every entry holds a hash (caller admits over-capacity instead).
		bool evict_one();

		TrackerConfig _cfg;
		Stats _stats;
		std::unordered_map<uint64_t, TrackedTexture> _by_handle;
		uint64_t _tick = 0; // admission counter for LRU fairness
	};

	// -----------------------------------------------------------------------

	// 3DMigoto texture hash -> rule index over a loaded mod set. Texture
	// hashes are the low 32 bits of the ini hash token (parse_hash).
	//
	// M2: a hash may be claimed by several TextureOverride sections (e.g.
	// one 'handling = skip' rule plus one binding rule for the same mesh).
	// 3DMigoto keys its overrides in a multimap and activates every match,
	// so all rules are kept in load order; find() keeps returning the
	// first for the M1 replacement path, find_all()/find_draw_rules()
	// expose the rest.
	class HashIndex
	{
	public:
		// Takes ownership of the override rules from one or more ModRules
		// loads (rules keep their mod directory for diagnostics later).
		void build(std::vector<TextureOverrideRule> rules);

		// First rule registered for this hash (texture-replacement path).
		const TextureOverrideRule *find(uint32_t texture_hash) const;

		// All rules registered for this hash, in load order.
		void find_all(uint32_t texture_hash, std::vector<const TextureOverrideRule *> &out) const;

		// Rules for this hash that participate in draw interception
		// (handling = skip/abort), in load order. Cheap when empty: the
		// draw event handler consults has_draw_rules() first.
		void find_draw_rules(uint32_t texture_hash, std::vector<const TextureOverrideRule *> &out) const;

		// False when no loaded mod intercepts draws at all; lets the
		// bind/draw event handlers early out without hash lookups.
		bool has_draw_rules() const { return !_draw_by_hash.empty(); }

		size_t size() const { return _rules.size(); }
		uint32_t collision_count() const { return _collisions; }
		size_t draw_rule_count() const { return _draw_rule_count; }

		// Immutable after build(): read-only view for the overlay panel.
		const std::vector<TextureOverrideRule> &rules() const { return _rules; }

		void clear();

	private:
		std::vector<TextureOverrideRule> _rules;
		std::unordered_map<uint32_t, std::vector<size_t>> _by_hash; // hash -> rule indices (load order)
		std::unordered_map<uint32_t, std::vector<size_t>> _draw_by_hash; // handling != none
		uint32_t _collisions = 0; // hashes claimed by more than one rule
		size_t _draw_rule_count = 0;
	};

	// -----------------------------------------------------------------------

	// Learned pairings between a hash observed at runtime (the DX12 texture
	// hash our hasher computes) and the mod-side hash a TextureOverride
	// matches on. Persisted as JSON so a hunting session carries over.
	class SessionCache
	{
	public:
		// runtime hash -> mod hash
		void pair(uint64_t runtime_hash, uint32_t mod_hash);
		bool lookup(uint64_t runtime_hash, uint32_t *out_mod_hash) const;
		size_t size() const { return _pairs.size(); }

		bool save(const std::filesystem::path &path) const;
		bool load(const std::filesystem::path &path);

		void clear() { _pairs.clear(); }

	private:
		std::unordered_map<uint64_t, uint32_t> _pairs;
	};

} // namespace wwmi

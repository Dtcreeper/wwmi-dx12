// WWMI-DX12: runtime VB/IB tracking + per-command-list IA state (M2).
//
// ReShade's D3D12 backend reports one generic resource_desc for buffers
// with every usage flag set, so the VB/IB role (and with it the D3D11
// bind flags in the 3DMigoto hash) only becomes known when the buffer is
// bound to the input assembler. BufferTracker therefore admits buffers
// lazily from the bind events:
//
//   bind_index_buffer / bind_vertex_buffers
//     -> bridge resolves the buffer size (device get_resource_desc)
//     -> BufferTracker::track(handle, byte_width, role, frame)
//     -> readback learns data_hash asynchronously (budgeted per frame)
//     -> full 3DMigoto hash = calc_buffer_hash(data_hash, width, role)
//
// IaState snapshots the IA bindings of one command-list recording:
// D3D12 IA state persists across draws within a recording and every
// DrawIndexed* requires a fresh IB bind after Reset, so keying by the
// command-list handle is sound (bundles are separate recordings and get
// their own entry).
//
// find_skip_rule() ties it together: given the current IaState and the
// DrawCallInfo of the draw about to happen, it looks up draw-intercepting
// rules by the bound IB hash (indexed draws) and the VB slot 0 hash, and
// returns the rule whose handling skips the draw.
#pragma once

#include "buffer_hash.hpp"
#include "texture_tracker.hpp" // HashState, HashIndex

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace wwmi
{

	struct BufferTrackerConfig
	{
		// Readback cost window: buffers below/above these sizes are not
		// hashed (IA-bound buffers outside this range are noise or
		// prohibitively expensive to read back).
		uint64_t min_bytes = 16;
		uint64_t max_bytes = 64ull * 1024 * 1024;

		// Pool bound: evict least-recently-used unhashed entries beyond this.
		size_t max_tracked = 4096;

		// How many buffers per frame the readback hasher may process.
		// Large index buffers are picked first (see pick_pending), so a
		// budget of 4 learns a body IB within seconds even behind a
		// flood of pending vertex buffers.
		uint32_t hash_budget_per_frame = 4;
	};

	struct TrackedBuffer
	{
		uint64_t handle = 0;    // ReShade api::resource handle
		uint64_t byte_width = 0;
		BufferRole role = BufferRole::unknown; // first bind wins

		HashState hash_state = HashState::pending;
		uint32_t data_hash = 0; // crc32c over readback contents
		uint32_t hash = 0;      // full 3DMigoto buffer hash (mods match on)

		uint64_t frame_seen = 0;  // LRU tick
		uint64_t bind_count = 0;  // bind events observed (priority/stats)
	};

	class BufferTracker
	{
	public:
		explicit BufferTracker(BufferTrackerConfig cfg = {}) : _cfg(cfg) {}

		const BufferTrackerConfig &config() const { return _cfg; }

		bool should_track(uint64_t byte_width) const;

		// Lazy admission from a bind event. An existing entry is refreshed
		// (role/width from the first admission are kept: UE never binds a
		// mesh buffer under two roles). Returns nullptr when rejected by
		// the size window or the pool is full.
		TrackedBuffer *track(uint64_t handle, uint64_t byte_width,
			BufferRole role, uint64_t frame);

		void untrack(uint64_t handle);

		TrackedBuffer *find(uint64_t handle);
		const TrackedBuffer *find(uint64_t handle) const;

		// Records a learned hash; transitions state to done.
		void set_hash(uint64_t handle, uint32_t data_hash, uint32_t hash);

		// Marks a buffer as unsupported (readback failed etc.).
		void set_unsupported(uint64_t handle);

		// Picks up to <budget> pending buffers, oldest first, transitioning
		// them to 'learning'. Returns SNAPSHOTS (by value): the bridge does
		// the GPU readback outside the runtime lock.
		std::vector<TrackedBuffer> pick_pending(size_t budget);

		void reset_hashes();
		void clear();

		size_t size() const { return _by_handle.size(); }

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
			uint64_t tracked = 0;
			uint64_t rejected = 0;
			uint64_t evicted = 0;
			uint64_t hashed = 0;
			uint64_t unsupported = 0;
			uint64_t binds = 0; // bind events seen (tracked + rejected)
		};
		const Stats &stats() const { return _stats; }

	private:
		bool evict_one();

		BufferTrackerConfig _cfg;
		Stats _stats;
		std::unordered_map<uint64_t, TrackedBuffer> _by_handle;
	};

	// -----------------------------------------------------------------------

	// Input-assembler snapshot of one command-list recording.
	//
	// NOTE: named IaState, not DrawState -- winuser.h (via Windows.h, which
	// reshade.hpp pulls in) #defines DrawState to DrawStateA/W, which would
	// rename the struct per translation unit and break linking.
	struct IaState
	{
		static constexpr uint32_t max_vb_slots = 32; // D3D12 IA slot count

		struct VertexBinding
		{
			uint64_t buffer = 0; // 0 = unbound
			uint64_t offset = 0;
			uint32_t stride = 0;
		};

		VertexBinding vbs[max_vb_slots] = {};
		uint32_t vb_valid_mask = 0; // bit i set = slot i carries a buffer

		uint64_t ib_buffer = 0;    // 0 = no index buffer bound
		uint64_t ib_offset = 0;
		uint32_t ib_index_size = 0; // 2 or 4 bytes; 0 with a bound buffer

		// Replays bind_vertex_buffers: slots [first, first+count) are
		// replaced; a null buffer clears its slot.
		void bind_vertex_buffers(uint32_t first, uint32_t count,
			const uint64_t *buffers, const uint64_t *offsets, const uint32_t *strides);

		// Replays bind_index_buffer (buffer 0 unbinds).
		void bind_index_buffer(uint64_t buffer, uint64_t offset, uint32_t index_size);

		// Slot 0 binding, or nullptr when slot 0 is unbound.
		const VertexBinding *vb0() const
		{
			return (vb_valid_mask & 1u) ? &vbs[0] : nullptr;
		}
	};

	// -----------------------------------------------------------------------

	class IndexViewTracker; // M6 (defined below)

	// Draw-time skip decision (pure; unit-tested without ReShade).
	//
	// Evaluates draw-intercepting rules for the draw described by <call>
	// against the IA state in <state>, looking the rules up by the bound
	// index-buffer hash (indexed draws) and the vertex-buffer slot 0 hash
	// (both draw types). A rule only skips the draw when its handling is
	// skip/abort AND matches_draw_info() passes. Returns the rule that
	// requested the skip, or nullptr when the draw must run.
	//
	// M6: when <views> is provided, indexed draws first consult the
	// pooled-IB view table -- UE DX12 suballocates mesh index buffers
	// inside large pools, so the DX11 per-mesh hash can only be
	// reproduced by hashing the draw-window region of the pool
	// (IndexViewTracker below).
	const TextureOverrideRule *find_skip_rule(const HashIndex &index,
		const BufferTracker &buffers, const IndexViewTracker *views,
		const IaState &state, const DrawCallInfo &call, bool indexed);

	// -----------------------------------------------------------------------
	// M6: pooled index-buffer views (UE DX12 suballocation)
	//
	// The game allocates mesh index buffers inside shared 32 MB pools
	// (observed: every learned IB is exactly 33554432 bytes). A mod's
	// DX11-era hash references a DEDICATED per-mesh buffer, so hashing
	// the whole pool can never match. Instead the mod's own rules tell
	// us the mesh's index range: the component rules sharing one hash
	// carry (match_first_index, match_index_count) windows that tile the
	// original buffer exactly (0 -> 17970 -> 63606 -> ... contiguous).
	//
	// DrawWindowIndex extracts those groups at load time. At draw time a
	// window match on an unverified pool view schedules a region probe;
	// the probe reads back [view_offset + min_first_index*index_size,
	// span) and hashes it with the ordinary 3DMigoto formula (desc
	// ByteWidth = span). A verified view then routes draws exactly like
	// a learned whole-buffer hash.
	// -----------------------------------------------------------------------

	struct DrawWindow
	{
		uint32_t first_index = 0;
		uint32_t index_count = 0;
	};

	struct DrawRuleGroup
	{
		uint32_t hash = 0;              // the shared rule hash (mod key)
		uint32_t min_first_index = 0;   // window start (region base)
		uint32_t max_end_index = 0;     // last window end (region size /is)
		std::vector<DrawWindow> windows; // (first_index, count) pairs
		std::string section;            // first rule's name (diagnostics)
	};

	// Groups draw-intercepting rules by hash where every grouped rule
	// carries both match_first_index and match_index_count. Rules
	// without windows are ignored (they use the whole-buffer path).
	class DrawWindowIndex
	{
	public:
		void build(const std::vector<TextureOverrideRule> &rules);

		bool empty() const { return _groups.empty(); }

		// Group whose window list contains (first_index, index_count),
		// or nullptr. Small window counts (mesh mods use < 32): linear
		// scan is cheaper than a map on every indexed draw.
		const DrawRuleGroup *find_by_draw(uint32_t first_index, uint32_t index_count) const;

		const std::vector<DrawRuleGroup> &groups() const { return _groups; }
		std::vector<DrawRuleGroup> &groups() { return _groups; }

	private:
		std::vector<DrawRuleGroup> _groups;
	};

	// One (pool buffer, byte offset) index-buffer view and its
	// verification state. 'verified' views hash-match like learned
	// whole buffers; 'failed' views re-arm for a probe after a cooldown
	// (pool regions get recycled between scenes).
	struct TrackedIndexView
	{
		uint64_t handle = 0;   // pool resource handle
		uint64_t offset = 0;   // view byte offset inside the pool
		uint32_t index_size = 0;

		enum class State : uint8_t { unverified, probing, verified, failed };
		State state = State::unverified;
		uint32_t hash = 0;             // learned virtual hash (verified)
		uint32_t expected_hash = 0;    // group hash the probe compares to
		uint64_t frame_seen = 0;       // LRU tick
		uint64_t probe_frame = 0;      // last probe issue (cooldown)
		uint32_t retries = 0;
	};

	class IndexViewTracker
	{
	public:
		static constexpr uint64_t retry_cooldown_frames = 300;

		// Returns the view entry for (handle, offset), creating an
		// unverified one if absent (LRU-bounded). Nullptr only for a
		// null handle.
		TrackedIndexView *track(uint64_t handle, uint64_t offset,
			uint32_t index_size, uint64_t frame);

		TrackedIndexView *find(uint64_t handle, uint64_t offset);
		const TrackedIndexView *find(uint64_t handle, uint64_t offset) const;

		void set_verified(uint64_t handle, uint64_t offset, uint32_t hash);
		void set_failed(uint64_t handle, uint64_t offset);

		// Drops all view entries (device destroyed: handles are
		// device-specific). Cumulative stats are kept.
		void clear() { _views.clear(); }

		// True when the view may be probed now (unverified, or failed
		// past the cooldown).
		bool probe_due(const TrackedIndexView &v, uint64_t frame) const;

		size_t size() const { return _views.size(); }

		struct Stats
		{
			uint64_t tracked = 0;
			uint64_t verified = 0;
			uint64_t failed = 0;
			uint64_t evicted = 0;
		};
		const Stats &stats() const { return _stats; }

	private:
		bool evict_one();
		Stats _stats;
		std::unordered_map<uint64_t, TrackedIndexView> _views; // key: handle ^ (offset<<1)
		static constexpr size_t k_max_views = 512;
	};

}

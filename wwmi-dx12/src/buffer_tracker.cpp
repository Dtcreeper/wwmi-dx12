// WWMI-DX12: buffer tracking implementation (see buffer_tracker.hpp).
#include "buffer_tracker.hpp"

#include <algorithm>

namespace wwmi
{

	// ---- BufferTracker -----------------------------------------------------

	bool BufferTracker::should_track(uint64_t byte_width) const
	{
		if (byte_width < _cfg.min_bytes || byte_width > _cfg.max_bytes)
			return false;
		return true;
	}

	TrackedBuffer *BufferTracker::track(uint64_t handle, uint64_t byte_width,
		BufferRole role, uint64_t frame)
	{
		if (handle == 0)
			return nullptr;
		++_stats.binds;

		auto it = _by_handle.find(handle);
		if (it != _by_handle.end())
		{
			// Refresh LRU; role/width stay from the first admission.
			it->second.frame_seen = frame;
			++it->second.bind_count;
			return &it->second;
		}

		if (!should_track(byte_width))
		{
			++_stats.rejected;
			return nullptr;
		}

		// Same admission policy as TextureTracker: evict oldest entries
		// without a learned hash; never evict done/learning entries.
		while (_by_handle.size() >= _cfg.max_tracked)
		{
			if (!evict_one())
				break;
		}

		TrackedBuffer &b = _by_handle[handle];
		b.handle = handle;
		b.byte_width = byte_width;
		b.role = role;
		b.hash_state = HashState::pending;
		b.data_hash = 0;
		b.hash = 0;
		b.frame_seen = frame;
		b.bind_count = 1;
		++_stats.tracked;
		return &b;
	}

	bool BufferTracker::evict_one()
	{
		uint64_t best_handle = 0;
		uint64_t best_frame = UINT64_MAX;
		for (const auto &[handle, b] : _by_handle)
		{
			if (b.hash_state == HashState::done || b.hash_state == HashState::learning)
				continue;
			if (b.frame_seen < best_frame)
			{
				best_frame = b.frame_seen;
				best_handle = handle;
			}
		}
		if (best_handle == 0)
			return false;

		_by_handle.erase(best_handle);
		++_stats.evicted;
		return true;
	}

	void BufferTracker::untrack(uint64_t handle)
	{
		_by_handle.erase(handle);
	}

	TrackedBuffer *BufferTracker::find(uint64_t handle)
	{
		auto it = _by_handle.find(handle);
		return it != _by_handle.end() ? &it->second : nullptr;
	}

	const TrackedBuffer *BufferTracker::find(uint64_t handle) const
	{
		auto it = _by_handle.find(handle);
		return it != _by_handle.end() ? &it->second : nullptr;
	}

	void BufferTracker::set_hash(uint64_t handle, uint32_t data_hash, uint32_t hash)
	{
		auto it = _by_handle.find(handle);
		if (it == _by_handle.end())
			return;
		it->second.data_hash = data_hash;
		it->second.hash = hash;
		it->second.hash_state = HashState::done;
		++_stats.hashed;
	}

	void BufferTracker::set_unsupported(uint64_t handle)
	{
		auto it = _by_handle.find(handle);
		if (it == _by_handle.end())
			return;
		if (it->second.hash_state != HashState::unsupported)
		{
			it->second.hash_state = HashState::unsupported;
			++_stats.unsupported;
		}
	}

	std::vector<TrackedBuffer> BufferTracker::pick_pending(size_t budget)
	{
		std::vector<TrackedBuffer> picked;
		if (budget == 0)
			return picked;

		std::vector<TrackedBuffer *> pending;
	pending.reserve(16);
	for (auto &[handle, b] : _by_handle)
		if (b.hash_state == HashState::pending)
			pending.push_back(&b);

	// Large index buffers jump the queue: mesh-mod draw interception keys
	// on the body IB hash, and IBs >= 256 KB are rare, so they must not
	// wait behind a flood of small vertex buffers. Within each class the
	// oldest bind still goes first.
	constexpr uint64_t k_large_ib_bytes = 256ull * 1024;
	std::sort(pending.begin(), pending.end(), [](const TrackedBuffer *a, const TrackedBuffer *b) {
		const bool a_large = a->role == BufferRole::index && a->byte_width >= k_large_ib_bytes;
		const bool b_large = b->role == BufferRole::index && b->byte_width >= k_large_ib_bytes;
		if (a_large != b_large)
			return a_large; // large IBs first
		return a->frame_seen < b->frame_seen;
	});

		for (TrackedBuffer *b : pending)
		{
			b->hash_state = HashState::learning;
			picked.push_back(*b); // snapshot: caller works outside the lock
			if (picked.size() >= budget)
				break;
		}
		return picked;
	}

	void BufferTracker::reset_hashes()
	{
		for (auto &[handle, b] : _by_handle)
		{
			b.hash_state = HashState::pending;
			b.data_hash = 0;
			b.hash = 0;
		}
		_stats.hashed = 0;
		_stats.unsupported = 0;
	}

	BufferTracker::StateCounts BufferTracker::count_by_state() const
	{
		StateCounts c;
		for (const auto &[handle, b] : _by_handle)
		{
			switch (b.hash_state)
			{
			case HashState::pending: ++c.pending; break;
			case HashState::learning: ++c.learning; break;
			case HashState::done: ++c.done; break;
			case HashState::unsupported: ++c.unsupported; break;
			}
		}
		return c;
	}

	void BufferTracker::clear()
	{
		_by_handle.clear();
		_stats = Stats{};
	}

	// ---- IaState -----------------------------------------------------------

	void IaState::bind_vertex_buffers(uint32_t first, uint32_t count,
		const uint64_t *buffers, const uint64_t *offsets, const uint32_t *strides)
	{
		for (uint32_t i = 0; i < count; ++i)
		{
			const uint32_t slot = first + i;
			if (slot >= max_vb_slots)
				break; // defensive: D3D12 caps IA slots at 32
			if (buffers[i] == 0)
			{
				vbs[slot] = VertexBinding{};
				vb_valid_mask &= ~(1u << slot);
			}
			else
			{
				vbs[slot].buffer = buffers[i];
				vbs[slot].offset = offsets ? offsets[i] : 0;
				vbs[slot].stride = strides ? strides[i] : 0;
				vb_valid_mask |= (1u << slot);
			}
		}
	}

	void IaState::bind_index_buffer(uint64_t buffer, uint64_t offset, uint32_t index_size)
	{
		ib_buffer = buffer;
		ib_offset = buffer ? offset : 0;
		ib_index_size = buffer ? index_size : 0;
	}

	// ---- find_skip_rule ------------------------------------------------------

	namespace
	{
		// Rules keyed by <handle>'s learned hash. Returns the rule that
		// skips this draw, or nullptr (hash not learned / no rule match).
		const TextureOverrideRule *check_buffer_rules(const HashIndex &index,
			const BufferTracker &buffers, const DrawCallInfo &call, uint64_t handle,
			std::vector<const TextureOverrideRule *> &scratch)
		{
			const TrackedBuffer *b = buffers.find(handle);
			if (b == nullptr || b->hash_state != HashState::done || b->hash == 0)
				return nullptr; // not learned yet: draw passes through

			scratch.clear();
			index.find_draw_rules(b->hash, scratch);
			for (const TextureOverrideRule *r : scratch)
				if (r->handling != HandlingMode::none && matches_draw_info(*r, call))
					return r;
			return nullptr;
		}

		// Same rule evaluation for a verified pool view's virtual hash.
		const TextureOverrideRule *check_view_rules(const HashIndex &index,
			const DrawCallInfo &call, uint32_t hash,
			std::vector<const TextureOverrideRule *> &scratch)
		{
			scratch.clear();
			index.find_draw_rules(hash, scratch);
			for (const TextureOverrideRule *r : scratch)
				if (r->handling != HandlingMode::none && matches_draw_info(*r, call))
					return r;
			return nullptr;
		}
	}

	const TextureOverrideRule *find_skip_rule(const HashIndex &index,
		const BufferTracker &buffers, const IndexViewTracker *views,
		const IaState &state, const DrawCallInfo &call, bool indexed)
	{
		if (!index.has_draw_rules())
			return nullptr;

		std::vector<const TextureOverrideRule *> scratch;

		if (indexed && state.ib_buffer != 0)
		{
			// M6: pooled views first -- a verified view reproduces the
			// DX11 per-mesh hash via region hashing and routes draws
			// exactly like a learned dedicated buffer.
			if (views != nullptr)
			{
				const TrackedIndexView *v = views->find(state.ib_buffer, state.ib_offset);
				if (v != nullptr && v->state == TrackedIndexView::State::verified && v->hash != 0)
				{
					const TextureOverrideRule *r = check_view_rules(index, call, v->hash, scratch);
					if (r != nullptr)
						return r;
				}
			}

			const TextureOverrideRule *r = check_buffer_rules(index, buffers, call, state.ib_buffer, scratch);
			if (r != nullptr)
				return r;
		}

		// Both draw types: rules keyed by the slot-0 vertex buffer.
		if (const IaState::VertexBinding *vb = state.vb0())
		{
			const TextureOverrideRule *r = check_buffer_rules(index, buffers, call, vb->buffer, scratch);
			if (r != nullptr)
				return r;
		}

		return nullptr;
	}

	// ---- DrawWindowIndex (M6) ------------------------------------------------

	void DrawWindowIndex::build(const std::vector<TextureOverrideRule> &rules)
	{
		_groups.clear();

		// Collect windowed draw rules keyed by hash. A rule joins a
		// group only when BOTH matchers are present -- windowless rules
		// keep using the whole-buffer path.
		std::vector<DrawRuleGroup *> by_hash; // parallel probe of _groups
		for (const TextureOverrideRule &rule : rules)
		{
			if (!rule.has_hash || rule.handling == HandlingMode::none)
				continue;
			if (!rule.match_first_index.enabled || !rule.match_index_count.enabled)
				continue;

			DrawRuleGroup *grp = nullptr;
			for (DrawRuleGroup &g : _groups)
				if (g.hash == rule.hash)
				{
					grp = &g;
					break;
				}
			if (grp == nullptr)
			{
				DrawRuleGroup g;
				g.hash = rule.hash;
				g.section = rule.section;
				_groups.push_back(g);
				grp = &_groups.back();
			}

			DrawWindow w;
			w.first_index = rule.match_first_index.value;
			w.index_count = rule.match_index_count.value;
			grp->windows.push_back(w);

			const uint32_t end = w.first_index + w.index_count;
			if (grp->windows.size() == 1)
			{
				grp->min_first_index = w.first_index;
				grp->max_end_index = end;
			}
			else
			{
				if (w.first_index < grp->min_first_index)
					grp->min_first_index = w.first_index;
				if (end > grp->max_end_index)
					grp->max_end_index = end;
			}
		}
	}

	const DrawRuleGroup *DrawWindowIndex::find_by_draw(uint32_t first_index,
		uint32_t index_count) const
	{
		for (const DrawRuleGroup &g : _groups)
			for (const DrawWindow &w : g.windows)
				if (w.first_index == first_index && w.index_count == index_count)
					return &g;
		return nullptr;
	}

	// ---- IndexViewTracker (M6) -----------------------------------------------

	namespace
	{
		uint64_t view_key(uint64_t handle, uint64_t offset)
		{
			// Handles are pointer-aligned (>= 16); mixing the offset
			// into the low bits stays collision-free for real inputs.
			return handle ^ (offset << 1) ^ (offset >> 32);
		}
	}

	TrackedIndexView *IndexViewTracker::track(uint64_t handle, uint64_t offset,
		uint32_t index_size, uint64_t frame)
	{
		if (handle == 0)
			return nullptr;

		const uint64_t key = view_key(handle, offset);
		auto it = _views.find(key);
		if (it != _views.end())
		{
			it->second.frame_seen = frame;
			if (it->second.index_size == 0)
				it->second.index_size = index_size;
			return &it->second;
		}

		while (_views.size() >= k_max_views)
		{
			if (!evict_one())
				break;
		}

		TrackedIndexView &v = _views[key];
		v.handle = handle;
		v.offset = offset;
		v.index_size = index_size;
		v.state = TrackedIndexView::State::unverified;
		v.hash = 0;
		v.expected_hash = 0;
		v.frame_seen = frame;
		v.probe_frame = 0;
		v.retries = 0;
		++_stats.tracked;
		return &v;
	}

	bool IndexViewTracker::evict_one()
	{
		// Verified views are load-bearing (they route draws); only
		// unverified/probing/failed entries are eviction candidates,
		// oldest first.
		uint64_t best_key = 0;
		uint64_t best_frame = UINT64_MAX;
		for (const auto &[key, v] : _views)
		{
			if (v.state == TrackedIndexView::State::verified)
				continue;
			if (v.frame_seen < best_frame)
			{
				best_frame = v.frame_seen;
				best_key = key;
			}
		}
		if (best_key == 0)
		{
			// All entries verified: evict the oldest verified entry
			// anyway -- stale views of destroyed pools must not pin the
			// table.
			for (const auto &[key, v] : _views)
				if (v.frame_seen < best_frame)
				{
					best_frame = v.frame_seen;
					best_key = key;
				}
			if (best_key == 0)
				return false;
		}
		_views.erase(best_key);
		++_stats.evicted;
		return true;
	}

	TrackedIndexView *IndexViewTracker::find(uint64_t handle, uint64_t offset)
	{
		const auto it = _views.find(view_key(handle, offset));
		return it != _views.end() ? &it->second : nullptr;
	}

	const TrackedIndexView *IndexViewTracker::find(uint64_t handle, uint64_t offset) const
	{
		const auto it = _views.find(view_key(handle, offset));
		return it != _views.end() ? &it->second : nullptr;
	}

	void IndexViewTracker::set_verified(uint64_t handle, uint64_t offset, uint32_t hash)
	{
		TrackedIndexView *v = find(handle, offset);
		if (v == nullptr)
			return;
		v->hash = hash;
		v->state = TrackedIndexView::State::verified;
		++_stats.verified;
	}

	void IndexViewTracker::set_failed(uint64_t handle, uint64_t offset)
	{
		TrackedIndexView *v = find(handle, offset);
		if (v == nullptr)
			return;
		v->state = TrackedIndexView::State::failed;
		++v->retries;
		++_stats.failed;
	}

	bool IndexViewTracker::probe_due(const TrackedIndexView &v, uint64_t frame) const
	{
		switch (v.state)
		{
		case TrackedIndexView::State::unverified:
			return true;
		case TrackedIndexView::State::failed:
			return frame - v.probe_frame >= retry_cooldown_frames;
		default:
			return false; // probing / verified
		}
	}

}

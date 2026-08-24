// WWMI-DX12: texture tracking implementation (see texture_tracker.hpp).
#include "texture_tracker.hpp"

#include <algorithm>
#include <cstdio>
#include <fstream>

namespace wwmi
{

	// ---- TextureTracker ----------------------------------------------------

	bool TextureTracker::should_track(uint32_t width, uint32_t height) const
	{
		if (width < _cfg.min_width || height < _cfg.min_height)
			return false;
		if (width > _cfg.max_dimension || height > _cfg.max_dimension)
			return false;
		return true;
	}

	TrackedTexture *TextureTracker::track(uint64_t handle, uint32_t width, uint32_t height,
		uint32_t format, uint16_t levels, uint16_t layers, uint64_t frame)
	{
		if (handle == 0)
			return nullptr;

		auto it = _by_handle.find(handle);
		if (it != _by_handle.end())
		{
			it->second.frame_seen = frame;
			return &it->second;
		}

		if (!should_track(width, height))
		{
			++_stats.rejected;
			return nullptr;
		}

		// Admit: evict LRU entries while at capacity. Entries whose hash is
		// already known are the whole point of the tracker, so only pending /
		// unsupported entries are eviction candidates; if none exist, prefer
		// dropping the oldest pending entry over refusing the newcomer (the
		// newcomer is at least as valuable: it is fresh and unhashed).
		while (_by_handle.size() >= _cfg.max_tracked)
		{
			if (!evict_one())
				break;
		}

		TrackedTexture &t = _by_handle[handle];
		t.handle = handle;
		t.width = width;
		t.height = height;
		t.format = format;
		t.levels = levels;
		t.layers = layers;
		t.hash_state = HashState::pending;
		t.data_hash = 0;
		t.desc_hash = 0;
		t.frame_seen = frame;
		++_stats.tracked;
		return &t;
	}

	bool TextureTracker::evict_one()
	{
		// Oldest entry that carries no learned hash.
		uint64_t best_handle = 0;
		uint64_t best_frame = UINT64_MAX;
		for (const auto &[handle, t] : _by_handle)
		{
			if (t.hash_state == HashState::done || t.hash_state == HashState::learning)
				continue;
			if (t.frame_seen < best_frame)
			{
				best_frame = t.frame_seen;
				best_handle = handle;
			}
		}
		if (best_handle == 0)
			return false; // everything holds a learned hash: admit anyway

		_by_handle.erase(best_handle);
		++_stats.evicted;
		return true;
	}

	void TextureTracker::untrack(uint64_t handle)
	{
		_by_handle.erase(handle);
	}

	TrackedTexture *TextureTracker::find(uint64_t handle)
	{
		auto it = _by_handle.find(handle);
		return it != _by_handle.end() ? &it->second : nullptr;
	}

	const TrackedTexture *TextureTracker::find(uint64_t handle) const
	{
		auto it = _by_handle.find(handle);
		return it != _by_handle.end() ? &it->second : nullptr;
	}

	void TextureTracker::set_hash(uint64_t handle, uint32_t data_hash, uint32_t desc_hash)
	{
		auto it = _by_handle.find(handle);
		if (it == _by_handle.end())
			return;
		it->second.data_hash = data_hash;
		it->second.desc_hash = desc_hash;
		it->second.hash_state = HashState::done;
		++_stats.hashed;
	}

	void TextureTracker::set_unsupported(uint64_t handle)
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

	std::vector<TrackedTexture> TextureTracker::pick_pending(size_t budget)
	{
		std::vector<TrackedTexture> picked;
		if (budget == 0)
			return picked;

		// Oldest-first so a burst of new textures is hashed in admission
		// order and the overlay list stays stable.
		std::vector<TrackedTexture *> pending;
		pending.reserve(16);
		for (auto &[handle, t] : _by_handle)
			if (t.hash_state == HashState::pending)
				pending.push_back(&t);

		std::sort(pending.begin(), pending.end(), [](const TrackedTexture *a, const TrackedTexture *b) {
			return a->frame_seen < b->frame_seen;
		});

		for (TrackedTexture *t : pending)
		{
			t->hash_state = HashState::learning;
			picked.push_back(*t); // snapshot: caller works outside the lock
			if (picked.size() >= budget)
				break;
		}
		return picked;
	}

	void TextureTracker::requeue(uint64_t handle)
	{
		// learning -> pending with a fresh LRU tick: used when the bridge
		// defers a readback (e.g. unsafe present-time resource state). The
		// new tick moves the texture behind the others so one hot texture
		// that is always mid-use cannot starve the hash budget.
		TrackedTexture *t = find(handle);
		if (t == nullptr || t->hash_state != HashState::learning)
			return;
		t->hash_state = HashState::pending;
		t->frame_seen = ++_tick;
	}

	void TextureTracker::reset_hashes()
	{
		for (auto &[handle, t] : _by_handle)
		{
			t.hash_state = HashState::pending;
			t.data_hash = 0;
			t.desc_hash = 0;
		}
		_stats.hashed = 0;
		_stats.unsupported = 0;
	}

	TextureTracker::StateCounts TextureTracker::count_by_state() const
	{
		StateCounts c;
		for (const auto &[handle, t] : _by_handle)
		{
			switch (t.hash_state)
			{
			case HashState::pending: ++c.pending; break;
			case HashState::learning: ++c.learning; break;
			case HashState::done: ++c.done; break;
			case HashState::unsupported: ++c.unsupported; break;
			}
		}
		return c;
	}

	void TextureTracker::clear()
	{
		_by_handle.clear();
		_stats = Stats{};
	}

	// ---- HashIndex ----------------------------------------------------------

	void HashIndex::build(std::vector<TextureOverrideRule> rules)
	{
		clear();
		_rules = std::move(rules);
		_by_hash.reserve(_rules.size() * 2);

		for (size_t i = 0; i < _rules.size(); ++i)
		{
			const TextureOverrideRule &r = _rules[i];
			if (!r.has_hash)
				continue;

			if (!_by_hash[r.hash].empty())
				++_collisions;
			_by_hash[r.hash].push_back(i);

			if (r.handling != HandlingMode::none)
			{
				_draw_by_hash[r.hash].push_back(i);
				++_draw_rule_count;
			}
		}
	}

	const TextureOverrideRule *HashIndex::find(uint32_t texture_hash) const
	{
		auto it = _by_hash.find(texture_hash);
		return (it != _by_hash.end() && !it->second.empty()) ? &_rules[it->second.front()] : nullptr;
	}

	void HashIndex::find_all(uint32_t texture_hash, std::vector<const TextureOverrideRule *> &out) const
	{
		auto it = _by_hash.find(texture_hash);
		if (it == _by_hash.end())
			return;
		for (size_t i : it->second)
			out.push_back(&_rules[i]);
	}

	void HashIndex::find_draw_rules(uint32_t texture_hash, std::vector<const TextureOverrideRule *> &out) const
	{
		auto it = _draw_by_hash.find(texture_hash);
		if (it == _draw_by_hash.end())
			return;
		for (size_t i : it->second)
			out.push_back(&_rules[i]);
	}

	void HashIndex::clear()
	{
		_rules.clear();
		_by_hash.clear();
		_draw_by_hash.clear();
		_collisions = 0;
		_draw_rule_count = 0;
	}

	// ---- SessionCache -------------------------------------------------------

	void SessionCache::pair(uint64_t runtime_hash, uint32_t mod_hash)
	{
		_pairs[runtime_hash] = mod_hash;
	}

	bool SessionCache::lookup(uint64_t runtime_hash, uint32_t *out_mod_hash) const
	{
		auto it = _pairs.find(runtime_hash);
		if (it == _pairs.end())
			return false;
		if (out_mod_hash)
			*out_mod_hash = it->second;
		return true;
	}

	bool SessionCache::save(const std::filesystem::path &path) const
	{
		std::error_code ec;
		auto parent = path.parent_path();
		if (!parent.empty())
			std::filesystem::create_directories(parent, ec);

		std::ofstream out(path, std::ios::out | std::ios::trunc);
		if (!out)
			return false;

		out << "{\n  \"version\": 1,\n  \"pairs\": [\n";
		bool first = true;
		for (const auto &[runtime, mod] : _pairs)
		{
			if (!first)
				out << ",\n";
			first = false;
			char line[96];
			std::snprintf(line, sizeof(line), "    { \"runtime\": %llu, \"mod\": %u }",
				static_cast<unsigned long long>(runtime), mod);
			out << line;
		}
		out << "\n  ]\n}\n";
		return static_cast<bool>(out);
	}

	bool SessionCache::load(const std::filesystem::path &path)
	{
		std::ifstream in(path);
		if (!in)
			return false;

		// Tiny hand parser: the file is only ever written by save() above.
		_pairs.clear();
		std::string content((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());

		size_t pos = 0;
		while ((pos = content.find('{', pos)) != std::string::npos)
		{
			const size_t end = content.find('}', pos);
			if (end == std::string::npos)
				break;

			const std::string entry = content.substr(pos, end - pos);
			pos = end + 1;

			const size_t r = entry.find("\"runtime\"");
			const size_t m = entry.find("\"mod\"");
			if (r == std::string::npos || m == std::string::npos)
				continue; // the outer '{' of the document: skip

			const unsigned long long runtime = std::strtoull(entry.c_str() + entry.find(':', r) + 1, nullptr, 10);
			const uint32_t mod = static_cast<uint32_t>(
				std::strtoul(entry.c_str() + entry.find(':', m) + 1, nullptr, 10));
			_pairs[runtime] = mod;
		}
		return true;
	}

} // namespace wwmi

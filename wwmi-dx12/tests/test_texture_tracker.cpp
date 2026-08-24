// WWMI-DX12: texture tracker / hash index / session cache unit tests.
#include "test_framework.hpp"
#include "texture_hash.hpp"
#include "texture_tracker.hpp"

#include <dxgiformat.h>

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <vector>

namespace
{

	wwmi::TrackerConfig small_pool_cfg()
	{
		wwmi::TrackerConfig cfg;
		cfg.min_width = 256;
		cfg.min_height = 256;
		cfg.max_tracked = 4;
		return cfg;
	}

	wwmi::TrackedTexture *add(wwmi::TextureTracker &tr, uint64_t handle,
		uint32_t w, uint32_t h, uint64_t frame, uint32_t fmt = DXGI_FORMAT_BC7_UNORM)
	{
		return tr.track(handle, w, h, fmt, 1, 1, frame);
	}

} // namespace

// ---- prefilter ----

WWMI_TEST(tracker_prefilter_size_window)
{
	wwmi::TextureTracker tr(small_pool_cfg());

	EXPECT(tr.should_track(256, 256));   // lower bound inclusive
	EXPECT(tr.should_track(1920, 1080));
	EXPECT(tr.should_track(16384, 16384)); // upper bound inclusive
	EXPECT(!tr.should_track(255, 1024));   // too narrow
	EXPECT(!tr.should_track(1024, 255));   // too short
	EXPECT(!tr.should_track(16385, 512));  // too wide

	// track() applies the same prefilter and counts rejections.
	EXPECT(add(tr, 1, 64, 64, 1) == nullptr);
	EXPECT_EQ(tr.stats().rejected, 1u);
	EXPECT(add(tr, 2, 512, 512, 1) != nullptr);
	EXPECT_EQ(tr.stats().rejected, 1u);
}

// ---- track / find / untrack ----

WWMI_TEST(tracker_track_find_untrack)
{
	wwmi::TextureTracker tr(small_pool_cfg());

	wwmi::TrackedTexture *t = add(tr, 0xAABB, 1024, 1024, 7);
	EXPECT(t != nullptr);
	EXPECT_EQ(t->handle, 0xAABBu);
	EXPECT_EQ(t->width, 1024u);
	EXPECT_EQ(t->height, 1024u);
	EXPECT_EQ(t->format, static_cast<uint32_t>(DXGI_FORMAT_BC7_UNORM));
	EXPECT_EQ(t->hash_state, wwmi::HashState::pending);

	EXPECT(tr.find(0xAABB) == t);
	EXPECT(tr.find(0xDEAD) == nullptr);
	EXPECT_EQ(tr.size(), 1u);

	// Re-tracking the same handle refreshes, does not duplicate.
	wwmi::TrackedTexture *t2 = add(tr, 0xAABB, 1024, 1024, 9);
	EXPECT(t2 == t);
	EXPECT_EQ(tr.size(), 1u);
	EXPECT_EQ(t->frame_seen, 9u);
	EXPECT_EQ(tr.stats().tracked, 1u);

	tr.untrack(0xAABB);
	EXPECT(tr.find(0xAABB) == nullptr);
	EXPECT_EQ(tr.size(), 0u);

	// Unknown handles are no-ops.
	tr.untrack(0x1234);
	EXPECT_EQ(tr.size(), 0u);
}

// ---- hash state machine ----

WWMI_TEST(tracker_hash_state_transitions)
{
	wwmi::TextureTracker tr(small_pool_cfg());
	add(tr, 0x1, 512, 512, 1);

	EXPECT_EQ(tr.find(0x1)->hash_state, wwmi::HashState::pending);
	tr.set_hash(0x1, 0xDEADBEEF, 0x11223344);
	EXPECT_EQ(tr.find(0x1)->hash_state, wwmi::HashState::done);
	EXPECT_EQ(tr.find(0x1)->data_hash, 0xDEADBEEFu);
	EXPECT_EQ(tr.find(0x1)->desc_hash, 0x11223344u);
	EXPECT_EQ(tr.stats().hashed, 1u);

	// Unknown handle / unsupported transitions are safe.
	tr.set_hash(0x99, 1, 2);
	tr.set_unsupported(0x99);
	EXPECT_EQ(tr.stats().unsupported, 0u);

	add(tr, 0x2, 512, 512, 2);
	tr.set_unsupported(0x2);
	EXPECT_EQ(tr.find(0x2)->hash_state, wwmi::HashState::unsupported);
	tr.set_unsupported(0x2); // idempotent
	EXPECT_EQ(tr.stats().unsupported, 1u);

	// reset_hashes returns everything to pending but keeps tracking.
	tr.reset_hashes();
	EXPECT_EQ(tr.find(0x1)->hash_state, wwmi::HashState::pending);
	EXPECT_EQ(tr.find(0x1)->data_hash, 0u);
	EXPECT_EQ(tr.find(0x2)->hash_state, wwmi::HashState::pending);
	EXPECT_EQ(tr.stats().hashed, 0u);
	EXPECT_EQ(tr.size(), 2u);
}

// ---- pending picker ----

WWMI_TEST(tracker_pick_pending_oldest_first_with_budget)
{
	wwmi::TextureTracker tr(small_pool_cfg());
	add(tr, 0x1, 512, 512, 30);
	add(tr, 0x2, 512, 512, 10);
	add(tr, 0x3, 512, 512, 20);
	add(tr, 0x4, 512, 512, 40);

	// Budget 0: nothing picked.
	EXPECT(tr.pick_pending(0).empty());

	// Budget 2: the two oldest (frames 10, 20), both transition to learning.
	std::vector<wwmi::TrackedTexture> picked = tr.pick_pending(2);
	EXPECT_EQ(picked.size(), 2u);
	EXPECT_EQ(picked[0].handle, 0x2u);
	EXPECT_EQ(picked[1].handle, 0x3u);
	// Snapshots carry the post-transition state.
	EXPECT_EQ(picked[0].hash_state, wwmi::HashState::learning);
	EXPECT_EQ(tr.find(0x2)->hash_state, wwmi::HashState::learning);
	EXPECT_EQ(tr.find(0x1)->hash_state, wwmi::HashState::pending);

	// Already-learning entries are not re-picked; the next pick gets 0x1.
	std::vector<wwmi::TrackedTexture> more = tr.pick_pending(5);
	EXPECT_EQ(more.size(), 2u);
	EXPECT_EQ(more[0].handle, 0x1u);
	EXPECT_EQ(more[1].handle, 0x4u);

	// Done entries never come back.
	tr.set_hash(0x1, 1, 2);
	tr.set_hash(0x2, 1, 2);
	tr.set_hash(0x3, 1, 2);
	tr.set_hash(0x4, 1, 2);
	EXPECT(tr.pick_pending(5).empty());
}

// ---- state counting (overlay snapshot) ----

WWMI_TEST(tracker_count_by_state)
{
	wwmi::TextureTracker tr(small_pool_cfg());
	EXPECT_EQ(tr.count_by_state().pending, 0u); // empty pool

	add(tr, 0x1, 512, 512, 1); // -> done
	add(tr, 0x2, 512, 512, 2); // -> learning
	add(tr, 0x3, 512, 512, 3); // -> unsupported
	add(tr, 0x4, 512, 512, 4); // stays pending
	tr.set_hash(0x1, 1, 2);
	tr.pick_pending(1); // picks the oldest pending: 0x1 already done, so 0x2
	tr.set_unsupported(0x3);

	const wwmi::TextureTracker::StateCounts c = tr.count_by_state();
	EXPECT_EQ(c.pending, 1u);
	EXPECT_EQ(c.learning, 1u);
	EXPECT_EQ(c.done, 1u);
	EXPECT_EQ(c.unsupported, 1u);
}

// ---- LRU eviction ----

WWMI_TEST(tracker_lru_evicts_unhashed_keeps_learned)
{
	wwmi::TextureTracker tr(small_pool_cfg()); // max_tracked = 4
	add(tr, 0x1, 512, 512, 1);
	add(tr, 0x2, 512, 512, 2);
	add(tr, 0x3, 512, 512, 3);
	add(tr, 0x4, 512, 512, 4);
	tr.set_hash(0x2, 1, 2); // learned entry must survive eviction

	// Pool is full: admitting 0x5 evicts the oldest unhashed entry (0x1).
	EXPECT(add(tr, 0x5, 512, 512, 5) != nullptr);
	EXPECT(tr.find(0x1) == nullptr);
	EXPECT(tr.find(0x2) != nullptr); // done entry kept
	EXPECT_EQ(tr.stats().evicted, 1u);
	EXPECT_EQ(tr.size(), 4u);

	// When every entry is learned, the pool over-extends rather than
	// dropping hashes (they are the tracker's whole purpose).
	tr.set_hash(0x3, 1, 2);
	tr.set_hash(0x4, 1, 2);
	tr.set_hash(0x5, 1, 2);
	EXPECT(add(tr, 0x6, 512, 512, 6) != nullptr);
	EXPECT_EQ(tr.size(), 5u);
	EXPECT_EQ(tr.stats().evicted, 1u);
}

// ---- HashIndex ----

WWMI_TEST(hash_index_build_and_lookup)
{
	std::vector<wwmi::TextureOverrideRule> rules(3);
	rules[0].section = "TextureOverrideBodyDiffuse";
	rules[0].hash = 0x12345678;
	rules[0].has_hash = true;
	rules[1].section = "TextureOverrideHairNormal";
	rules[1].hash = 0xAABBCCDD;
	rules[1].has_hash = true;
	rules[2].section = "TextureOverrideNoHash"; // no hash: not indexed

	wwmi::HashIndex idx;
	idx.build(std::move(rules));
	EXPECT_EQ(idx.size(), 3u);

	const wwmi::TextureOverrideRule *r = idx.find(0x12345678);
	EXPECT(r != nullptr);
	EXPECT_EQ(r->hash, 0x12345678u);

	EXPECT(idx.find(0xAABBCCDD) != nullptr);
	EXPECT(idx.find(0xFFFFFFFF) == nullptr);

	idx.clear();
	EXPECT(idx.find(0x12345678) == nullptr);
}

WWMI_TEST(hash_index_duplicate_hash_first_wins)
{
	std::vector<wwmi::TextureOverrideRule> rules(2);
	rules[0].section = "First";
	rules[0].hash = 0xCAFEBABE;
	rules[0].has_hash = true;
	rules[1].section = "Second";
	rules[1].hash = 0xCAFEBABE;
	rules[1].has_hash = true;

	wwmi::HashIndex idx;
	idx.build(std::move(rules));
	EXPECT_EQ(idx.collision_count(), 1u);

	const wwmi::TextureOverrideRule *r = idx.find(0xCAFEBABE);
	EXPECT(r != nullptr);
	EXPECT_EQ(r->section, std::string("First")); // profile order wins, like 3DMigoto
}

// M2: duplicate hashes keep every rule (3DMigoto multimap semantics);
// find_all() returns them in load order.
WWMI_TEST(hash_index_find_all_keeps_every_rule)
{
	std::vector<wwmi::TextureOverrideRule> rules(3);
	rules[0].section = "Bind";
	rules[0].hash = 0x0BAD1DEA;
	rules[0].has_hash = true;
	rules[1].section = "Skip";
	rules[1].hash = 0x0BAD1DEA;
	rules[1].has_hash = true;
	rules[1].handling = wwmi::HandlingMode::skip;
	rules[2].section = "Other";
	rules[2].hash = 0x600DF00D;
	rules[2].has_hash = true;

	wwmi::HashIndex idx;
	idx.build(std::move(rules));
	EXPECT_EQ(idx.collision_count(), 1u);

	std::vector<const wwmi::TextureOverrideRule *> all;
	idx.find_all(0x0BAD1DEA, all);
	EXPECT_EQ(all.size(), static_cast<size_t>(2));
	if (all.size() == 2)
	{
		EXPECT_EQ(all[0]->section, std::string("Bind"));
		EXPECT_EQ(all[1]->section, std::string("Skip"));
	}

	std::vector<const wwmi::TextureOverrideRule *> none;
	idx.find_all(0xFFFFFFFF, none);
	EXPECT(none.empty());
}

// M2: the draw-rule index only carries handling = skip/abort rules and
// has_draw_rules() gates the draw event fast path.
WWMI_TEST(hash_index_draw_rule_index)
{
	EXPECT(!wwmi::HashIndex{}.has_draw_rules());

	std::vector<wwmi::TextureOverrideRule> rules(4);
	rules[0].section = "PlainBind"; // bindings only: not a draw rule
	rules[0].hash = 0x11111111;
	rules[0].has_hash = true;
	rules[1].section = "SkirtSkip";
	rules[1].hash = 0x22222222;
	rules[1].has_hash = true;
	rules[1].handling = wwmi::HandlingMode::skip;
	rules[2].section = "CloakAbort";
	rules[2].hash = 0x22222222; // same hash, different section
	rules[2].has_hash = true;
	rules[2].handling = wwmi::HandlingMode::abort;
	rules[3].section = "MatcherOnly"; // match_* but no handling: not a draw rule
	rules[3].hash = 0x33333333;
	rules[3].has_hash = true;
	rules[3].match_index_count.enabled = true;

	wwmi::HashIndex idx;
	idx.build(std::move(rules));

	EXPECT(idx.has_draw_rules());
	EXPECT_EQ(idx.draw_rule_count(), static_cast<size_t>(2));

	std::vector<const wwmi::TextureOverrideRule *> draws;
	idx.find_draw_rules(0x22222222, draws);
	EXPECT_EQ(draws.size(), static_cast<size_t>(2));
	if (draws.size() == 2)
	{
		EXPECT_EQ(draws[0]->section, std::string("SkirtSkip"));
		EXPECT_EQ(draws[1]->section, std::string("CloakAbort"));
	}

	std::vector<const wwmi::TextureOverrideRule *> empty;
	idx.find_draw_rules(0x11111111, empty); // bind-only rule
	EXPECT(empty.empty());
	idx.find_draw_rules(0x33333333, empty); // matcher-only rule
	EXPECT(empty.empty());

	// Rules without a valid hash= never enter either index.
	std::vector<const wwmi::TextureOverrideRule *> bogus;
	idx.find_draw_rules(0, bogus);
	EXPECT(bogus.empty());
}

// ---- SessionCache ----

WWMI_TEST(session_cache_pair_and_lookup)
{
	wwmi::SessionCache cache;
	EXPECT(!cache.lookup(0x1234, nullptr));

	cache.pair(0x1234567812345678ull, 0xDEADBEEFu);
	uint32_t mod_hash = 0;
	EXPECT(cache.lookup(0x1234567812345678ull, &mod_hash));
	EXPECT_EQ(mod_hash, 0xDEADBEEFu);
	EXPECT_EQ(cache.size(), 1u);

	// Re-pair overwrites.
	cache.pair(0x1234567812345678ull, 0x11111111u);
	EXPECT(cache.lookup(0x1234567812345678ull, &mod_hash));
	EXPECT_EQ(mod_hash, 0x11111111u);
	EXPECT_EQ(cache.size(), 1u);

	cache.clear();
	EXPECT(!cache.lookup(0x1234567812345678ull, nullptr));
}

WWMI_TEST(session_cache_json_roundtrip)
{
	const std::filesystem::path tmp =
		std::filesystem::temp_directory_path() / "wwmi-test-session.json";
	std::error_code ec;
	std::filesystem::remove(tmp, ec);

	wwmi::SessionCache cache;
	cache.pair(0x1111111122222222ull, 0xAAAABBBBu);
	cache.pair(0, 0x1u); // zero runtime hash is a legal key
	cache.pair(UINT64_MAX, 0xFFFFFFFFu);
	EXPECT(cache.save(tmp));

	wwmi::SessionCache loaded;
	EXPECT(loaded.load(tmp));
	uint32_t v = 0;
	EXPECT(loaded.lookup(0x1111111122222222ull, &v));
	EXPECT_EQ(v, 0xAAAABBBBu);
	EXPECT(loaded.lookup(0, &v));
	EXPECT_EQ(v, 0x1u);
	EXPECT(loaded.lookup(UINT64_MAX, &v));
	EXPECT_EQ(v, 0xFFFFFFFFu);
	EXPECT_EQ(loaded.size(), 3u);

	// Loading a missing file fails silently.
	wwmi::SessionCache empty;
	EXPECT(!empty.load(std::filesystem::temp_directory_path() / "wwmi-no-such-file.json"));

	std::filesystem::remove(tmp, ec);
}

// ---- compact_rows (readback compaction) ----

WWMI_TEST(compact_rows_strips_alignment_padding)
{
	// 3 rows of 4 real bytes stored with a 16-byte aligned pitch.
	std::vector<uint8_t> src(16 * 3);
	for (size_t r = 0; r < 3; ++r)
	{
		src[r * 16 + 0] = static_cast<uint8_t>(0x10 + r);
		src[r * 16 + 1] = static_cast<uint8_t>(0x20 + r);
		src[r * 16 + 2] = static_cast<uint8_t>(0x30 + r);
		src[r * 16 + 3] = static_cast<uint8_t>(0x40 + r);
		// bytes 4..15 are alignment garbage
		std::memset(src.data() + r * 16 + 4, 0xEE, 12);
	}

	uint8_t dst[12] = {};
	const size_t n = wwmi::compact_rows(src.data(), 16, 3, 4, dst, sizeof(dst));
	EXPECT_EQ(n, 12u);
	EXPECT_EQ(dst[0], 0x10);
	EXPECT_EQ(dst[4], 0x11);
	EXPECT_EQ(dst[5], 0x21);
	EXPECT_EQ(dst[11], 0x42);
	// No garbage leaked in.
	for (int i = 0; i < 12; ++i)
		EXPECT(dst[i] != 0xEE);

	// Tight pitch (no padding) is a straight copy.
	const size_t tight = wwmi::compact_rows(src.data(), 4, 3, 4, dst, sizeof(dst));
	EXPECT_EQ(tight, 12u);
	EXPECT_EQ(dst[0], 0x10);
}

WWMI_TEST(compact_rows_rejects_bad_arguments)
{
	uint8_t buf[16] = {};
	EXPECT_EQ(wwmi::compact_rows(nullptr, 16, 1, 4, buf, sizeof(buf)), 0u);
	EXPECT_EQ(wwmi::compact_rows(buf, 16, 1, 4, nullptr, 16), 0u);
	EXPECT_EQ(wwmi::compact_rows(buf, 16, 0, 4, buf, sizeof(buf)), 0u);
	// row_bytes wider than the source pitch is nonsense.
	EXPECT_EQ(wwmi::compact_rows(buf, 4, 1, 8, buf, sizeof(buf)), 0u);
	// Destination too small.
	EXPECT_EQ(wwmi::compact_rows(buf, 16, 2, 4, buf, 7), 0u);
}

// ---- end-to-end: readback pitch -> 3DMigoto hash ----

WWMI_TEST(readback_pipeline_aligned_pitch_matches_tight_hash)
{
	// 64x64 BC1: tight block-row = 16 blocks * 8 bytes = 128 bytes, but D3D12
	// readback aligns the row pitch to 256. Compacting the mapped data back
	// to the tight layout must reproduce the hash 3DMigoto computed over the
	// game's (tight) upload.
	wwmi::SurfaceInfo si;
	EXPECT(wwmi::get_surface_info(64, 64, DXGI_FORMAT_BC1_UNORM, &si));
	EXPECT_EQ(si.row_bytes, 128u);
	EXPECT_EQ(si.num_rows, 16u);
	EXPECT_EQ(si.num_bytes, 2048u);

	std::vector<uint8_t> tight(si.num_bytes);
	for (size_t i = 0; i < tight.size(); ++i)
		tight[i] = static_cast<uint8_t>((i * 17 + 3) & 0xFF);

	const size_t aligned_pitch = 256;
	std::vector<uint8_t> mapped(aligned_pitch * si.num_rows);
	for (size_t r = 0; r < si.num_rows; ++r)
		std::memcpy(mapped.data() + r * aligned_pitch, tight.data() + r * si.row_bytes, si.row_bytes);

	// Compact and hash exactly as the runtime bridge does.
	std::vector<uint8_t> compacted(si.num_bytes);
	EXPECT_EQ(wwmi::compact_rows(mapped.data(), aligned_pitch, si.num_rows, si.row_bytes,
		compacted.data(), compacted.size()), si.num_bytes);

	wwmi::Tex2DDesc desc{};
	desc.width = 64;
	desc.height = 64;
	desc.mip_levels = 1;
	desc.array_size = 1;
	desc.format = DXGI_FORMAT_BC1_UNORM;
	desc.sample_count = 1;
	desc.bind_flags = 0x8; // D3D11_BIND_SHADER_RESOURCE

	wwmi::SubresourceData from_readback;
	from_readback.sys_mem = compacted.data();
	from_readback.sys_mem_pitch = static_cast<uint32_t>(si.row_bytes); // tight

	wwmi::SubresourceData from_upload;
	from_upload.sys_mem = tight.data();
	from_upload.sys_mem_pitch = static_cast<uint32_t>(si.row_bytes); // tight

	const uint32_t h1 = wwmi::calc_texture2d_data_hash(desc, from_readback);
	const uint32_t h2 = wwmi::calc_texture2d_data_hash(desc, from_upload);
	EXPECT_EQ(h1, h2);
	EXPECT_NE(h1, 0u);

	const uint32_t full1 = wwmi::calc_texture2d_desc_hash(h1, desc);
	const uint32_t full2 = wwmi::calc_texture2d_desc_hash(h2, desc);
	EXPECT_EQ(full1, full2);
}

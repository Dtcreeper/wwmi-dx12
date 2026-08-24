#include "buffer_tracker.hpp"
#include "buffer_hash.hpp"
#include "mod_rules.hpp"
#include "test_framework.hpp"

#include <filesystem>
#include <fstream>
#include <vector>

using namespace wwmi;

// ---- BufferTracker ----

WWMI_TEST(buffer_tracker_prefilter_and_admission)
{
	BufferTrackerConfig cfg;
	cfg.min_bytes = 64;
	cfg.max_bytes = 1024;
	BufferTracker t(cfg);

	// Lazy admission at bind time: only IA-bound buffers enter the pool.
	EXPECT(t.track(1, 512, BufferRole::index, 1) != nullptr);
	EXPECT(t.track(2, 32, BufferRole::vertex, 1) == nullptr);  // too small
	EXPECT(t.track(3, 2048, BufferRole::vertex, 1) == nullptr); // too large
	EXPECT(t.track(0, 512, BufferRole::index, 1) == nullptr);   // null handle
	EXPECT_EQ(t.size(), static_cast<size_t>(1));
	EXPECT_EQ(t.stats().tracked, 1u);
	EXPECT_EQ(t.stats().rejected, 2u);
	EXPECT_EQ(t.stats().binds, 3u); // null handle is not a bind

	// Re-binding an existing handle refreshes LRU and counts binds; the
	// entry is returned without re-admission.
	TrackedBuffer *b = t.track(1, 512, BufferRole::index, 5);
	EXPECT(b != nullptr);
	EXPECT_EQ(b->bind_count, 2u); // admission bind + this rebind
	b = t.track(1, 9999, BufferRole::vertex, 6); // different width/role
	EXPECT(b != nullptr);
	EXPECT_EQ(b->byte_width, 512u); // first admission wins
	EXPECT_EQ(b->role, BufferRole::index);
	EXPECT_EQ(b->bind_count, 3u);
	EXPECT_EQ(t.stats().tracked, 1u); // no second admission
}

WWMI_TEST(buffer_tracker_hash_state_machine)
{
	BufferTracker t;

	TrackedBuffer *b = t.track(7, 4096, BufferRole::index, 1);
	EXPECT(b != nullptr);
	EXPECT_EQ(b->hash_state, HashState::pending);

	// Budgeted pick: pending -> learning, oldest first, snapshot by value.
	t.track(8, 8192, BufferRole::vertex, 2);
	std::vector<TrackedBuffer> picked = t.pick_pending(1);
	EXPECT_EQ(picked.size(), static_cast<size_t>(1));
	EXPECT_EQ(picked[0].handle, 7u); // frame 1 before frame 2
	EXPECT_EQ(t.find(7)->hash_state, HashState::learning);

	// Learning entries are not picked again.
	std::vector<TrackedBuffer> again = t.pick_pending(4);
	EXPECT_EQ(again.size(), static_cast<size_t>(1));
	EXPECT_EQ(again[0].handle, 8u);

	// Readback completes: full 3DMigoto hash lands.
	const uint32_t data_hash = calc_buffer_data_hash("abcd", 4);
	const uint32_t full = calc_buffer_hash(data_hash, 4096, BufferRole::index);
	t.set_hash(7, data_hash, full);
	b = t.find(7);
	EXPECT_EQ(b->hash_state, HashState::done);
	EXPECT_EQ(b->hash, full);
	EXPECT_EQ(b->data_hash, data_hash);
	EXPECT_EQ(t.stats().hashed, 1u);

	// Unsupported path is terminal for the entry.
	t.set_unsupported(8);
	EXPECT_EQ(t.find(8)->hash_state, HashState::unsupported);
	EXPECT_EQ(t.stats().unsupported, 1u);

	// reset_hashes(): everything back to pending for a relearn.
	t.reset_hashes();
	EXPECT_EQ(t.find(7)->hash_state, HashState::pending);
	EXPECT_EQ(t.find(7)->hash, 0u);

	// count_by_state drives the overlay.
	t.set_unsupported(8);
	const auto counts = t.count_by_state();
	EXPECT_EQ(counts.pending, 1u);
	EXPECT_EQ(counts.done, 0u);
	EXPECT_EQ(counts.unsupported, 1u);
}

WWMI_TEST(buffer_tracker_lru_evicts_unhashed_keeps_learned)
{
	BufferTrackerConfig cfg;
	cfg.max_tracked = 3;
	BufferTracker t(cfg);

	EXPECT(t.track(1, 100, BufferRole::index, 1) != nullptr);
	EXPECT(t.track(2, 100, BufferRole::index, 2) != nullptr);
	EXPECT(t.track(3, 100, BufferRole::index, 3) != nullptr);
	t.set_hash(1, 0x11111111, 0x11111111); // oldest, but learned

	// Pool is full: admitting #4 evicts the oldest UNHASHED entry (#2).
	EXPECT(t.track(4, 100, BufferRole::index, 4) != nullptr);
	EXPECT(t.find(2) == nullptr);
	EXPECT(t.find(1) != nullptr); // learned entry survives
	EXPECT_EQ(t.stats().evicted, 1u);

	// untrack removes explicitly (resource destroyed).
	t.untrack(3);
	EXPECT(t.find(3) == nullptr);
	EXPECT_EQ(t.size(), static_cast<size_t>(2));
}

WWMI_TEST(buffer_tracker_pick_prioritizes_large_index_buffers)
{
	BufferTracker t;

	// A flood of older pending vertex buffers...
	for (uint64_t i = 1; i <= 8; ++i)
		EXPECT(t.track(i, 4096, BufferRole::vertex, i) != nullptr);
	// ...and a younger large index buffer (the body IB of a mesh mod).
	constexpr uint64_t large_ib = 256ull * 1024;
	EXPECT(t.track(100, large_ib, BufferRole::index, 20) != nullptr);
	// A large vertex buffer of the same age must NOT jump the queue.
	EXPECT(t.track(101, large_ib, BufferRole::vertex, 21) != nullptr);

	// Budget 1: the large IB wins despite being the youngest.
	std::vector<TrackedBuffer> picked = t.pick_pending(1);
	EXPECT_EQ(picked.size(), static_cast<size_t>(1));
	EXPECT_EQ(picked[0].handle, 100u);
	EXPECT_EQ(t.find(100)->hash_state, HashState::learning);

	// Budget 4 on the next frame: the oldest vertex buffers follow.
	std::vector<TrackedBuffer> next = t.pick_pending(4);
	EXPECT_EQ(next.size(), static_cast<size_t>(4));
	EXPECT_EQ(next[0].handle, 1u);
	EXPECT_EQ(next[3].handle, 4u);
	for (const TrackedBuffer &b : next)
		EXPECT(b.role == BufferRole::vertex);
}

// ---- IaState ----

WWMI_TEST(draw_state_bind_replay)
{
	IaState s;

	// Initial state: nothing bound.
	EXPECT(s.vb0() == nullptr);
	EXPECT_EQ(s.ib_buffer, 0u);
	EXPECT_EQ(s.ib_index_size, 0u);

	// Two VB slots bound, one null: mask reflects exactly the non-null.
	const uint64_t bufs[3] = { 0xAA, 0, 0xBB };
	const uint64_t offs[3] = { 0, 0, 96 };
	const uint32_t strides[3] = { 32, 0, 12 };
	s.bind_vertex_buffers(0, 3, bufs, offs, strides);
	EXPECT_EQ(s.vb_valid_mask, (1u << 0) | (1u << 2));
	EXPECT(s.vb0() != nullptr);
	EXPECT_EQ(s.vb0()->buffer, 0xAAu);
	EXPECT_EQ(s.vb0()->stride, 32u);
	EXPECT_EQ(s.vbs[2].buffer, 0xBBu);
	EXPECT_EQ(s.vbs[2].offset, 96u);
	EXPECT_EQ(s.vbs[2].stride, 12u);

	// Rebinding slot 0 with a null buffer clears it.
	const uint64_t null_buf[1] = { 0 };
	s.bind_vertex_buffers(0, 1, null_buf, nullptr, nullptr);
	EXPECT(s.vb0() == nullptr);
	EXPECT_EQ(s.vb_valid_mask, 1u << 2);

	// Slot overflow is clamped, not corrupting adjacent slots.
	const uint64_t bufs2[4] = { 0xCC, 0xCC, 0xCC, 0xCC };
	s.bind_vertex_buffers(31, 4, bufs2, nullptr, nullptr); // only slot 31 fits
	EXPECT_EQ(s.vb_valid_mask, (1u << 2) | (1u << 31));
	EXPECT_EQ(s.vbs[31].buffer, 0xCCu);

	// IB bind/unbind.
	constexpr uint64_t kIbHandle = 0xD44;
	s.bind_index_buffer(kIbHandle, 12, 4);
	EXPECT_EQ(s.ib_buffer, kIbHandle);
	EXPECT_EQ(s.ib_offset, 12u);
	EXPECT_EQ(s.ib_index_size, 4u);
	s.bind_index_buffer(0, 99, 2); // unbind zeroes the view fields
	EXPECT_EQ(s.ib_buffer, 0u);
	EXPECT_EQ(s.ib_offset, 0u);
	EXPECT_EQ(s.ib_index_size, 0u);
}

// ---- find_skip_rule ----

namespace
{
	// Builds a mod rule set: one skip rule keyed on hash <ib_hash>, one
	// on <vb_hash> with a vertex-count filter.
	std::vector<TextureOverrideRule> make_draw_rules(uint32_t ib_hash, uint32_t vb_hash)
	{
		std::vector<TextureOverrideRule> rules(2);
		rules[0].section = "SkipByIB";
		rules[0].hash = ib_hash;
		rules[0].has_hash = true;
		rules[0].handling = HandlingMode::skip;
		rules[1].section = "SkipByVB0";
		rules[1].hash = vb_hash;
		rules[1].has_hash = true;
		rules[1].handling = HandlingMode::skip;
		rules[1].match_vertex_count = { FuzzyOp::equal, 3000, true };
		return rules;
	}
}

WWMI_TEST(find_skip_rule_ib_hash)
{
	constexpr uint64_t kIbHandle = 0xA11;
	constexpr uint64_t kVbHandle = 0xB22;

	HashIndex index;
	index.build(make_draw_rules(0x1BAD0001, 0x1BAD0002));

	BufferTracker t;
	t.track(kIbHandle, 8316, BufferRole::index, 1);
	t.track(kVbHandle, 65536, BufferRole::vertex, 1);

	IaState s;
	s.bind_index_buffer(kIbHandle, 0, 2);
	s.bind_vertex_buffers(0, 1, std::vector<uint64_t>{ kVbHandle }.data(), nullptr, nullptr);

	DrawCallInfo call;
	call.index_count = 4158;
	call.instance_count = 1;

	// Hash not learned yet: draw passes through.
	EXPECT(find_skip_rule(index, t, nullptr, s, call, true) == nullptr);

	// Learned IB hash with matching rule: skipped.
	const uint32_t data_hash = calc_buffer_data_hash("ib", 2);
	t.set_hash(kIbHandle, data_hash, 0x1BAD0001);
	const TextureOverrideRule *r = find_skip_rule(index, t, nullptr, s, call, true);
	EXPECT(r != nullptr);
	EXPECT_EQ(r->section, std::string("SkipByIB"));

	// Non-indexed draw ignores the IB rule.
	EXPECT(find_skip_rule(index, t, nullptr, s, call, false) == nullptr);
}

WWMI_TEST(find_skip_rule_vb0_hash_and_match_filter)
{
	constexpr uint64_t kVbHandle = 0xB22;

	HashIndex index;
	index.build(make_draw_rules(0x1BAD0001, 0x1BAD0002));

	BufferTracker t;
	t.track(kVbHandle, 65536, BufferRole::vertex, 1);
	const uint32_t data_hash = calc_buffer_data_hash("vb", 2);
	t.set_hash(kVbHandle, data_hash, 0x1BAD0002);

	IaState s;
	s.bind_vertex_buffers(0, 1, std::vector<uint64_t>{ kVbHandle }.data(), nullptr, nullptr);

	DrawCallInfo call;
	call.vertex_count = 3000; // matches SkipByVB0's matcher

	const TextureOverrideRule *r = find_skip_rule(index, t, nullptr, s, call, false);
	EXPECT(r != nullptr);
	EXPECT_EQ(r->section, std::string("SkipByVB0"));

	// Wrong vertex count: the matcher blocks the skip.
	call.vertex_count = 2999;
	EXPECT(find_skip_rule(index, t, nullptr, s, call, false) == nullptr);

	// Unbound slot 0 (null bind clears it): no VB0 rules apply.
	const uint64_t nulls[1] = { 0 };
	s.bind_vertex_buffers(0, 1, nulls, nullptr, nullptr);
	EXPECT(find_skip_rule(index, t, nullptr, s, call, false) == nullptr);
}

WWMI_TEST(find_skip_rule_fast_path_and_unknown_hash)
{
	constexpr uint64_t kVbHandle = 0xB22;
	constexpr uint64_t kUntrackedHandle = 0xC33;

	BufferTracker t;
	t.track(kVbHandle, 65536, BufferRole::vertex, 1);
	t.set_hash(kVbHandle, 1, 0xDEADBEEF);

	IaState s;
	s.bind_vertex_buffers(0, 1, std::vector<uint64_t>{ kVbHandle }.data(), nullptr, nullptr);

	DrawCallInfo call;

	// No draw rules loaded at all: immediate nullptr, hash never consulted.
	HashIndex empty;
	EXPECT(find_skip_rule(empty, t, nullptr, s, call, false) == nullptr);

	// Draw rules exist, but this buffer's hash has no rule.
	HashIndex index;
	index.build(make_draw_rules(0x1BAD0001, 0x1BAD0002));
	EXPECT(find_skip_rule(index, t, nullptr, s, call, false) == nullptr);

	// Untracked buffer (rejected by prefilter): passes through.
	IaState s2;
	s2.bind_vertex_buffers(0, 1, std::vector<uint64_t>{ kUntrackedHandle }.data(), nullptr, nullptr);
	EXPECT(find_skip_rule(index, t, nullptr, s2, call, false) == nullptr);
}

// ---- M6: pooled index-buffer views -----------------------------------------

namespace
{
	// A Lynae-style rule set: two component rules sharing one hash and
	// split by (first_index, index_count) windows, plus a windowless
	// rule on the same hash that must NOT join the probe group.
	std::vector<TextureOverrideRule> make_windowed_rules(uint32_t hash)
	{
		std::vector<TextureOverrideRule> rules(3);
		rules[0].section = "Component0";
		rules[0].hash = hash;
		rules[0].has_hash = true;
		rules[0].handling = HandlingMode::skip;
		rules[0].match_first_index = { FuzzyOp::equal, 0, true };
		rules[0].match_index_count = { FuzzyOp::equal, 17970, true };
		rules[1].section = "Component1";
		rules[1].hash = hash;
		rules[1].has_hash = true;
		rules[1].handling = HandlingMode::skip;
		rules[1].match_first_index = { FuzzyOp::equal, 17970, true };
		rules[1].match_index_count = { FuzzyOp::equal, 45636, true };
		rules[2].section = "Windowless";
		rules[2].hash = hash;
		rules[2].has_hash = true;
		rules[2].handling = HandlingMode::skip;
		return rules;
	}
}

WWMI_TEST(draw_window_index_groups_windowed_rules)
{
	const uint32_t hash = 0x0c33d628;

	DrawWindowIndex windows;
	windows.build(make_windowed_rules(hash));
	EXPECT_EQ(windows.groups().size(), static_cast<size_t>(1));

	const DrawRuleGroup &g = windows.groups()[0];
	EXPECT_EQ(g.hash, hash);
	EXPECT_EQ(g.windows.size(), static_cast<size_t>(2)); // windowless rule stays out
	EXPECT_EQ(g.min_first_index, 0u);
	EXPECT_EQ(g.max_end_index, 17970u + 45636u);

	EXPECT(windows.find_by_draw(0, 17970) == &g);
	EXPECT(windows.find_by_draw(17970, 45636) == &g);
	EXPECT(windows.find_by_draw(0, 1) == nullptr);          // count mismatch
	EXPECT(windows.find_by_draw(999999, 17970) == nullptr); // first_index mismatch
	EXPECT(windows.find_by_draw(63606, 12324) == nullptr);  // not in this group
}

WWMI_TEST(index_view_probe_state_machine_and_routing)
{
	const uint32_t hash = 0x0c33d628;

	// Routing index uses only the windowed rules: the windowless rule
	// on the same hash would (correctly, by 3DMigoto semantics) match
	// every draw of a verified view and muddy the window assertions.
	std::vector<TextureOverrideRule> routing_rules = make_windowed_rules(hash);
	routing_rules.resize(2);
	HashIndex index;
	index.build(routing_rules);

	// A 32 MB-style pool; the mesh view sits at offset 0x1000.
	constexpr uint64_t kPool = 0x600000;
	constexpr uint64_t kOff = 0x1000;

	IndexViewTracker views;
	TrackedIndexView *v = views.track(kPool, kOff, 4, 10);
	EXPECT(v != nullptr);
	EXPECT(views.probe_due(*v, 10)); // fresh view arms immediately

	IaState s;
	s.bind_index_buffer(kPool, kOff, 4);
	DrawCallInfo call;
	call.first_index = 0;
	call.index_count = 17970;
	call.instance_count = 1;

	// Unverified view routes nothing (and the whole-buffer path cannot
	// learn a pooled buffer's DX11 hash).
	BufferTracker empty_buffers;
	EXPECT(find_skip_rule(index, empty_buffers, &views, s, call, true) == nullptr);

	// Probe verified: draws route by window exactly like a learned
	// dedicated buffer would.
	views.set_verified(kPool, kOff, hash);
	v = views.find(kPool, kOff);
	EXPECT(v != nullptr && v->state == TrackedIndexView::State::verified);
	EXPECT(!views.probe_due(*v, 10)); // verified: never re-probed

	const TextureOverrideRule *r0 =
		find_skip_rule(index, empty_buffers, &views, s, call, true);
	EXPECT(r0 != nullptr);
	EXPECT_EQ(r0->section, std::string("Component0"));

	call.first_index = 17970;
	call.index_count = 45636;
	const TextureOverrideRule *r1 =
		find_skip_rule(index, empty_buffers, &views, s, call, true);
	EXPECT(r1 != nullptr);
	EXPECT_EQ(r1->section, std::string("Component1"));

	// Outside every window: passes through even with the verified view.
	call.first_index = 999999;
	EXPECT(find_skip_rule(index, empty_buffers, &views, s, call, true) == nullptr);

	// Another offset in the same pool is a distinct view.
	EXPECT(views.find(kPool, kOff + 16) == nullptr);

	// Failure path: cooldown holds, then re-arms.
	TrackedIndexView *v2 = views.track(kPool, 0x2000, 4, 10);
	EXPECT(v2 != nullptr);
	views.set_failed(kPool, 0x2000);
	v2 = views.find(kPool, 0x2000);
	EXPECT(v2 != nullptr && v2->state == TrackedIndexView::State::failed);
	EXPECT(!views.probe_due(*v2, 10)); // cooldown active
	EXPECT(views.probe_due(*v2, 10 + IndexViewTracker::retry_cooldown_frames));
}

// Regression for the load_mods assembly bug: rules parsed from a real
// mod.ini (Lynae-style windowed components) must produce a non-empty
// DrawWindowIndex when fed through the same build() call the addon's
// load path makes. The original defect -- windows.build() never called
// in load_mods -- left find_by_draw permanently dead while every unit
// test passed (tests built the object directly). This test covers the
// parser -> window-index data flow end to end so a parse regression
// dropping the window matchers fails here, and the missing-build-call
// class of bug is additionally guarded by the addon's startup warn
// (windowed rules present but no probe group).
WWMI_TEST(windowed_rules_from_mod_ini_feed_draw_window_index)
{
	const std::filesystem::path dir =
		std::filesystem::temp_directory_path() / "wwmi_test_windowed";
	std::filesystem::create_directories(dir);
	const std::filesystem::path ini = dir / "mod.ini";

	{
		std::ofstream f(ini, std::ios::binary);
		f << "[TextureOverrideComponent0]\n"
			<< "hash = 0c33d628\n"
			<< "handling = skip\n"
			<< "match_first_index = 0\n"
			<< "match_index_count = 17970\n"
			<< "\n"
			<< "[TextureOverrideComponent1]\n"
			<< "hash = 0c33d628\n"
			<< "handling = skip\n"
			<< "match_first_index = 17970\n"
			<< "match_index_count = 45636\n";
	}

	ModRules mod;
	EXPECT(load_mod_rules(ini, mod));
	EXPECT_EQ(mod.overrides.size(), static_cast<size_t>(2));

	// Same assembly the addon performs in load_mods(): windows BEFORE
	// index.build() consumes (moves) the rules.
	DrawWindowIndex windows;
	windows.build(mod.overrides);
	EXPECT(!windows.empty());
	EXPECT_EQ(windows.groups().size(), static_cast<size_t>(1));

	const DrawRuleGroup &g = windows.groups()[0];
	EXPECT_EQ(g.hash, 0x0c33d628u);
	EXPECT_EQ(g.windows.size(), static_cast<size_t>(2));
	EXPECT_EQ(g.min_first_index, 0u);
	EXPECT_EQ(g.max_end_index, 17970u + 45636u);
	EXPECT(windows.find_by_draw(0, 17970) == &g);
	EXPECT(windows.find_by_draw(17970, 45636) == &g);

	std::filesystem::remove_all(dir);
}

// M7: signature-coverage verification -- the pooled-view substitute
// for the byte-hash probe, which cannot succeed on UE DX12 (vertex
// re-indexing changes the bytes while the section layout survives).
WWMI_TEST(signature_coverage_verifies_pooled_view)
{
	IndexViewTracker views;
	DrawRuleGroup grp;
	grp.hash = 0x0c33d628u;
	grp.section = "TextureOverrideComponent0";
	// Lynae body tiling: 8 windows covering [0, 310422).
	const uint32_t firsts[8] = { 0, 17970, 63606, 75930,
		173856, 274170, 301470, 308814 };
	const uint32_t counts[8] = { 17970, 45636, 12324, 97926,
		100314, 27300, 7344, 1608 };
	for (int i = 0; i < 8; ++i)
		grp.windows.push_back({ firsts[i], counts[i] });

	TrackedIndexView *v = views.track(0x1234, 0x2000, 4, 1);
	EXPECT(v != nullptr);
	v->interesting = true;

	// Draw 7 of the 8 windows: coverage incomplete.
	for (int i = 0; i < 7; ++i)
		v->tally(firsts[i], counts[i]);
	EXPECT(!signature_coverage_complete(*v, grp));

	// Interleave a foreign signature (another mesh's draw): it must
	// not affect coverage of the group's windows.
	v->tally(999999, 42);
	EXPECT(!signature_coverage_complete(*v, grp));

	// The last window draws: coverage complete -> the addon would now
	// call set_verified with the group hash.
	v->tally(firsts[7], counts[7]);
	EXPECT(signature_coverage_complete(*v, grp));

	views.set_verified(0x1234, 0x2000, grp.hash);
	const TrackedIndexView *done = views.find(0x1234, 0x2000);
	EXPECT(done != nullptr);
	EXPECT(done->state == TrackedIndexView::State::verified);
	EXPECT_EQ(done->hash, grp.hash);

	// Verified views route via the whole-buffer rule path: a matching
	// draw signature must resolve to a windowed rule.
	HashIndex index;
	std::vector<TextureOverrideRule> rules;
	for (int i = 0; i < 8; ++i)
	{
		TextureOverrideRule r;
		r.section = "Component" + std::to_string(i);
		r.hash = grp.hash;
		r.has_hash = true;
		r.match_first_index.enabled = true;
		r.match_first_index.value = firsts[i];
		r.match_index_count.enabled = true;
		r.match_index_count.value = counts[i];
		r.handling = HandlingMode::skip;
		rules.push_back(r);
	}
	index.build(std::move(rules));

	IaState ia{};
	ia.ib_buffer = 0x1234;
	ia.ib_offset = 0x2000;
	ia.ib_index_size = 4;
	DrawCallInfo call{};
	call.first_index = firsts[3];
	call.index_count = counts[3];

	const TextureOverrideRule *hit = find_skip_rule(
		index, BufferTracker{}, &views, ia, call, true);
	EXPECT(hit != nullptr);
	EXPECT_EQ(hit->match_first_index.value, firsts[3]);
	EXPECT_EQ(hit->match_index_count.value, counts[3]);
}

// M9 unit tests: CPU-side skeleton chain (SkeletonMerger +
// SkeletonRemapper semantics replicated from the WWMIv1 HLSL).
//
// Heuristic note (SkeletonMerger.hlsl): bone vg is copied when any float
// in rows vg, vg+1, vg+2 of the game CB (UN-multiplied index) is
// non-zero -- those rows belong to bones around vg/3, not to bone vg.
// With the dense skeletons the game produces every check inside the
// alive region passes; the synthetic data below mirrors that.
#include "test_framework.hpp"

#include "skeleton_chain.hpp"

#include <algorithm>

using namespace wwmi;

namespace
{
	// Game skeleton CB where rows 0..(live_bones*3-1) are non-zero:
	// row r carries the value r+1 in x (bone b = rows 3b..3b+2).
	std::vector<float> make_game_cb(uint32_t live_bones, float sign = 1.0f)
	{
		std::vector<float> cb(SkeletonChain::k_game_rows * 4, 0.0f);
		const uint32_t live_rows = std::min(live_bones * 3, SkeletonChain::k_game_rows);
		for (uint32_t r = 0; r < live_rows; ++r)
			cb[r * 4] = sign * static_cast<float>(r + 1);
		return cb;
	}
}

// The merger maps game bone vg of each component window to merged slot
// vg_offset+vg (comp0: 0..44, comp1: 45..116 in the Lynae config).
WWMI_TEST(skeleton_merge_window_mapping)
{
	SkeletonChain c;
	c.rules["comp0"] = { 0, 45, -1, 0 };
	c.rules["comp1"] = { 45, 72, -1, 0 };
	c.finalize_config();

	const std::vector<float> main = make_game_cb(80);
	const std::vector<float> extra = make_game_cb(0);
	c.merge(main.data(), extra.data());

	// comp0: game bone 2 (rows 6..8) -> merged slot 2.
	EXPECT_EQ(c.merged()[(2 * 3 + 0) * 4], 7.0f);
	EXPECT_EQ(c.merged()[(2 * 3 + 2) * 4], 9.0f);
	// comp1: game bone 2 -> merged slot 45+2 = 47.
	EXPECT_EQ(c.merged()[((45 + 2) * 3 + 0) * 4], 7.0f);
	// comp1 window reaches game bone 60 (rows 180..182).
	EXPECT_EQ(c.merged()[((45 + 60) * 3 + 0) * 4], 181.0f);
	// extra skeleton stays all zero.
	EXPECT_EQ(c.extra()[128], 0.0f);
	// every mapped bone of both windows was written (dense game data).
	EXPECT_EQ(c.merged_bones_written(), 45u + 72u);
	EXPECT_EQ(c.extra_bones_written(), 0u);
}

// Sparse game data: the presence heuristic (rows vg..vg+2) skips bones
// whose check region is zero while bones in the alive check region are
// still copied (zero-data copies included, exactly like the CS).
WWMI_TEST(skeleton_merge_presence_heuristic)
{
	SkeletonChain c;
	c.rules["comp"] = { 0, 8, -1, 0 };
	c.finalize_config();

	std::vector<float> main(SkeletonChain::k_game_rows * 4, 0.0f);
	// bone 1 (rows 3..5) alive.
	for (uint32_t k = 0; k < 3; ++k)
		main[(3 + k) * 4] = 100.0f + static_cast<float>(k);
	const std::vector<float> extra(SkeletonChain::k_game_rows * 4, 0.0f);
	c.merge(main.data(), extra.data());

	// Checks pass for vg in 1..5 (their row windows touch rows 3..5):
	// bone 1 carries data, bones 2..4 copy zero rows.
	EXPECT_EQ(c.merged_bones_written(), 5u);
	EXPECT_EQ(c.merged()[(3 + 0) * 4], 100.0f); // bone 1 row 0
	EXPECT_EQ(c.merged()[(3 + 1) * 4], 101.0f); // bone 1 row 1
	// bone 0's check window (rows 0..2) is zero: not merged.
	EXPECT_EQ(c.merged()[0], 0.0f);
	// bones 6..7's windows (rows 6..9) are zero: not merged.
	EXPECT_EQ(c.merged()[(6 * 3) * 4], 0.0f);
}

// The remapper gathers remapped[local] = merged[forward[r*512+local]]
// for both the main and extra skeletons; forward may point into another
// component's merged range (cross-component bone references).
WWMI_TEST(skeleton_remap_gather)
{
	SkeletonChain c;
	c.rules["comp0"] = { 0, 45, -1, 0 };
	c.rules["comp3"] = { 118, 111, 0, 4 };
	c.finalize_config();

	// forward table 0: local 0..3 -> merged slots 123 (comp3 range),
	// 5 (comp0 range), 125, 0.
	c.forward.assign(
		static_cast<size_t>(SkeletonChain::k_remap_stride) * SkeletonChain::k_remap_tables, 0);
	c.forward[0] = 123;
	c.forward[1] = 5;
	c.forward[2] = 125;
	c.forward[3] = 0;

	// 10 live game bones (rows 0..29 non-zero); the extra skeleton the
	// same with negated values.
	const std::vector<float> main = make_game_cb(10);
	const std::vector<float> extra = make_game_cb(10, -1.0f);
	c.merge(main.data(), extra.data());

	const float *rem = c.remapped(0);
	const float *erem = c.extra_remapped(0);
	EXPECT(rem != nullptr);
	EXPECT(erem != nullptr);

	// local 0 -> merged 123 = comp3's game bone 5 (rows 15..17 = 16,17,18).
	EXPECT_EQ(rem[(0 * 3 + 0) * 4], 16.0f);
	EXPECT_EQ(rem[(0 * 3 + 2) * 4], 18.0f);
	// local 1 -> merged 5 = comp0's game bone 5 (cross-component).
	EXPECT_EQ(rem[(1 * 3 + 0) * 4], 16.0f);
	// local 2 -> merged 125 = comp3's game bone 7 (rows 21..23 = 22,23,24).
	EXPECT_EQ(rem[(2 * 3 + 0) * 4], 22.0f);
	// local 3 -> merged 0 = comp0's game bone 0 (row 0 = 1).
	EXPECT_EQ(rem[(3 * 3 + 0) * 4], 1.0f);
	// extra skeleton gathered identically (negated).
	EXPECT_EQ(erem[(0 * 3 + 0) * 4], -16.0f);
	EXPECT_EQ(erem[(1 * 3 + 2) * 4], -18.0f);
	// unused tables report no buffer.
	EXPECT(c.remapped(1) == nullptr);
	EXPECT(c.remapped(2) == nullptr);
	EXPECT(c.remapped(3) == nullptr);
}

// Region math the upload layout depends on (256-byte alignment).
WWMI_TEST(skeleton_region_math)
{
	EXPECT_EQ(SkeletonChain::bone_rows_bytes(512), 24576ull);
	EXPECT_EQ(SkeletonChain::bone_rows_bytes(105), 5040ull);
	EXPECT_EQ(SkeletonChain::bone_rows_bytes(0), 0ull);
}

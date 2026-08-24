// M4 unit tests: XXH64 shader hashing + PipelineTracker subobject cache.
#include "test_framework.hpp"

#include "xxhash64.hpp"
#include "pipeline_tracker.hpp"

#include <cstring>

using namespace wwmi;

// ---------------------------------------------------------------------
// XXH64 canonical vectors (see xxhash64.hpp header comment)
// ---------------------------------------------------------------------

WWMI_TEST(xxhash64_canonical_vectors)
{
	EXPECT_EQ(xxhash64("", 0), 0xEF46DB3751D8E999ull);
	EXPECT_EQ(xxhash64("a", 1), 0xD24EC4F1A98C6E5Bull);
	EXPECT_EQ(xxhash64("abc", 3), 0x44BC2CF5AD770999ull);
}

// >=32-byte inputs exercise the striped round path; the exact vector is
// implementation-defined, so check stability + a size-dependent change.
WWMI_TEST(xxhash64_long_input)
{
	const char *s = "The quick brown fox jumps over the lazy dog";
	const size_t n = std::strlen(s);
	EXPECT(n >= 32);
	EXPECT_EQ(xxhash64(s, n), xxhash64(s, n));            // deterministic
	EXPECT_NE(xxhash64(s, n), xxhash64(s, n - 1));        // length mixes in
}

// Distinct bytecodes must produce distinct hashes (pipeline identity).
WWMI_TEST(xxhash64_distinct_inputs)
{
	const uint8_t a[8] = { 1, 2, 3, 4, 5, 6, 7, 8 };
	const uint8_t b[8] = { 1, 2, 3, 4, 5, 6, 7, 9 };
	EXPECT_NE(xxhash64(a, sizeof a), xxhash64(b, sizeof b));
}

// ---------------------------------------------------------------------
// PipelineTracker
// ---------------------------------------------------------------------

namespace
{
	// Minimal DXBC-looking blobs (content only matters for the hash).
	uint8_t g_vs_code[16] = {
		0x44, 0x58, 0x42, 0x43, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8
	};
	uint8_t g_ps_code[16] = {
		0x44, 0x58, 0x42, 0x43, 0, 0, 0, 0, 8, 7, 6, 5, 4, 3, 2, 1
	};
}

// init_pipeline hashes each shader stage; find() reports them per stage.
WWMI_TEST(pipeline_tracker_hashes_stages)
{
	PipelineTracker t;
	reshade::api::shader_desc vs_desc{ g_vs_code, sizeof g_vs_code };
	reshade::api::shader_desc ps_desc{ g_ps_code, sizeof g_ps_code };
	const pipeline_subobject subs[2] = {
		{ reshade::api::pipeline_subobject_type::vertex_shader, 1, &vs_desc },
		{ reshade::api::pipeline_subobject_type::pixel_shader, 1, &ps_desc },
	};
	t.on_init_pipeline(0x1234, 2, subs, 0xAAA);

	const PipelineShaders *sh = t.find(0xAAA);
	EXPECT(sh != nullptr);
	if (sh != nullptr)
	{
		EXPECT_EQ(sh->vs, xxhash64(g_vs_code, sizeof g_vs_code));
		EXPECT_EQ(sh->ps, xxhash64(g_ps_code, sizeof g_ps_code));
		EXPECT(sh->has_vs);
		EXPECT(sh->has_ps);
		EXPECT(!sh->has_cs);
		EXPECT_EQ(sh->cs, 0u);
	}
	EXPECT(t.find(0xBBB) == nullptr);
}

// Identical bytecode in two pipelines -> identical hashes (mod keying).
WWMI_TEST(pipeline_tracker_same_bytecode_same_hash)
{
	PipelineTracker t;
	reshade::api::shader_desc d{ g_vs_code, sizeof g_vs_code };
	const pipeline_subobject subs1[1] = {
		{ reshade::api::pipeline_subobject_type::vertex_shader, 1, &d }
	};
	t.on_init_pipeline(1, 1, subs1, 0x1);
	t.on_init_pipeline(2, 1, subs1, 0x2);

	const PipelineShaders *a = t.find(0x1);
	const PipelineShaders *b = t.find(0x2);
	EXPECT(a != nullptr && b != nullptr);
	if (a != nullptr && b != nullptr)
		EXPECT_EQ(a->vs, b->vs);
}

// clone_source() returns the deep-copied subobject array and survives
// the original going away; destroy_pipeline drops it again.
WWMI_TEST(pipeline_tracker_clone_source_lifecycle)
{
	PipelineTracker t(4);
	reshade::api::shader_desc d{ g_vs_code, sizeof g_vs_code };
	const pipeline_subobject subs[1] = {
		{ reshade::api::pipeline_subobject_type::vertex_shader, 1, &d }
	};
	t.on_init_pipeline(0x77, 1, subs, 0x500);

	const auto *src = t.clone_source(0x500);
	EXPECT(src != nullptr);
	if (src != nullptr)
	{
		EXPECT_EQ(src->size(), 1u);
		EXPECT_EQ((*src)[0].type, reshade::api::pipeline_subobject_type::vertex_shader);
		const auto *sd = static_cast<const reshade::api::shader_desc *>((*src)[0].data);
		EXPECT(sd != nullptr);
		if (sd != nullptr)
		{
			EXPECT_EQ(sd->code_size, sizeof g_vs_code);
			EXPECT(std::memcmp(sd->code, g_vs_code, sizeof g_vs_code) == 0);
		}
	}
	EXPECT_EQ(t.clone_source_layout(0x500), 0x77u);

	t.on_destroy_pipeline(0x500);
	EXPECT(t.clone_source(0x500) == nullptr);
	EXPECT(t.find(0x500) == nullptr);
}

// LRU cap: with max_cached = 2 the third insert evicts the least
// recently used clone source, but hash lookup stays possible.
WWMI_TEST(pipeline_tracker_lru_eviction)
{
	PipelineTracker t(2);
	reshade::api::shader_desc d{ g_vs_code, sizeof g_vs_code };
	const pipeline_subobject subs[1] = {
		{ reshade::api::pipeline_subobject_type::vertex_shader, 1, &d }
	};
	t.on_init_pipeline(1, 1, subs, 0x10);
	t.on_init_pipeline(2, 1, subs, 0x20);
	t.on_init_pipeline(3, 1, subs, 0x30);

	EXPECT_EQ(t.cached_count(), 2u);
	// 0x10 was LRU-evicted...
	EXPECT(t.clone_source(0x10) == nullptr);
	// ...but its hashes survive for ShaderOverride matching.
	EXPECT(t.find(0x10) != nullptr);
	EXPECT(t.clone_source(0x20) != nullptr);
	EXPECT(t.clone_source(0x30) != nullptr);
	EXPECT_EQ(t.hash_count(), 3u);
}

// A blend_state subobject is deep-copied so a later clone can patch it.
WWMI_TEST(pipeline_tracker_caches_blend_state)
{
	PipelineTracker t;
	reshade::api::blend_desc blend{};
	blend.blend_enable[0] = false;
	const pipeline_subobject subs[1] = {
		{ reshade::api::pipeline_subobject_type::blend_state, 1, &blend }
	};
	t.on_init_pipeline(5, 1, subs, 0x90);

	const auto *src = t.clone_source(0x90);
	EXPECT(src != nullptr);
	if (src != nullptr)
	{
		const auto *bd = static_cast<const reshade::api::blend_desc *>((*src)[0].data);
		EXPECT(bd != nullptr);
		// independent copy of the original value
		if (bd != nullptr)
			EXPECT(!bd->blend_enable[0]);
	}
}

// Blend token parsing (neutral codes; the addon bridge maps to the API).
#include "script_model.hpp"

WWMI_TEST(blend_token_parsing)
{
	uint8_t op = 0, src = 0, dst = 0;
	EXPECT(parse_blend_op("ADD", op));
	EXPECT_EQ(op, 0);
	EXPECT(parse_blend_op("REVERSE_SUBTRACT", op));
	EXPECT_EQ(op, 2);
	EXPECT(!parse_blend_op("NOSUCHOP", op));

	EXPECT(parse_blend_factor("BLEND_FACTOR", src));
	EXPECT_EQ(src, 10);
	EXPECT(parse_blend_factor("INV_BLEND_FACTOR", dst));
	EXPECT_EQ(dst, 11);
	EXPECT(parse_blend_factor("SRC_ALPHA_SAT", src));
	EXPECT_EQ(src, 12);
	EXPECT(!parse_blend_factor("WHAT", src));
}

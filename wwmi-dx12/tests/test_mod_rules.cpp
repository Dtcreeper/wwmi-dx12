#include "mod_rules.hpp"
#include "test_framework.hpp"

#include <fstream>

using namespace wwmi;

WWMI_TEST(hash_parse_matches_3dmigoto)
{
	uint64_t h = 0;
	EXPECT(parse_hash("0x1234abcd", h));
	EXPECT_EQ(h, 0x1234abcdULL);

	// No 0x prefix is still hex ('%16llx').
	EXPECT(parse_hash("deadBEEF", h));
	EXPECT_EQ(h, 0xdeadbeefULL);

	// 64-bit shader hashes must be written without the 0x prefix: the
	// '%16llx' width counts the 0x prefix, so '0x' + 16 hex digits is a
	// parse error in 3DMigoto itself (verified against MSVC sscanf_s).
	EXPECT(parse_hash("1234567890abcdef", h));
	EXPECT_EQ(h, 0x1234567890abcdefULL);
	EXPECT(!parse_hash("0x1234567890abcdef", h));

	// Negative values wrap to two's complement (scanf '%llx' semantics).
	EXPECT(parse_hash("-1", h));
	EXPECT_EQ(h, 0xffffffffffffffffULL);

	// Trailing garbage / empty / bare prefix -> rejected (3dmigoto warns).
	EXPECT(!parse_hash("0x12 zz", h));
	EXPECT(!parse_hash("", h));
	EXPECT(!parse_hash("0x", h));
}

WWMI_TEST(texture_slot_key_parsing)
{
	char stage = 0;
	uint32_t slot = 0;

	EXPECT(parse_texture_slot_key("ps-t0", stage, slot));
	EXPECT_EQ(stage, 'p');
	EXPECT_EQ(slot, 0u);

	EXPECT(parse_texture_slot_key("VS-T127", stage, slot)); // case-insensitive
	EXPECT_EQ(stage, 'v');
	EXPECT_EQ(slot, 127u);

	EXPECT(parse_texture_slot_key("cs-t3", stage, slot));
	EXPECT_EQ(stage, 'c');
	EXPECT_EQ(slot, 3u);

	// Invalid stages / wrong type / out-of-range slot / malformed.
	EXPECT(!parse_texture_slot_key("xs-t0", stage, slot));
	EXPECT(!parse_texture_slot_key("ps-s0", stage, slot)); // samplers not slots
	EXPECT(!parse_texture_slot_key("ps-cb0", stage, slot)); // constant buffer
	EXPECT(!parse_texture_slot_key("ps-t128", stage, slot)); // >= 128
	EXPECT(!parse_texture_slot_key("ps-t", stage, slot));
	EXPECT(!parse_texture_slot_key("ps-t1a", stage, slot));
	EXPECT(!parse_texture_slot_key("vb0", stage, slot));
}

WWMI_TEST(resource_ref_parsing)
{
	std::string res;

	// Canonical form (case preserved in name is lowered).
	EXPECT(parse_resource_ref("ResourceBodyDiffuse", res));
	EXPECT_EQ(res, "bodydiffuse");

	// Copy options are skipped.
	EXPECT(parse_resource_ref("ref ResourceFoo", res));
	EXPECT_EQ(res, "foo");
	EXPECT(parse_resource_ref("copy stereo ResourceBar", res));
	EXPECT_EQ(res, "bar");

	// Special sources are rejected in M1.
	EXPECT(!parse_resource_ref("this", res));
	EXPECT(!parse_resource_ref("null", res));
	EXPECT(!parse_resource_ref("bb", res));

	// Bare names without the resource prefix are not valid.
	EXPECT(!parse_resource_ref("BodyDiffuse", res));
	EXPECT(!parse_resource_ref("", res));
}

WWMI_TEST(load_mod_rules_end_to_end)
{
	// Write a representative appearance-mod mod.ini to a temp dir.
	const std::filesystem::path dir = std::filesystem::temp_directory_path() / "wwmi_test_mod";
	std::filesystem::create_directories(dir);
	const std::filesystem::path ini = dir / "mod.ini";

	{
		std::ofstream f(ini, std::ios::binary);
		f << "; appearance mod\n"
			<< "[TextureOverrideRoverBody]\n"
			<< "hash = 0x1a2b3c4d\n"
			<< "ps-t0 = ResourceBodyDiffuse\n"
			<< "ps-t1 = ResourceBodyNormal\n"
			<< "vs-t0 = ResourceBodyDiffuse\n"
			<< "\n"
			<< "[TextureOverrideBadHash]\n"
			<< "hash = not-a-hash\n"
			<< "ps-t0 = ResourceBodyDiffuse\n"
			<< "\n"
			<< "[ResourceBodyDiffuse]\n"
			<< "filename = Textures/Body_Diffuse.dds\n"
			<< "\n"
			<< "[ResourceBodyNormal]\n"
			<< "filename = Textures/Body_Normal.dds\n";
	}

	ModRules rules;
	EXPECT(load_mod_rules(ini, rules));
	EXPECT_EQ(rules.mod_dir, dir);
	EXPECT_EQ(rules.overrides.size(), static_cast<size_t>(1)); // bad-hash section dropped
	EXPECT_EQ(rules.resources.size(), static_cast<size_t>(2));

	if (rules.overrides.size() != 1)
	{
		EXPECT(false);
		std::filesystem::remove_all(dir);
		return;
	}

	const TextureOverrideRule &ov = rules.overrides[0];
	EXPECT(ov.has_hash);
	EXPECT_EQ(ov.hash, 0x1a2b3c4du);
	EXPECT_EQ(ov.bindings.size(), static_cast<size_t>(3));
	if (ov.bindings.size() == 3)
	{
		EXPECT_EQ(ov.bindings[0].stage, 'p');
		EXPECT_EQ(ov.bindings[0].slot, 0u);
		EXPECT_EQ(ov.bindings[0].resource, "bodydiffuse");
		EXPECT_EQ(ov.bindings[2].stage, 'v');
		EXPECT_EQ(ov.bindings[2].slot, 0u);
	}

	const auto it = rules.resources.find("resourcebodydiffuse");
	EXPECT(it != rules.resources.end());
	if (it != rules.resources.end())
		EXPECT_EQ(it->second.filename, "Textures/Body_Diffuse.dds");

	// One warning for the malformed hash section (the section itself is
	// dropped, matching 3DMigoto's "missing Hash=" behaviour).
	EXPECT(!rules.warnings.empty());

	std::filesystem::remove_all(dir);
}

WWMI_TEST(load_mod_rules_missing_file)
{
	ModRules rules;
	EXPECT(!load_mod_rules(std::filesystem::temp_directory_path() / "wwmi_no_such_mod.ini", rules));
}

// ---- M2: handling / match_* draw-interception rules ----

WWMI_TEST(fuzzy_match_parsing)
{
	FuzzyMatch m;

	// Bare value defaults to '='.
	EXPECT(parse_fuzzy_match("123", m));
	EXPECT(m.enabled);
	EXPECT_EQ(static_cast<int>(m.op), static_cast<int>(FuzzyOp::equal));
	EXPECT_EQ(m.value, 123u);

	// Explicit operators, optional whitespace after the operator.
	EXPECT(parse_fuzzy_match("= 123", m));
	EXPECT_EQ(static_cast<int>(m.op), static_cast<int>(FuzzyOp::equal));
	EXPECT(parse_fuzzy_match("!123", m));
	EXPECT_EQ(static_cast<int>(m.op), static_cast<int>(FuzzyOp::not_equal));
	EXPECT(parse_fuzzy_match("< 123", m));
	EXPECT_EQ(static_cast<int>(m.op), static_cast<int>(FuzzyOp::less));
	EXPECT(parse_fuzzy_match("<=123", m));
	EXPECT_EQ(static_cast<int>(m.op), static_cast<int>(FuzzyOp::less_equal));
	EXPECT(parse_fuzzy_match("> 123", m));
	EXPECT_EQ(static_cast<int>(m.op), static_cast<int>(FuzzyOp::greater));
	EXPECT(parse_fuzzy_match(">=123", m));
	EXPECT_EQ(static_cast<int>(m.op), static_cast<int>(FuzzyOp::greater_equal));

	// Leading whitespace before the operator is tolerated.
	EXPECT(parse_fuzzy_match("  >= 7", m));
	EXPECT_EQ(static_cast<int>(m.op), static_cast<int>(FuzzyOp::greater_equal));
	EXPECT_EQ(m.value, 7u);

	// Field-name right-hand sides and other junk are rejected.
	FuzzyMatch bad;
	EXPECT(!parse_fuzzy_match("width", bad));
	EXPECT(!parse_fuzzy_match("", bad));
	EXPECT(!parse_fuzzy_match("= ", bad));
	EXPECT(!parse_fuzzy_match("! ", bad));
	EXPECT(!parse_fuzzy_match("12x", bad));
	EXPECT(!parse_fuzzy_match("-5", bad));
	EXPECT(!bad.enabled);
}

WWMI_TEST(fuzzy_match_evaluation)
{
	const auto make = [](FuzzyOp op, uint32_t v)
	{
		FuzzyMatch m;
		m.op = op;
		m.value = v;
		m.enabled = true;
		return m;
	};

	EXPECT(make(FuzzyOp::equal, 10).matches(10));
	EXPECT(!make(FuzzyOp::equal, 10).matches(11));
	EXPECT(make(FuzzyOp::not_equal, 10).matches(11));
	EXPECT(!make(FuzzyOp::not_equal, 10).matches(10));
	EXPECT(make(FuzzyOp::less, 10).matches(9));
	EXPECT(!make(FuzzyOp::less, 10).matches(10));
	EXPECT(make(FuzzyOp::less_equal, 10).matches(10));
	EXPECT(!make(FuzzyOp::less_equal, 10).matches(11));
	EXPECT(make(FuzzyOp::greater, 10).matches(11));
	EXPECT(!make(FuzzyOp::greater, 10).matches(10));
	EXPECT(make(FuzzyOp::greater_equal, 10).matches(10));
	EXPECT(!make(FuzzyOp::greater_equal, 10).matches(9));

	// 0 matches with relational ops (empty-draw corner cases).
	EXPECT(make(FuzzyOp::less, 1).matches(0));
	EXPECT(make(FuzzyOp::greater_equal, 0).matches(0));
}

WWMI_TEST(matches_draw_info_semantics)
{
	TextureOverrideRule rule;
	rule.hash = 0x11223344u;
	rule.has_hash = true;

	DrawCallInfo call;
	call.vertex_count = 100;
	call.index_count = 5400;
	call.instance_count = 1;
	call.first_vertex = 0;
	call.first_index = 6;
	call.first_instance = 0;

	// A rule without match_* keys matches every draw (3DMigoto
	// MatchDrawContext with all matchers disabled).
	EXPECT(matches_draw_info(rule, call));

	// Single matcher: only that field constrains the result.
	rule.match_index_count = { FuzzyOp::equal, 5400, true };
	EXPECT(matches_draw_info(rule, call));
	rule.match_index_count.value = 5401;
	EXPECT(!matches_draw_info(rule, call));

	// Multiple matchers: all must pass (AND).
	rule.match_index_count.value = 5400;
	rule.match_first_index = { FuzzyOp::greater, 0, true };
	EXPECT(matches_draw_info(rule, call));
	rule.match_first_index.value = 100; // first_index 6 is not > 100
	EXPECT(!matches_draw_info(rule, call));

	// Absent matchers impose no constraint: a rule that only filters
	// index_count still matches draws with any vertex_count/instances.
	TextureOverrideRule single;
	single.match_instance_count = { FuzzyOp::not_equal, 0, true };
	DrawCallInfo instanced = call;
	instanced.instance_count = 4;
	EXPECT(matches_draw_info(single, instanced));
	EXPECT(matches_draw_info(single, call)); // instance_count == 1 != 0
	DrawCallInfo noninstanced = call;
	noninstanced.instance_count = 0;
	EXPECT(!matches_draw_info(single, noninstanced));
}

WWMI_TEST(load_mod_rules_handling_and_match_keys)
{
	const std::filesystem::path dir = std::filesystem::temp_directory_path() / "wwmi_test_mod_m2";
	std::filesystem::create_directories(dir);
	const std::filesystem::path ini = dir / "mod.ini";

	{
		std::ofstream f(ini, std::ios::binary);
		f << "; part-hiding mod (M2)\n"
			<< "[TextureOverrideSkirt]\n"
			<< "hash = 0x00aa00aa\n"
			<< "handling = skip\n"
			<< "match_index_count = 4158\n"
			<< "match_first_index = 1234\n"
			<< "\n"
			<< "[TextureOverrideAbortApprox]\n"
			<< "hash = 0x00bb00bb\n"
			<< "handling = abort\n"
			<< "\n"
			<< "[TextureOverrideNoHandling]\n"
			<< "hash = 0x00cc00cc\n"
			<< "match_vertex_count = >= 3000\n"
			<< "\n"
			<< "[TextureOverrideBadHandling]\n"
			<< "hash = 0x00dd00dd\n"
			<< "handling = next\n"
			<< "\n"
			<< "[TextureOverrideBadMatch]\n"
			<< "hash = 0x00ee00ee\n"
			<< "match_index_count = width\n";
	}

	ModRules rules;
	EXPECT(load_mod_rules(ini, rules));
	EXPECT_EQ(rules.overrides.size(), static_cast<size_t>(5));
	if (rules.overrides.size() != 5)
	{
		std::filesystem::remove_all(dir);
		return;
	}

	// [TextureOverrideSkirt]: skip + two matchers.
	const TextureOverrideRule &skirt = rules.overrides[0];
	EXPECT_EQ(static_cast<int>(skirt.handling), static_cast<int>(HandlingMode::skip));
	EXPECT(skirt.has_draw_context_match);
	EXPECT(skirt.match_index_count.enabled);
	EXPECT_EQ(skirt.match_index_count.value, 4158u);
	EXPECT_EQ(static_cast<int>(skirt.match_index_count.op), static_cast<int>(FuzzyOp::equal));
	EXPECT(skirt.match_first_index.enabled);
	EXPECT_EQ(skirt.match_first_index.value, 1234u);
	EXPECT(!skirt.match_vertex_count.enabled);
	EXPECT_EQ(skirt.bindings.size(), static_cast<size_t>(0));

	DrawCallInfo hit;
	hit.index_count = 4158;
	hit.first_index = 1234;
	DrawCallInfo miss;
	miss.index_count = 4158;
	miss.first_index = 9999;
	EXPECT(matches_draw_info(skirt, hit));
	EXPECT(!matches_draw_info(skirt, miss));

	// [TextureOverrideAbortApprox]: abort approximated as skip.
	EXPECT_EQ(static_cast<int>(rules.overrides[1].handling), static_cast<int>(HandlingMode::abort));
	EXPECT(!rules.overrides[1].has_draw_context_match);

	// [TextureOverrideNoHandling]: matchers without handling (legal:
	// such a rule just never skips; it can still bind textures later).
	EXPECT_EQ(static_cast<int>(rules.overrides[2].handling), static_cast<int>(HandlingMode::none));
	EXPECT(rules.overrides[2].match_vertex_count.enabled);
	EXPECT_EQ(static_cast<int>(rules.overrides[2].match_vertex_count.op), static_cast<int>(FuzzyOp::greater_equal));

	// [TextureOverrideBadHandling]/[BadMatch]: warned, section kept with
	// default handling / matcher disabled.
	EXPECT_EQ(static_cast<int>(rules.overrides[3].handling), static_cast<int>(HandlingMode::none));
	EXPECT(!rules.overrides[4].match_index_count.enabled);
	EXPECT(!rules.overrides[4].has_draw_context_match);
	EXPECT_EQ(rules.warnings.size(), static_cast<size_t>(2));

	std::filesystem::remove_all(dir);
}

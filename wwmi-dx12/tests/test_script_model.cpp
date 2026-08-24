#include "script_model.hpp"
#include "ini_file.hpp"
#include "test_framework.hpp"

#include <string>
#include <vector>

using namespace wwmi;

namespace
{
	// Raw lines (already 'key = value' pairs plus key-only block lines).
	using Lines = std::vector<std::pair<std::string, std::string>>;

	CommandList parse(const Lines &lines, std::vector<std::string> *warn = nullptr)
	{
		CommandList cl;
		cl.name = "test";
		std::vector<std::string> w;
		parse_script_body("[Test]", lines, cl, w);
		if (warn != nullptr)
			*warn = std::move(w);
		return cl;
	}
}

WWMI_TEST(script_if_else_endif_flattening)
{
	// $object_detected ? run A : { run B; run C }
	CommandList cl = parse({
		{"if $object_detected", ""},
		{"run", "CommandListA"},
		{"else", ""},
		{"run", "CommandListB"},
		{"run", "CommandListC"},
		{"endif", ""},
		{"run", "CommandListD"},
	});

	// body: 0 jump_false, 1 run A, 2 jump(else), 3 run B, 4 run C,
	//       5 run D  (endif resolves targets)
	EXPECT_EQ(cl.body.size(), static_cast<size_t>(6));
	EXPECT(cl.body[0].kind == Command::Kind::jump_false);
	EXPECT(cl.body[1].kind == Command::Kind::run);
	EXPECT_EQ(cl.body[1].target, "commandlista");
	EXPECT(cl.body[2].kind == Command::Kind::jump);
	EXPECT_EQ(cl.body[0].jump_target, 3u);  // false -> after else
	EXPECT_EQ(cl.body[2].jump_target, 5u);  // else-jump -> after endif
	EXPECT(cl.body[5].kind == Command::Kind::run);
	EXPECT_EQ(cl.body[5].target, "commandlistd");
}

WWMI_TEST(script_nested_if_no_else)
{
	CommandList cl = parse({
		{"if $a", ""},
		{"if $b", ""},
		{"$x", "1"},
		{"endif", ""},
		{"$y", "2"},
		{"endif", ""},
	});

	EXPECT_EQ(cl.body.size(), static_cast<size_t>(4));
	EXPECT(cl.body[0].kind == Command::Kind::jump_false);
	EXPECT(cl.body[1].kind == Command::Kind::jump_false);
	EXPECT_EQ(cl.body[1].jump_target, 3u);   // inner if-false -> $y (skips $x)
	EXPECT_EQ(cl.body[0].jump_target, 4u);   // outer endif -> end
	EXPECT(cl.body[2].kind == Command::Kind::assign);
	EXPECT_EQ(cl.body[2].target, "x");
}

WWMI_TEST(script_assign_post_and_drawindexed)
{
	CommandList cl = parse({
		{"post $object_detected", "0"},
		{"$state_id", "$state_id + 1"},
		{"drawindexed", "17970, 0, 0"},
		{"drawindexed", "22299, 161709, 0, 2"},
		{"handling", "skip"},
	});

	EXPECT_EQ(cl.body.size(), static_cast<size_t>(5));
	EXPECT(cl.body[0].kind == Command::Kind::post_assign);
	EXPECT_EQ(cl.body[0].target, "object_detected");
	EXPECT(cl.body[1].kind == Command::Kind::assign);
	EXPECT(cl.body[1].value != nullptr);
	EXPECT(cl.body[2].kind == Command::Kind::drawindexed);
	EXPECT_EQ(cl.body[2].index_count, 17970u);
	EXPECT_EQ(cl.body[2].first_index, 0u);
	EXPECT_EQ(cl.body[2].base_vertex, 0);
	EXPECT(cl.body[3].kind == Command::Kind::drawindexed);
	EXPECT_EQ(cl.body[3].index_count, 22299u);
	EXPECT_EQ(cl.body[3].first_index, 161709u);
	EXPECT_EQ(cl.body[3].instance_count, 2u);
	EXPECT(cl.body[4].kind == Command::Kind::handling_skip);
}

WWMI_TEST(script_bindings_and_resource_ops)
{
	CommandList cl = parse({
		{"ResourceBypassVB0", "ref vb0"},
		{"ib", "ResourceIndexBuffer"},
		{"vb0", "ResourcePositionBuffer"},
		{"vb4", "ref ResourceBlendBufferOverride"},
		{"cs-t34", "ref ResourceBlendRemapReverseBuffer"},
		{"cs-u4", "ref ResourceRemappedBlendBufferRW"},
		{"ResourceRemappedBlendBufferRW", "copy ResourceBlendBufferNoStride"},
		{"ResourceX", "copy_desc ResourceY"},
		{"ResourceMergedSkeletonOverride", "ref ResourceMergedSkeleton"},
		{"ResourceBlendBufferOverride", "null"},
		{"CheckTextureOverride", "ps-t0"},
	});

	EXPECT_EQ(cl.body.size(), static_cast<size_t>(11));
	EXPECT(cl.body[0].kind == Command::Kind::res_capture);
	EXPECT_EQ(cl.body[0].target, "vb0");
	EXPECT_EQ(cl.body[0].dest, "resourcebypassvb0");
	EXPECT(cl.body[1].kind == Command::Kind::bind_ib);
	EXPECT_EQ(cl.body[1].target, "resourceindexbuffer");
	EXPECT(cl.body[2].kind == Command::Kind::bind_vb);
	EXPECT_EQ(cl.body[2].slot, 0u);
	EXPECT_EQ(cl.body[2].target, "resourcepositionbuffer");
	EXPECT(cl.body[3].kind == Command::Kind::bind_vb);
	EXPECT_EQ(cl.body[3].slot, 4u);
	EXPECT(cl.body[3].is_ref, true);
	EXPECT(cl.body[4].kind == Command::Kind::bind_slot);
	EXPECT_EQ(cl.body[4].stage, 'c');
	EXPECT_EQ(cl.body[4].slot_type, 't');
	EXPECT_EQ(cl.body[4].slot, 34u);
	EXPECT(cl.body[5].kind == Command::Kind::bind_slot);
	EXPECT_EQ(cl.body[5].slot_type, 'u');
	EXPECT_EQ(cl.body[5].slot, 4u);
	EXPECT(cl.body[6].kind == Command::Kind::res_copy);
	EXPECT_EQ(cl.body[6].dest, "resourceremappedblendbufferrw");
	EXPECT_EQ(cl.body[6].target, "resourceblendbuffernostride");
	EXPECT(cl.body[7].kind == Command::Kind::res_copy_desc);
	EXPECT(cl.body[8].kind == Command::Kind::res_ref);
	EXPECT(cl.body[9].kind == Command::Kind::res_null);
	EXPECT(cl.body[10].kind == Command::Kind::check_override);
	EXPECT_EQ(cl.body[10].stage, 'p');
	EXPECT_EQ(cl.body[10].slot, 0u);
}

WWMI_TEST(script_namespaced_warn_but_parse)
{
	std::vector<std::string> warnings;
	CommandList cl = parse({
		{"run", "CommandList\\WWMIv1\\RegisterMod"},
		{"$\\WWMIv1\\blend_remap_id", "0"},
		{"Resource\\WWMIv1\\ModName", "ref ResourceModName"},
	}, &warnings);

	EXPECT_EQ(cl.body.size(), static_cast<size_t>(3));
	EXPECT(cl.body[0].kind == Command::Kind::run);
	EXPECT(cl.body[1].kind == Command::Kind::assign);
	EXPECT_EQ(cl.body[1].target, "\\wwmiv1\\blend_remap_id");
	// namespaced run/resource flagged
	EXPECT(warnings.size() >= 2);
}

WWMI_TEST(script_unbalanced_blocks)
{
	std::vector<std::string> warnings;
	CommandList cl = parse({
		{"if $a", ""},
		{"$x", "1"},
	}, &warnings);

	// Unterminated if: jumps to end, warning emitted
	EXPECT_EQ(cl.body.size(), static_cast<size_t>(2));
	EXPECT_EQ(cl.body[0].jump_target, 2u);
	EXPECT(!warnings.empty());
}

WWMI_TEST(script_unsupported_keys_warn)
{
	std::vector<std::string> warnings;
	CommandList cl = parse({
		{"blend", "ADD BLEND_FACTOR INV_BLEND_FACTOR"},
		{"blend_factor[0]", "0.8"},
		{"totally_unknown_key", "1"},
	}, &warnings);

	// M4: blend/blend_factor parse to blend_state commands; only the
	// unknown key warns.
	EXPECT_EQ(cl.body.size(), static_cast<size_t>(2));
	EXPECT(cl.body[0].kind == Command::Kind::blend_state);
	EXPECT(cl.body[0].blend.has_color);
	EXPECT(cl.body[0].blend.op == 0);
	EXPECT(cl.body[0].blend.src == 10);
	EXPECT(cl.body[0].blend.dst == 11);
	EXPECT(cl.body[1].kind == Command::Kind::blend_state);
	EXPECT(cl.body[1].blend.has_factors);
	EXPECT_EQ(cl.body[1].blend.factor_mask, 1u);
	EXPECT(cl.body[1].blend.factors[0] == 0.8f);
	EXPECT_EQ(warnings.size(), static_cast<size_t>(1)); // only the unknown key
}

WWMI_TEST(script_key_section_parsing)
{
	std::vector<std::string> warnings;
	KeyBinding key;
	bool ok = parse_key_section("[Keyleg]", {
		{"condition", "$object_detected"},
		{"key", "VK_RIGHT"},
		{"type", "cycle"},
		{"$leg", "0,1,2"},
	}, key, warnings);

	EXPECT(ok);
	EXPECT_EQ(key.var, "leg");
	EXPECT_EQ(key.values.size(), static_cast<size_t>(3));
	EXPECT_EQ(key.values[0], 0.0f);
	EXPECT_EQ(key.values[2], 2.0f);
	EXPECT_EQ(key.vk, 0x27u); // VK_RIGHT
	EXPECT(key.condition != nullptr);

	EXPECT_EQ(parse_vk("VK_UP"), 0x26u);
	EXPECT_EQ(parse_vk(","), 0xBCu); // ',' -> VK_OEM_COMMA region
	EXPECT_EQ(parse_vk("9"), '9');
	EXPECT_EQ(parse_vk("nope"), 0u);
}

WWMI_TEST(script_full_ini_roundtrip_lynne_style)
{
	// End-to-end: ini text -> sections -> script body (Lynae excerpt)
	IniFile ini;
	EXPECT(parse_ini(
		"[CommandListOverrideSharedResources]\n"
		"ResourceBypassVB0 = ref vb0\n"
		"ib = ResourceIndexBuffer\n"
		"vb0 = ResourcePositionBuffer\n"
		"if ResourceBlendBufferOverride === null\n"
		"vb4 = ResourceBlendBuffer\n"
		"endif\n"
		, ini));

	const IniSection *sec = nullptr;
	for (const IniSection &s : ini.sections)
		if (to_lower(s.name) == "commandlistoverridesharedresources")
			sec = &s;
	EXPECT(sec != nullptr);
	if (sec == nullptr)
		return;

	Lines lines;
	for (const IniEntry &e : sec->entries)
		lines.push_back({ e.key, e.value });

	std::vector<std::string> warnings;
	CommandList cl = parse(lines, &warnings);
	EXPECT_EQ(cl.body.size(), static_cast<size_t>(5));
	EXPECT(cl.body[3].kind == Command::Kind::jump_false); // 'if' preserved
	EXPECT(warnings.empty());
}

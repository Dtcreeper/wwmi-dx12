#include "ini_file.hpp"
#include "test_framework.hpp"

using namespace wwmi;

WWMI_TEST(ini_basic_sections_and_keys)
{
	const char *text =
		"; leading comment\n"
		"[SectionOne]\n"
		"key = value\n"
		"another=  spaced \n"
		"\n"
		"[SectionTwo]\n"
		"foo=bar\n";

	IniFile ini;
	EXPECT(parse_ini(text, ini));
	EXPECT_EQ(ini.sections.size(), static_cast<size_t>(2));
	EXPECT_EQ(ini.sections[0].name, "SectionOne");
	EXPECT_EQ(ini.sections[0].entries.size(), static_cast<size_t>(2));
	EXPECT_EQ(ini.sections[0].entries[0].key, "key");
	EXPECT_EQ(ini.sections[0].entries[0].value, "value");
	EXPECT_EQ(ini.sections[0].entries[1].key, "another");
	EXPECT_EQ(ini.sections[0].entries[1].value, "spaced");
	EXPECT_EQ(ini.sections[1].entries[0].value, "bar");
}

WWMI_TEST(ini_inline_semicolon_is_not_a_comment)
{
	// 3DMigoto semantics: only whole-line ';' comments; a ';' after the
	// value is part of the value. Keys before any section are dropped.
	IniFile ini;
	EXPECT(parse_ini("[S]\nk = val ; trailing\n", ini));
	EXPECT_EQ(ini.sections.size(), static_cast<size_t>(1));
	EXPECT_EQ(ini.sections[0].entries[0].value, "val ; trailing");
}

WWMI_TEST(ini_duplicate_sections_merge_and_keep_order)
{
	IniFile ini;
	EXPECT(parse_ini("[A]\nk1 = v1\n[A]\nk2 = v2\nk1 = v1b\n", ini));
	EXPECT_EQ(ini.sections.size(), static_cast<size_t>(1));
	EXPECT_EQ(ini.sections[0].entries.size(), static_cast<size_t>(3));
	EXPECT_EQ(ini.sections[0].entries[0].value, "v1");
	EXPECT_EQ(ini.sections[0].entries[2].value, "v1b"); // duplicates preserved
}

WWMI_TEST(ini_case_insensitive_lookup)
{
	IniFile ini;
	EXPECT(parse_ini("[TextureOverrideHair]\nHASH = 0x1234\n", ini));

	const std::string *v = find_value(ini, "textureoverridehair", "hash");
	EXPECT(v != nullptr);
	if (v != nullptr)
		EXPECT_EQ(*v, "0x1234");

	EXPECT(find_value(ini, "TextureOverrideHair", "nope") == nullptr);
	EXPECT(find_value(ini, "Missing", "hash") == nullptr);
}

WWMI_TEST(ini_malformed_lines_skipped)
{
	IniFile ini;
	EXPECT(parse_ini("[S]\nno_equals_here\n= no_key\n[bad section\nok = 1\n", ini));
	EXPECT_EQ(ini.sections.size(), static_cast<size_t>(1));
	EXPECT_EQ(ini.sections[0].entries.size(), static_cast<size_t>(1));
	EXPECT_EQ(ini.sections[0].entries[0].key, "ok");
}

WWMI_TEST(ini_entry_outside_section_dropped)
{
	IniFile ini;
	EXPECT(parse_ini("orphan = 1\n[S]\nok = 2\n", ini));
	EXPECT_EQ(ini.sections.size(), static_cast<size_t>(1));
	EXPECT_EQ(ini.sections[0].entries.size(), static_cast<size_t>(1));
}

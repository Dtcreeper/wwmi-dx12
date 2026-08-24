// WWMI-DX12: minimal INI file parser with 3DMigoto-compatible semantics.
//
// Semantics mirrored from 3DMigoto (IniHandler.cpp / ParseIniStream):
//  - Only lines whose first non-whitespace char is ';' are comments.
//    A ';' in the middle of a line is NOT a comment.
//  - [Section] lines start a section; duplicate sections are merged.
//  - 'key = value': whitespace around '=' is stripped; order and duplicate
//    keys are preserved (command-list sections rely on ordered duplicates).
//  - Keys are matched case-insensitively; values keep their case.
#pragma once

#include <string>
#include <string_view>
#include <vector>

namespace wwmi
{
	struct IniEntry
	{
		std::string key;   // original case
		std::string value; // original case
	};

	struct IniSection
	{
		std::string name;       // original case
		std::vector<IniEntry> entries; // in file order, duplicates preserved
	};

	struct IniFile
	{
		std::vector<IniSection> sections;
	};

	// Parses text into sections. Returns false on catastrophic error only;
	// malformed lines are skipped (with nothing recorded, like 3DMigoto's
	// "malformed line" warnings).
	bool parse_ini(std::string_view text, IniFile &out);

	// Case-insensitive section/key lookup. Sections are matched by name with
	// ASCII case folding. Returns nullptr when not found.
	const std::string *find_value(const IniFile &file, std::string_view section, std::string_view key);

	// ASCII lowercase helper.
	std::string to_lower(std::string_view s);

	// Checks whether a string starts with a prefix, ASCII case-insensitively.
	bool istarts_with(std::string_view s, std::string_view prefix);

	// Full string equality, ASCII case-insensitive.
	bool iequals(std::string_view a, std::string_view b);
}

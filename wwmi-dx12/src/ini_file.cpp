#include "ini_file.hpp"

namespace wwmi
{
	namespace
	{
		bool is_space(char c)
		{
			return c == ' ' || c == '\t' || c == '\r' || c == '\n';
		}

		std::string_view trim(std::string_view s)
		{
			size_t first = 0, last = s.size();
			while (first < last && is_space(s[first]))
				++first;
			while (last > first && is_space(s[last - 1]))
				--last;
			return s.substr(first, last - first);
		}

		char lower_char(char c)
		{
			return (c >= 'A' && c <= 'Z') ? static_cast<char>(c - 'A' + 'a') : c;
		}
	}

	std::string to_lower(std::string_view s)
	{
		std::string out;
		out.reserve(s.size());
		for (char c : s)
			out.push_back(lower_char(c));
		return out;
	}

	bool istarts_with(std::string_view s, std::string_view prefix)
	{
		if (s.size() < prefix.size())
			return false;
		for (size_t i = 0; i < prefix.size(); ++i)
			if (lower_char(s[i]) != lower_char(prefix[i]))
				return false;
		return true;
	}

	bool iequals(std::string_view a, std::string_view b)
	{
		if (a.size() != b.size())
			return false;
		for (size_t i = 0; i < a.size(); ++i)
			if (lower_char(a[i]) != lower_char(b[i]))
				return false;
		return true;
	}

	bool parse_ini(std::string_view text, IniFile &out)
	{
		out.sections.clear();

		IniSection *current = nullptr;

		size_t pos = 0;
		while (pos < text.size())
		{
			size_t eol = text.find('\n', pos);
			std::string_view line = text.substr(pos, eol == std::string_view::npos ? std::string_view::npos : eol - pos);
			pos = (eol == std::string_view::npos) ? text.size() : eol + 1;

			line = trim(line);
			if (line.empty())
				continue;

			// Only whole-line comments (';' as first non-whitespace char).
			if (line[0] == ';')
				continue;

			if (line.front() == '[')
			{
				const size_t close = line.find(']');
				if (close == std::string_view::npos || close == 1)
					continue; // malformed section line

				std::string_view name = trim(line.substr(1, close - 1));
				if (name.empty())
					continue;

				// Merge duplicate sections (case-insensitive).
				const std::string lower_name = to_lower(name);
				current = nullptr;
				for (IniSection &sec : out.sections)
				{
					if (to_lower(sec.name) == lower_name)
					{
						current = &sec;
						break;
					}
				}
				if (current == nullptr)
				{
					out.sections.push_back({ std::string(name), {} });
					current = &out.sections.back();
				}
				continue;
			}

			const size_t delim = line.find('=');
			if (delim == std::string_view::npos)
			{
				// 3DMigoto script keywords arrive WITHOUT '=' ('if $x',
				// 'else', 'endif'). Keep them as key-only entries so the
				// command-list parser sees the block structure; every
				// other bare line stays skipped.
				std::string_view word = line;
				const size_t sp = word.find(' ');
				if (sp != std::string_view::npos)
					word = word.substr(0, sp);
				const std::string w = to_lower(word);
				if ((w == "if" || w == "else" || w == "endif") && current != nullptr)
					current->entries.push_back({ std::string(line), std::string() });
				continue;
			}

			std::string_view key = trim(line.substr(0, delim));
			std::string_view val = trim(line.substr(delim + 1));
			if (key.empty())
				continue;

			if (current == nullptr)
				continue; // entry outside of any section

			current->entries.push_back({ std::string(key), std::string(val) });
		}

		return true;
	}

	const std::string *find_value(const IniFile &file, std::string_view section, std::string_view key)
	{
		for (const IniSection &sec : file.sections)
		{
			if (!iequals(sec.name, section))
				continue;
			for (const IniEntry &e : sec.entries)
			{
				if (iequals(e.key, key))
					return &e.value;
			}
		}
		return nullptr;
	}
}

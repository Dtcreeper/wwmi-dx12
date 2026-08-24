#include "mod_rules.hpp"

#include <cstdio>
#include <fstream>

namespace wwmi
{
	namespace
	{
		// 3DMigoto TextureOverrideIniKeys whitelist: keys handled outside the
		// command-list parser. M1 recognises 'hash' and records the rest as
		// ignored (no warning spam for well-formed mods).
		bool is_texture_override_whitelisted_key(const std::string &lower_key)
		{
			static const char *keys[] = {
				"hash", "stereomode", "format", "width", "height",
				"width_multiply", "height_multiply", "iteration", "filter_index",
				"expand_region_copy", "deny_cpu_read", "match_priority",
				"draw", "match_first_vertex", "match_first_index",
				"match_first_instance", "match_vertex_count", "match_index_count",
				"match_instance_count",
			};
			for (const char *k : keys)
				if (lower_key == k)
					return true;
			// Fuzzy-match keys ('match_type', 'match_width', ...) cannot be
			// combined with hash in 3DMigoto; treat the whole family as
			// whitelisted so it at least parses.
			if (lower_key.rfind("match_", 0) == 0)
				return true;
			return false;
		}

		// Inline-script lines inside a TextureOverride body ('handling=skip'
		// draw-intercept rules). The ScriptRuntime owns these; the rule
		// parser must not warn about them. Covers:
		//   '$var = ...'            variable assignment (key starts with $)
		//   'if ...' / 'else' / 'endif'   conditionals. Complex conditions
		//                          contain '==' so the INI splitter cuts the
		//                          line at the first '=' ('if ($leg').
		//   'run' / 'drawindexed'   command-list calls
		bool is_inline_script_key(const std::string &lower_key)
		{
			if (!lower_key.empty() && (lower_key[0] == '$' || lower_key[0] == '%'))
				return true;
			if (lower_key == "else" || lower_key == "endif" || lower_key == "run" ||
				lower_key == "drawindexed" || lower_key == "draw" ||
				lower_key == "post" || lower_key == "pre")
				return true;
			if (lower_key.rfind("if ", 0) == 0 || lower_key == "if")
				return true;
			return false;
		}

		// Copy-option words that may prefix a binding value
		// (ResourceCopyOptionNames). All are argument-less.
		bool is_copy_option_word(const std::string &token)
		{
			static const char *words[] = {
				"copy", "ref", "reference", "copy_desc", "copy_description",
				"unless_null", "stereo", "mono", "stereo2mono", "set_viewport",
				"no_view_cache", "raw", "resolve_msaa",
			};
			for (const char *w : words)
				if (token == w)
					return true;
			return false;
		}

		bool is_stage_char(char c)
		{
			return c == 'v' || c == 'h' || c == 'd' || c == 'g' || c == 'p' || c == 'c';
		}

		bool is_digits(std::string_view s)
		{
			if (s.empty())
				return false;
			for (char c : s)
				if (c < '0' || c > '9')
					return false;
			return true;
		}

		// Other valid 3DMigoto slot-binding commands that M1 parses but does
		// not act on (constant buffers, UAVs, vertex/index buffers, RT/DS
		// slots, stream output). Silently ignored so real mods don't drown
		// the warnings log.
		bool is_other_slot_binding_key(const std::string &lower_key)
		{
			const size_t n = lower_key.size();

			// '<stage>s-cbN' / '<stage>s-uN' / '<stage>s-sN'
			if (n >= 6 && is_stage_char(lower_key[0]) && lower_key[1] == 's' && lower_key[2] == '-')
			{
				if (lower_key[3] == 'c' && lower_key[4] == 'b' && is_digits(std::string_view(lower_key).substr(5)))
					return true;
				if (lower_key[3] == 'u' && is_digits(std::string_view(lower_key).substr(4)))
					return true;
				if (lower_key[3] == 's' && is_digits(std::string_view(lower_key).substr(4)))
					return true;
			}

			// 'vbN' / 'soN' / 'oN' / 'od' / 'ib'
			if (n >= 3 && lower_key.compare(0, 2, "vb") == 0 && is_digits(std::string_view(lower_key).substr(2)))
				return true;
			if (n >= 3 && lower_key.compare(0, 2, "so") == 0 && is_digits(std::string_view(lower_key).substr(2)))
				return true;
			if (n >= 2 && lower_key[0] == 'o' && is_digits(std::string_view(lower_key).substr(1)))
				return true;
			if (lower_key == "od" || lower_key == "ib")
				return true;

			return false;
		}

		// Special source targets in ResourceCopyTarget::ParseTarget that are
		// not custom resources. Unsupported in M1 (replacement-only path).
		bool is_special_source(const std::string &token)
		{
			static const char *words[] = {
				"null", "this", "bb", "r_bb", "f_bb", "stereoparams",
				"iniparams", "cursor_mask", "cursor_color",
			};
			for (const char *w : words)
				if (token == w)
					return true;
			return false;
		}

		void add_warning(ModRules &rules, const std::string &msg)
		{
			rules.warnings.push_back(msg);
		}

		void parse_resource_sections(const IniFile &ini, ModRules &rules)
		{
			for (const IniSection &sec : ini.sections)
			{
				if (!istarts_with(sec.name, "resource") || sec.name.size() <= 8)
					continue;

				ResourceDef def;
				def.section = sec.name;

				for (const IniEntry &e : sec.entries)
				{
					const std::string key = to_lower(e.key);
					if (key == "filename")
						def.filename = e.value;
					else if (key == "format")
						def.format = to_lower(e.value);
					else if (key == "width")
						def.width = atoi(e.value.c_str());
					else if (key == "height")
						def.height = atoi(e.value.c_str());
					else if (key == "depth")
						def.depth = atoi(e.value.c_str());
					else if (key == "mips")
						def.mips = atoi(e.value.c_str());
					else if (key == "array")
						def.array = atoi(e.value.c_str());
					// Other keys (bind_flags etc.) are ignored in M1.
				}

				rules.resources[to_lower(sec.name)] = std::move(def);
			}
		}

		// M4: [ShaderOverride<name>] sections. The parser records only the
		// 64-bit shader hash; the body is an inline script executed by the
		// ScriptRuntime (m4-2), so all other keys are skipped silently.
		void parse_shader_override_sections(const IniFile &ini, ModRules &rules)
		{
			for (const IniSection &sec : ini.sections)
			{
				if (!istarts_with(sec.name, "shaderoverride") || sec.name.size() <= 15)
					continue;

				ShaderOverrideRule rule;
				rule.section = sec.name;

				for (const IniEntry &e : sec.entries)
				{
					std::string key = to_lower(e.key);
					if (key.rfind("post ", 0) == 0)
						key = key.substr(5);
					else if (key.rfind("pre ", 0) == 0)
						key = key.substr(4);

					if (key == "hash")
					{
						// Shader hashes stay 64-bit (XXH64), unlike the
						// 32-bit texture hashes truncated above.
						uint64_t hash = 0;
						if (parse_hash(e.value, hash))
						{
							rule.hash = hash;
							rule.has_hash = true;
						}
						else
							add_warning(rules, "[" + sec.name + "] hash parse error: " + e.value);
						continue;
					}
					// Everything else (draw = ..., $var = ..., if/endif,
					// ps-t0 = ...) belongs to the script runtime.
				}

				if (rule.has_hash)
					rules.shader_overrides.push_back(std::move(rule));
				else
					add_warning(rules, "[" + sec.name + "] ShaderOverride without a valid hash: ignored");
			}
		}

		void parse_texture_override_sections(const IniFile &ini, ModRules &rules)
		{
			for (const IniSection &sec : ini.sections)
			{
				if (!istarts_with(sec.name, "textureoverride") || sec.name.size() <= 15)
					continue;

				TextureOverrideRule rule;
				rule.section = sec.name;

				for (const IniEntry &e : sec.entries)
				{
					std::string key = to_lower(e.key);

					// 'post '/'pre ' prefixes select the command list phase;
					// M1 applies all bindings at match time.
					if (key.rfind("post ", 0) == 0)
						key = key.substr(5);
					else if (key.rfind("pre ", 0) == 0)
						key = key.substr(4);

					if (key == "hash")
				{
					uint64_t hash = 0;
					if (parse_hash(e.value, hash))
					{
						rule.hash = static_cast<uint32_t>(hash); // texture hashes are 32-bit
						rule.has_hash = true;
					}
					else
					{
						add_warning(rules, "[" + sec.name + "] hash parse error: " + e.value);
					}
					continue;
				}

				// M2: handling = skip|abort (draw interception).
				if (key == "handling")
				{
					const std::string val = to_lower(e.value);
					if (val == "skip")
						rule.handling = HandlingMode::skip;
					else if (val == "abort")
						rule.handling = HandlingMode::abort;
					else
						add_warning(rules, "[" + sec.name + "] handling = " + e.value +
							": unsupported (supported: skip, abort)");
					continue;
				}

				// M2: draw-context fuzzy matchers.
				if (key.rfind("match_", 0) == 0)
				{
					FuzzyMatch *m = nullptr;
					if (key == "match_first_vertex")
						m = &rule.match_first_vertex;
					else if (key == "match_first_index")
						m = &rule.match_first_index;
					else if (key == "match_first_instance")
						m = &rule.match_first_instance;
					else if (key == "match_vertex_count")
						m = &rule.match_vertex_count;
					else if (key == "match_index_count")
						m = &rule.match_index_count;
					else if (key == "match_instance_count")
						m = &rule.match_instance_count;

					if (m != nullptr)
					{
						if (parse_fuzzy_match(e.value, *m))
							rule.has_draw_context_match = true;
						else
							add_warning(rules, "[" + sec.name + "] " + e.key + " = " + e.value +
								": cannot parse fuzzy match expression");
					}
					// Other match_* keys (match_type/match_width/... resource
					// fuzzy matching) stay whitelisted-ignored.
					continue;
				}

				// Texture-slot binding line?
				char stage = 0;
				uint32_t slot = 0;
					if (parse_texture_slot_key(key, stage, slot))
					{
						std::string resource;
						if (parse_resource_ref(e.value, resource))
						{
							rule.bindings.push_back({ stage, slot, std::move(resource) });
						}
						else
						{
							add_warning(rules, "[" + sec.name + "] " + e.key + " = " + e.value +
								": unsupported source (expected resource reference)");
						}
						continue;
					}

					// Whitelisted/unknown keys: silently ignore the whitelist,
					// other slot-binding commands and inline-script lines
					// (owned by the ScriptRuntime); warn on genuinely
					// unknown keys to help mod authors.
					if (!is_texture_override_whitelisted_key(key) &&
						!is_other_slot_binding_key(key) &&
						!is_inline_script_key(key))
						add_warning(rules, "[" + sec.name + "] unrecognised key: " + e.key);
				}

				if (!rule.has_hash)
				{
					add_warning(rules, "[" + sec.name + "] missing valid hash=, section ignored");
					continue;
				}

				rules.overrides.push_back(std::move(rule));
			}
		}
	}

	bool parse_hash(const std::string &value, uint64_t &out_hash)
	{
		// Mirrors 3DMigoto GetIniHash: '%16llx%n' must consume everything.
		unsigned long long hash = 0;
		int len = 0;
		const int ret = sscanf_s(value.c_str(), "%16llx%n", &hash, &len);
		if (ret != 1 || len != static_cast<int>(value.size()))
			return false;
		out_hash = hash;
		return true;
	}

	bool parse_texture_slot_key(std::string_view key, char &out_stage, uint32_t &out_slot)
	{
		// 3DMigoto ParseTarget: '%1cs-t%u' -> e.g. "ps-t0" is
		// stage char 'p', literal "s-t", then the slot number.
		// Stage set: {v,h,d,g,p,c}; slot < 128
		// (D3D11_COMMONSHADER_INPUT_RESOURCE_SLOT_COUNT).
		// Keys are case-insensitive (3DMigoto lowercases before parsing).
		const std::string lowered = to_lower(key);
		if (lowered.size() < 5 || lowered[1] != 's' || lowered[2] != '-' || lowered[3] != 't')
			return false;

		const char stage = lowered[0];
		if (stage != 'v' && stage != 'h' && stage != 'd' && stage != 'g' && stage != 'p' && stage != 'c')
			return false;

		const std::string_view num = std::string_view(lowered).substr(4);
		if (num.empty() || num.size() > 3)
			return false;
		for (char c : num)
			if (c < '0' || c > '9')
				return false;

		const uint32_t slot = static_cast<uint32_t>(atoi(std::string(num).c_str()));
		if (slot >= 128)
			return false;

		out_stage = stage;
		out_slot = slot;
		return true;
	}

	bool parse_resource_ref(std::string_view value, std::string &out_resource)
	{
		std::string lowered = to_lower(value);

		// Skip leading whitespace and copy-option words.
		size_t pos = 0;
		bool had_options = false;
		while (pos < lowered.size())
		{
			while (pos < lowered.size() && (lowered[pos] == ' ' || lowered[pos] == '\t'))
				++pos;
			const size_t start = pos;
			while (pos < lowered.size() && lowered[pos] != ' ' && lowered[pos] != '\t')
				++pos;
			if (start == pos)
				break;
			const std::string token = lowered.substr(start, pos - start);
			if (is_copy_option_word(token))
			{
				had_options = true;
				continue;
			}
			// First non-option token is the source reference.
			if (token == "resource" || (token.size() > 8 && token.compare(0, 8, "resource") == 0))
			{
				out_resource = token.substr(8); // strip 'resource' prefix
				return !out_resource.empty();
			}
			if (is_special_source(token))
				return false;
			// Bare resource name without the 'resource' prefix is not valid
			// 3DMigoto syntax for custom resources.
			return false;
		}
		(void)had_options;
		return false;
	}

	bool parse_fuzzy_match(const std::string &value, FuzzyMatch &out)
	{
		// Mirrors 3DMigoto parse_fuzzy_numeric_match_expression:
		// an optional operator {=,!,<,>,<=,>=} followed by a plain
		// unsigned integer. Absent operator means '='.
		std::string_view v = value;

		auto skip_ws = [&v]()
		{
			while (!v.empty() && (v.front() == ' ' || v.front() == '\t'))
				v.remove_prefix(1);
		};
		skip_ws();

		FuzzyOp op = FuzzyOp::equal;
		if (!v.empty())
		{
			const char c = v.front();
			if (c == '=' || c == '!')
			{
				op = (c == '=') ? FuzzyOp::equal : FuzzyOp::not_equal;
				v.remove_prefix(1);
			}
			else if (c == '<' || c == '>')
			{
				op = (c == '<') ? FuzzyOp::less : FuzzyOp::greater;
				v.remove_prefix(1);
				if (!v.empty() && v.front() == '=')
				{
					op = (c == '<') ? FuzzyOp::less_equal : FuzzyOp::greater_equal;
					v.remove_prefix(1);
				}
			}
		}

		skip_ws();
		if (!is_digits(v))
			return false;

		out.op = op;
		out.value = static_cast<uint32_t>(strtoul(std::string(v).c_str(), nullptr, 10));
		out.enabled = true;
		return true;
	}

	bool FuzzyMatch::matches(uint32_t v) const
	{
		switch (op)
		{
		case FuzzyOp::equal: return v == value;
		case FuzzyOp::not_equal: return v != value;
		case FuzzyOp::less: return v < value;
		case FuzzyOp::less_equal: return v <= value;
		case FuzzyOp::greater: return v > value;
		case FuzzyOp::greater_equal: return v >= value;
		}
		return false;
	}

	bool matches_draw_info(const TextureOverrideRule &rule, const DrawCallInfo &call)
	{
		// 3DMigoto DrawCallInfo::MatchDrawContext: every enabled matcher
		// must pass; matchers absent from the ini impose no constraint.
		if (rule.match_first_vertex.enabled && !rule.match_first_vertex.matches(call.first_vertex))
			return false;
		if (rule.match_first_index.enabled && !rule.match_first_index.matches(call.first_index))
			return false;
		if (rule.match_first_instance.enabled && !rule.match_first_instance.matches(call.first_instance))
			return false;
		if (rule.match_vertex_count.enabled && !rule.match_vertex_count.matches(call.vertex_count))
			return false;
		if (rule.match_index_count.enabled && !rule.match_index_count.matches(call.index_count))
			return false;
		if (rule.match_instance_count.enabled && !rule.match_instance_count.matches(call.instance_count))
			return false;
		return true;
	}

	bool load_mod_rules(const std::filesystem::path &ini_path, ModRules &out)
	{
		out = ModRules{};
		out.mod_dir = ini_path.parent_path();

		std::ifstream file(ini_path, std::ios::binary);
		if (!file.is_open())
			return false;

		std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

		IniFile ini;
		parse_ini(content, ini);

		parse_resource_sections(ini, out);
		parse_texture_override_sections(ini, out);
		parse_shader_override_sections(ini, out);

		return true;
	}
}

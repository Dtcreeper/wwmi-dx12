#include "script_model.hpp"
#include "ini_file.hpp"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <sstream>

namespace wwmi
{
	std::string normalize_section_name(std::string_view section)
	{
		std::string s = to_lower(section);
		for (const char *prefix : { "commandlist", "customshader", "textureoverride", "shaderoverride" })
		{
			const std::string p(prefix);
			if (s.rfind(p, 0) == 0)
			{
				s.erase(0, p.size());
				break;
			}
		}
		return s;
	}

	bool is_namespaced(std::string_view name)
	{
		return name.find('\\') != std::string_view::npos;
	}

	namespace
	{
		bool is_number(const std::string &s)
		{
			if (s.empty())
				return false;
			char *stop = nullptr;
			std::strtod(s.c_str(), &stop);
			return stop != nullptr && *stop == '\0';
		}

		// Splits "a, b, c" on commas (whitespace trimmed).
		std::vector<std::string> split_csv(const std::string &value)
		{
			std::vector<std::string> out;
			size_t pos = 0;
			for (;;)
			{
				const size_t comma = value.find(',', pos);
				std::string part = value.substr(pos,
					comma == std::string::npos ? std::string::npos : comma - pos);
				const size_t b = part.find_first_not_of(" \t");
				const size_t e = part.find_last_not_of(" \t");
				out.push_back(b == std::string::npos ? std::string() : part.substr(b, e - b + 1));
				if (comma == std::string::npos)
					break;
				pos = comma + 1;
			}
			return out;
		}

		// Strips a leading copy-option keyword ('ref', 'copy', 'stereo',
		// 'from', ...). Returns the remainder.
		std::string strip_bind_option(const std::string &value, bool &was_ref)
		{
			const size_t sp = value.find_first_of(" \t");
			const std::string head = to_lower(sp == std::string::npos ? value : value.substr(0, sp));
			if (head == "ref" || head == "copy" || head == "stereo" || head == "from")
			{
				was_ref = head == "ref";
				const std::string rest = sp == std::string::npos ? std::string() : value.substr(sp + 1);
				const size_t b = rest.find_first_not_of(" \t");
				return b == std::string::npos ? std::string() : rest.substr(b);
			}
			was_ref = false;
			return value;
		}

		// '^vb<N>$' or 'ib' -- a game IA slot usable as a capture source.
		bool is_game_ia_slot(const std::string &name)
		{
			if (name == "ib")
				return true;
			if (name.size() > 2 && name[0] == 'v' && name[1] == 'b')
			{
				for (size_t i = 2; i < name.size(); ++i)
					if (!std::isdigit(static_cast<unsigned char>(name[i])))
						return false;
				return true;
			}
			return false;
		}

		// '<stage>-<type><slot>' binding key: vb3, ib, cs-t34, cs-u4,
		// vs-cb3, ps-t0. Fills stage/slot_type/slot; returns false when
		// the key is not a slot binding.
		bool parse_slot_key(const std::string &key, char &stage, char &slot_type, uint32_t &slot)
		{
			const std::string k = to_lower(key);
			if (k == "ib")
			{
				stage = 'i';
				slot_type = 'i';
				slot = 0;
				return true;
			}
			if (k.size() > 2 && k[0] == 'v' && k[1] == 'b' &&
				std::all_of(k.begin() + 2, k.end(), [](char c) { return std::isdigit(static_cast<unsigned char>(c)); }))
			{
				stage = 'v';
				slot_type = 'v';
				slot = static_cast<uint32_t>(std::strtoul(k.c_str() + 2, nullptr, 10));
				return true;
			}
			// <stage>-<type><slot>: 'cs-t34', 'ps-t0', 'vs-cb3', 'cs-u4'
			const size_t dash = k.find('-');
			if (dash == std::string::npos || dash == 0 || dash + 2 >= k.size() + 1)
				return false;
			if (dash != 2)
				return false; // 'vs'/'ps'/'cs'/'gs'/'hs'/'ds' are 2 chars
			stage = k[0];
			slot_type = k[dash + 1];
			if (slot_type != 't' && slot_type != 'u' && slot_type != 'c')
				return false;
			// 'cb' consumes the 'b'
			const char *num = k.c_str() + dash + 2;
			if (slot_type == 'c' && *num == 'b')
				++num;
			if (*num == '\0')
				return false;
			for (const char *p = num; *p != '\0'; ++p)
				if (!std::isdigit(static_cast<unsigned char>(*p)))
					return false;
			slot = static_cast<uint32_t>(std::strtoul(num, nullptr, 10));
			return true;
		}

		void warn(std::vector<std::string> &warnings, const std::string &section,
			const std::string &line, const char *why)
		{
			warnings.push_back(section + ": '" + line + "': " + why);
		}

		// 'drawindexed = <count>, <first_index>, <base_vertex>[, <instances>]'
		bool parse_drawindexed(const std::string &value, Command &cmd)
		{
			const std::vector<std::string> parts = split_csv(value);
			if (parts.empty() || !is_number(parts[0]))
				return false;
			cmd.index_count = static_cast<uint32_t>(std::strtoul(parts[0].c_str(), nullptr, 10));
			if (parts.size() > 1 && is_number(parts[1]))
				cmd.first_index = static_cast<uint32_t>(std::strtoul(parts[1].c_str(), nullptr, 10));
			if (parts.size() > 2 && is_number(parts[2]))
				cmd.base_vertex = static_cast<int32_t>(std::strtol(parts[2].c_str(), nullptr, 10));
			if (parts.size() > 3 && is_number(parts[3]))
				cmd.instance_count = static_cast<uint32_t>(std::strtoul(parts[3].c_str(), nullptr, 10));
			return true;
		}

		// 3DMigoto blend token tables (see OverrideSetting BLendState in
		// 3DMigoto's ini parsing; tokens are upper-case in mods; compared
		// lower-cased here since only to_lower exists in the project).
		bool parse_blend_op_impl(std::string_view token, uint8_t &out)
		{
			const std::string t = to_lower(token);
			if (t == "add") { out = 0; return true; }
			if (t == "subtract") { out = 1; return true; }
			if (t == "reverse_subtract") { out = 2; return true; }
			if (t == "min") { out = 3; return true; }
			if (t == "max") { out = 4; return true; }
			return false;
		}

		bool parse_blend_factor_impl(std::string_view token, uint8_t &out)
		{
			// 0..11 line up with reshade::api::blend_factor; the neutral
			// code is the 3DMigoto ordering and the bridge remaps >=12.
			const std::string t = to_lower(token);
			static const std::pair<const char *, uint8_t> table[] = {
				{"zero", 0}, {"one", 1}, {"src_color", 2}, {"inv_src_color", 3},
				{"dest_color", 4}, {"inv_dest_color", 5}, {"src_alpha", 6},
				{"inv_src_alpha", 7}, {"dest_alpha", 8}, {"inv_dest_alpha", 9},
				{"blend_factor", 10}, {"inv_blend_factor", 11},
				{"src_alpha_sat", 12}, {"src1_color", 13}, {"inv_src1_color", 14},
				{"src1_alpha", 15}, {"inv_src1_alpha", 16},
			};
			for (const auto &[name, code] : table)
				if (t == name)
				{
					out = code;
					return true;
				}
			return false;
		}

		// 'blend = OP SRC DEST' / 'blendalpha = OP SRC DEST' -> three
		// whitespace-separated tokens.
		bool parse_blend_tokens(const std::string &value, uint8_t &op,
			uint8_t &src, uint8_t &dst)
		{
			std::istringstream ss(value);
			std::string t_op, t_src, t_dst;
			if (!(ss >> t_op >> t_src >> t_dst))
				return false;
			return parse_blend_op(t_op, op) && parse_blend_factor(t_src, src) &&
				parse_blend_factor(t_dst, dst);
		}
	}

	bool parse_blend_op(std::string_view token, uint8_t &out)
	{
		return parse_blend_op_impl(token, out);
	}

	bool parse_blend_factor(std::string_view token, uint8_t &out)
	{
		return parse_blend_factor_impl(token, out);
	}

	void parse_script_body(const std::string &section,
		const std::vector<std::pair<std::string, std::string>> &lines,
		CommandList &out, std::vector<std::string> &warnings)
	{
		out.section = section;
		if (out.name.empty())
			out.name = normalize_section_name(section);

		// if/else/endif block stack: {jump_false index, else jump index}
		std::vector<std::pair<uint32_t, uint32_t>> blocks;

		for (const auto &[raw_key, raw_value] : lines)
		{
			const std::string key = to_lower(raw_key);
			const std::string &value = raw_value;
			const std::string line = raw_key + (value.empty() ? std::string() : " = " + value);

			// ---- block structure --------------------------------------------
		// 'if <expr>' / 'else' / 'endif' arrive as key-only entries
		// (no '='), with the condition embedded in the key text.
		const auto handle_if = [&](const std::string &cond_text)
		{
			Command c;
			c.kind = Command::Kind::jump_false;
			c.source = cond_text.empty() ? line : cond_text;
			std::string err;
			c.value = expr::compile(cond_text, &err);
			if (c.value == nullptr)
			{
				warn(warnings, section, c.source, ("condition parse failed: " + err).c_str());
				// Treat as 'if 0': push a never-taken block so else/endif
				// still balance.
				c.value = std::make_unique<expr::Node>();
			}
			blocks.emplace_back(static_cast<uint32_t>(out.body.size()), UINT32_MAX);
			out.body.push_back(std::move(c));
		};
		const auto handle_else = [&]()
		{
			if (blocks.empty() || blocks.back().second != UINT32_MAX)
			{
				warn(warnings, section, "else", "'else' without matching 'if'");
				return;
			}
			Command c;
			c.kind = Command::Kind::jump;
			c.source = "else";
			blocks.back().second = static_cast<uint32_t>(out.body.size());
			out.body.push_back(std::move(c));
			out.body[blocks.back().first].jump_target =
				static_cast<uint32_t>(out.body.size());
		};
		const auto handle_endif = [&]()
		{
			if (blocks.empty())
			{
				warn(warnings, section, "endif", "'endif' without matching 'if'");
				return;
			}
			const uint32_t after = static_cast<uint32_t>(out.body.size());
			if (blocks.back().second != UINT32_MAX)
			{
				// 'if' already targets past-'else'; only the else-jump
				// needs patching to past-'endif'.
				out.body[blocks.back().second].jump_target = after;
			}
			else
			{
				out.body[blocks.back().first].jump_target = after;
			}
			blocks.pop_back();
		};

		// ---- block structure --------------------------------------------
		// Block commands are recognised on the RECONSTRUCTED logical
		// line: 'if ResourceX === null' is split by the ini parser at
		// the first '=' (key 'if ResourceX', value '== null'), so
		// key/value inspection alone would miss them.
		const std::string logical = value.empty()
			? raw_key
			: raw_key + "=" + raw_value;
		const std::string logical_l = to_lower(logical);
		if (logical_l == "else" && value.empty())
		{
			handle_else();
			continue;
		}
		if (logical_l == "endif" && value.empty())
		{
			handle_endif();
			continue;
		}
		if (logical_l == "if" ||
			(logical_l.size() > 3 && logical_l.rfind("if ", 0) == 0))
		{
			// 'if <cond>'; conditions starting with '=' come from
			// 'if = x' lines, which are not block commands.
			std::string_view cond = logical;
			cond.remove_prefix(2);
			const size_t b = cond.find_first_not_of(" \t");
			const size_t e = cond.find_last_not_of(" \t");
			const std::string cond_text = b == std::string_view::npos
				? std::string()
				: std::string(cond.substr(b, e - b + 1));
			if (cond_text.empty() || cond_text[0] != '=')
			{
				handle_if(cond_text);
				continue;
			}
		}

		if (key == "run")
		{
			Command c;
			c.kind = Command::Kind::run;
			c.target = to_lower(value);
			c.source = line;
			if (is_namespaced(c.target))
				warn(warnings, section, line, "framework command list (namespaced) is not supported; no-op");
			out.body.push_back(std::move(c));
		}
		else if (key == "blend" || key == "blendalpha")
		{
			// 'blend[alpha] = OP SRC DEST' (may append '[N]' for per-RT
			// blend; WWMI mods use the global form).
			uint8_t op = 0, src = 0, dst = 0;
			if (!parse_blend_tokens(value, op, src, dst))
			{
				warn(warnings, section, line, "cannot parse blend op/factors");
			}
			else
			{
				Command c;
				c.kind = Command::Kind::blend_state;
				c.source = line;
				c.blend.enable = true;
				if (key == "blend")
				{
					c.blend.has_color = true;
					c.blend.op = op; c.blend.src = src; c.blend.dst = dst;
				}
				else
				{
					c.blend.has_alpha = true;
					c.blend.alpha_op = op; c.blend.alpha_src = src; c.blend.alpha_dst = dst;
				}
				out.body.push_back(std::move(c));
			}
		}
		else if (key.rfind("blend_factor[", 0) == 0)
		{
			// 'blend_factor[N] = v' partial OMSetBlendFactor update
			const size_t close = key.find(']');
			const size_t idx = key.rfind('[', close);
			char *stop = nullptr;
			const long n = (close != std::string::npos && idx != std::string::npos && close > idx + 1)
				? std::strtol(key.c_str() + idx + 1, &stop, 10)
				: -1;
			char *vstop = nullptr;
			const double v = std::strtod(value.c_str(), &vstop);
			if (n < 0 || n > 3 || vstop == nullptr || *vstop != '\0' || value.empty())
			{
				warn(warnings, section, line, "cannot parse blend_factor");
			}
			else
			{
				Command c;
				c.kind = Command::Kind::blend_state;
				c.source = line;
				c.blend.enable = true;
				c.blend.has_factors = true;
				c.blend.factor_mask = static_cast<uint8_t>(1u << n);
				c.blend.factors[n] = static_cast<float>(v);
				out.body.push_back(std::move(c));
			}
		}
		else if (key == "drawindexed" || key == "draw")
			{
				Command c;
				c.kind = Command::Kind::drawindexed;
				c.source = line;
				if (!parse_drawindexed(value, c))
				{
					warn(warnings, section, line, "cannot parse drawindexed parameters");
					continue;
				}
				out.body.push_back(std::move(c));
			}
			else if (key == "handling")
			{
				if (to_lower(value) == "skip" || to_lower(value) == "abort")
				{
					Command c;
					c.kind = Command::Kind::handling_skip;
					c.source = line;
					out.body.push_back(std::move(c));
				}
				else
					warn(warnings, section, line, "unsupported handling mode");
			}
			else if (key == "checktextureoverride")
			{
				Command c;
				c.kind = Command::Kind::check_override;
				c.source = line;
				const std::string slot = to_lower(value);
				char st = 0, ty = 0;
				uint32_t idx = 0;
				if (parse_slot_key(slot, st, ty, idx) && (ty == 't' || ty == 'u' || ty == 'c'))
				{
					c.stage = st;
					c.slot_type = ty == 'c' ? 'c' : ty;
					c.slot = idx;
				}
				else
					warn(warnings, section, line, "CheckTextureOverride target not understood");
				out.body.push_back(std::move(c));
			}
			else if (!key.empty() && key[0] == '$')
			{
				const bool post = false; // 'post' arrives as key 'post $x'
				(void)post;
				Command c;
				c.kind = Command::Kind::assign;
				c.target = expr::normalize_var(key);
				c.source = line;
				std::string err;
				c.value = expr::compile(value, &err);
				if (c.value == nullptr)
				{
					warn(warnings, section, line, ("expression parse failed: " + err).c_str());
					continue;
				}
				out.body.push_back(std::move(c));
			}
			else if (key.size() > 5 && key.rfind("post ", 0) == 0 && key[5] == '$')
			{
				Command c;
				c.kind = Command::Kind::post_assign;
				c.target = expr::normalize_var(key.substr(5));
				c.source = line;
				std::string err;
				c.value = expr::compile(value, &err);
				if (c.value == nullptr)
				{
					warn(warnings, section, line, ("expression parse failed: " + err).c_str());
					continue;
				}
				out.body.push_back(std::move(c));
			}
			else
			{
				// Slot bindings (vb0/ib/cs-t34/...) vs resource ops
				// (ResourceX = ...) vs known-unsupported keys.
				char stage = 0, slot_type = 0;
				uint32_t slot = 0;
				if (parse_slot_key(key, stage, slot_type, slot))
				{
					if (slot_type == 'v' || slot_type == 'i') // vbN / ib
					{
						bool ref = false;
						const std::string src = strip_bind_option(value, ref);
						Command c;
						c.kind = slot_type == 'i' ? Command::Kind::bind_ib : Command::Kind::bind_vb;
						c.slot = slot_type == 'i' ? 0 : slot;
						c.target = to_lower(src);
						c.is_ref = ref;
						c.source = line;
						if (c.target.empty())
							warn(warnings, section, line, "empty binding source");
						out.body.push_back(std::move(c));
					}
					else // cs-tN / cs-uN / vs-cbN / ps-tN ...
					{
						bool ref = false;
						const std::string src = strip_bind_option(value, ref);
						Command c;
						c.kind = Command::Kind::bind_slot;
						c.stage = stage;
						c.slot_type = slot_type;
						c.slot = slot;
						c.target = to_lower(src);
						c.is_ref = ref;
						c.source = line;
						if (is_namespaced(c.target))
							warn(warnings, section, line, "framework resource binding is not supported; no-op");
						out.body.push_back(std::move(c));
					}
					continue;
				}

				const std::string target = to_lower(key);
				const std::string v = value;
				if (target.rfind("resource", 0) == 0 || !v.empty())
				{
					// 'ResourceX = <op> <source>' forms
					const std::vector<std::string> words = [&]() {
						std::vector<std::string> w;
						size_t pos = 0;
						while (pos <= v.size())
						{
							const size_t sp = v.find(' ', pos);
							w.push_back(v.substr(pos, sp == std::string::npos ? std::string::npos : sp - pos));
							if (sp == std::string::npos) break;
							pos = sp + 1;
						}
						return w;
					}();

					Command c;
					c.dest = target;
					c.source = line;
					bool handled = true;

					if (!words.empty() && to_lower(words[0]) == "null")
					{
						c.kind = Command::Kind::res_null;
					}
					else if (!words.empty() && to_lower(words[0]) == "copy" && words.size() >= 2)
					{
						c.kind = Command::Kind::res_copy;
						c.target = to_lower(words[1]);
					}
					else if (!words.empty() && to_lower(words[0]) == "copy_desc" && words.size() >= 2)
					{
						c.kind = Command::Kind::res_copy_desc;
						c.target = to_lower(words[1]);
					}
					else if (!words.empty() && to_lower(words[0]) == "ref" && words.size() >= 2)
					{
						const std::string src = to_lower(words[1]);
						if (is_game_ia_slot(src))
						{
							c.kind = Command::Kind::res_capture;
							c.target = src; // 'vb0' / 'ib' / ...
						}
						else
						{
							c.kind = Command::Kind::res_ref;
							c.target = src;
						}
					}
					else
					{
						handled = false;
					}

					if (handled)
					{
						if (is_namespaced(c.dest) || is_namespaced(c.target))
							warn(warnings, section, line, "framework resource (namespaced) is not supported; no-op");
						out.body.push_back(std::move(c));
						continue;
					}
				}

				// Known-unsupported TextureOverride keys and CustomShader
				// state blocks degrade to warnings.
				if (key == "blend" || key.rfind("blend_factor", 0) == 0 ||
					key == "filter_index" || key == "match_priority" ||
					key == "override_byte_stride" || key == "override_vertex_count" ||
					key == "match_first_index" || key == "match_index_count" ||
					key == "match_first_vertex" || key == "match_vertex_count" ||
					key == "match_instance_count" || key == "hash" ||
					key == "type" || key == "def" || key == "as")
				{
					// Structural keys handled elsewhere; silent.
					continue;
				}
				warn(warnings, section, line, "unsupported script line");
			}
		}

		for (const auto &blk : blocks)
		{
			// Unterminated 'if': point it at the end of the body.
			out.body[blk.first].jump_target = static_cast<uint32_t>(out.body.size());
			warnings.push_back(section + ": 'if' block without 'endif'");
		}
	}

	uint32_t parse_vk(const std::string &key_name)
	{
		const std::string k = to_lower(key_name);
		static const std::unordered_map<std::string, uint32_t> named = {
			{"vk_back", 0x08}, {"vk_tab", 0x09}, {"vk_return", 0x0D}, {"vk_shift", 0x10},
			{"vk_control", 0x11}, {"vk_menu", 0x12}, {"vk_pause", 0x13}, {"vk_capital", 0x14},
			{"vk_escape", 0x1B}, {"vk_space", 0x20}, {"vk_prior", 0x21}, {"vk_next", 0x22},
			{"vk_end", 0x23}, {"vk_home", 0x24}, {"vk_left", 0x25}, {"vk_up", 0x26},
			{"vk_right", 0x27}, {"vk_down", 0x28}, {"vk_insert", 0x2D}, {"vk_delete", 0x2E},
			{"vk_0", '0'}, {"vk_1", '1'}, {"vk_2", '2'}, {"vk_3", '3'}, {"vk_4", '4'},
			{"vk_5", '5'}, {"vk_6", '6'}, {"vk_7", '7'}, {"vk_8", '8'}, {"vk_9", '9'},
			{"vk_a", 'A'}, {"vk_b", 'B'}, {"vk_c", 'C'}, {"vk_d", 'D'}, {"vk_e", 'E'},
			{"vk_f", 'F'}, {"vk_g", 'G'}, {"vk_h", 'H'}, {"vk_i", 'I'}, {"vk_j", 'J'},
			{"vk_k", 'K'}, {"vk_l", 'L'}, {"vk_m", 'M'}, {"vk_n", 'N'}, {"vk_o", 'O'},
			{"vk_p", 'P'}, {"vk_q", 'Q'}, {"vk_r", 'R'}, {"vk_s", 'S'}, {"vk_t", 'T'},
			{"vk_u", 'U'}, {"vk_v", 'V'}, {"vk_w", 'W'}, {"vk_x", 'X'}, {"vk_y", 'Y'},
			{"vk_z", 'Z'},
			{"vk_numpad0", 0x60}, {"vk_numpad1", 0x61}, {"vk_numpad2", 0x62},
			{"vk_numpad3", 0x63}, {"vk_numpad4", 0x64}, {"vk_numpad5", 0x65},
			{"vk_numpad6", 0x66}, {"vk_numpad7", 0x67}, {"vk_numpad8", 0x68},
			{"vk_numpad9", 0x69},
			{"vk_f1", 0x70}, {"vk_f2", 0x71}, {"vk_f3", 0x72}, {"vk_f4", 0x73},
			{"vk_f5", 0x74}, {"vk_f6", 0x75}, {"vk_f7", 0x76}, {"vk_f8", 0x77},
			{"vk_f9", 0x78}, {"vk_f10", 0x79}, {"vk_f11", 0x7A}, {"vk_f12", 0x7B},
			{"vk_scroll", 0x91}, {"vk_numlock", 0x90},
		};

		const auto it = named.find(k);
		if (it != named.end())
			return it->second;

		// Single printable character
		if (key_name.size() == 1)
		{
			const unsigned char ch = static_cast<unsigned char>(key_name[0]);
			switch (ch)
			{
			case ',': return 0xBC; // VK_OEM_COMMA
			case '.': return 0xBE; // VK_OEM_PERIOD
			case '/': return 0xBF; // VK_OEM_2
			case ';': return 0xBA; // VK_OEM_1
			case '\'': return 0xDE; // VK_OEM_7
			case '[': return 0xDB; // VK_OEM_4
			case ']': return 0xDD; // VK_OEM_6
			case '-': return 0xBD; // VK_OEM_MINUS
			case '=': return 0xBB; // VK_OEM_PLUS
			case '`': return 0xC0; // VK_OEM_3
			case '\\': return 0xDC; // VK_OEM_5
			default: return static_cast<uint32_t>(std::toupper(ch));
			}
		}

		// 'ctrl+x' / 'shift+x' combos: base key only (M3 simplification)
		const size_t plus = key_name.find('+');
		if (plus != std::string::npos && plus + 1 < key_name.size())
			return parse_vk(key_name.substr(plus + 1));

		return 0;
	}

	bool parse_key_section(const std::string &section,
		const std::vector<std::pair<std::string, std::string>> &lines,
		KeyBinding &out, std::vector<std::string> &warnings)
	{
		out.name = section;

		std::string key_name;
		std::string type;
		std::string condition_text;
		std::string var_name;   // '$leg'
		std::string var_values; // '0,1,2'

		for (const auto &[raw_key, raw_value] : lines)
		{
			const std::string key = to_lower(raw_key);
			if (key == "key")
				key_name = raw_value;
			else if (key == "type")
				type = to_lower(raw_value);
			else if (key == "condition")
				condition_text = raw_value;
			else if (!key.empty() && key[0] == '$')
			{
				var_name = key;
				var_values = raw_value;
			}
			// 'back'/'forward' etc. ignored
		}

		out.vk = parse_vk(key_name);
		out.key_name = key_name;
		if (out.vk == 0)
		{
			warnings.push_back(section + ": unknown key '" + key_name + "'");
			return false;
		}
		if (type != "cycle" && !type.empty())
		{
			warnings.push_back(section + ": key type '" + type + "' unsupported (only cycle)");
			return false;
		}

		// '$var = 0,1,2' -- ini split gives key '$var', value '0,1,2'
		if (var_name.empty() || var_values.empty())
		{
			warnings.push_back(section + ": no '$var = a,b,c' cycle line");
			return false;
		}
		out.var = expr::normalize_var(var_name);
		for (const std::string &part : split_csv(var_values))
		{
			if (!is_number(part))
			{
				warnings.push_back(section + ": non-numeric cycle value '" + part + "'");
				continue;
			}
			out.values.push_back(static_cast<float>(std::strtod(part.c_str(), nullptr)));
		}
		if (out.values.empty())
		{
			warnings.push_back(section + ": empty cycle list");
			return false;
		}

		if (!condition_text.empty())
		{
			std::string err;
			out.condition = expr::compile(condition_text, &err);
			if (out.condition == nullptr)
			{
				warnings.push_back(section + ": condition parse failed: " + err);
				out.condition = nullptr;
			}
		}
		return true;
	}
}

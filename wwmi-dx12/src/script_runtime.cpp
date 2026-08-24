#include "script_runtime.hpp"
#include "ini_file.hpp"
#include "buffer_tracker.hpp" // IaState

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>

namespace wwmi
{
	namespace
	{
		// Mods write both 'R32_UINT' and 'DXGI_FORMAT_R32_UINT'.
		std::string normalize_format(std::string fmt)
		{
			fmt = to_lower(fmt);
			if (fmt.rfind("dxgi_format_", 0) == 0)
				fmt = fmt.substr(12);
			return fmt;
		}

		float parse_number(const std::string &s, bool *ok = nullptr)
		{
			char *stop = nullptr;
			const float v = static_cast<float>(std::strtod(s.c_str(), &stop));
			const bool valid = stop != nullptr && *stop == '\0' && !s.empty();
			if (ok != nullptr)
				*ok = valid;
			return valid ? v : 0.0f;
		}

		// Bytes per element of the DXGI formats WWMI mods declare.
		uint32_t format_element_size(const std::string &fmt_raw)
		{
			const std::string fmt = normalize_format(fmt_raw);
			if (fmt == "r32g32b32a32_uint" || fmt == "r32g32b32a32_float" ||
				fmt == "r32g32b32a32_sint" || fmt == "r32g32b32a32_unorm")
				return 16;
			if (fmt == "r32g32b32_uint" || fmt == "r32g32b32_float" ||
				fmt == "r32g32b32_sint")
				return 12;
			if (fmt == "r32g32_uint" || fmt == "r32g32_float" ||
				fmt == "r16g16b16a16_uint" || fmt == "r16g16b16a16_float" ||
				fmt == "r16g16b16a16_snorm" || fmt == "r16g16b16a16_unorm" ||
				fmt == "r32g32_sint")
				return 8;
			if (fmt == "r32_uint" || fmt == "r32_float" || fmt == "r32_sint" ||
				fmt == "r16g16_uint" || fmt == "r16g16_float" || fmt == "r16g16_snorm" ||
				fmt == "r16g16_unorm" || fmt == "r8g8b8a8_uint" || fmt == "r8g8b8a8_snorm" ||
				fmt == "r8g8b8a8_unorm" || fmt == "r8g8b8a8_sint" ||
				fmt == "r10g10b10a2_unorm" || fmt == "r11g11b10_float")
				return 4;
			if (fmt == "r16_uint" || fmt == "r16_float" || fmt == "r16_snorm" ||
				fmt == "r16_unorm" || fmt == "r8g8_uint" || fmt == "r8g8_snorm" ||
				fmt == "r8g8_unorm")
				return 2;
			if (fmt == "r8_uint" || fmt == "r8_snorm" || fmt == "r8_unorm")
				return 1;
			return 0;
		}

		uint32_t format_index_size(const std::string &fmt_raw)
		{
			const std::string fmt = normalize_format(fmt_raw);
			if (fmt == "r32_uint" || fmt == "r32_sint")
				return 4;
			if (fmt == "r16_uint" || fmt == "r16_sint")
				return 2;
			return 0;
		}
	}

	// ---------------------------------------------------------------------
	// Loading
	// ---------------------------------------------------------------------

	bool ScriptRuntime::load(const std::filesystem::path &ini_path)
	{
		_mod_dir = ini_path.parent_path();
		_mod_name = _mod_dir.filename().string();
		_warnings.clear();
		_warned.clear();

		std::ifstream file(ini_path, std::ios::binary);
		if (!file.is_open())
			return false;
		const std::string content((std::istreambuf_iterator<char>(file)),
			std::istreambuf_iterator<char>());

		IniFile ini;
		parse_ini(content, ini);

		for (const IniSection &sec : ini.sections)
		{
			const std::string lower = to_lower(sec.name);

			// entry list in script form ('if x' entries have empty values)
			std::vector<std::pair<std::string, std::string>> lines;
			lines.reserve(sec.entries.size());
			for (const IniEntry &e : sec.entries)
				lines.emplace_back(e.key, e.value);

			if (lower == "constants")
			{
				for (const auto &[raw_key, raw_value] : lines)
				{
					// 'global [persist] $name' (any prefix order/combination)
					std::string name = raw_key;
					bool persist = false;
					for (;;)
					{
						if (istarts_with(name, "global "))
							name = name.substr(7);
						else if (istarts_with(name, "persist "))
						{
							persist = true;
							name = name.substr(8);
						}
						else
							break;
					}
					if (name.empty() || name[0] != '$')
					{
						_warnings.push_back("[Constants] unrecognized entry: " + raw_key);
						continue;
					}
					const std::string var = expr::normalize_var(name);
					if (var.empty())
						continue;

					// Values may reference earlier constants (3DMigoto
					// evaluates expressions at load).
					std::string err;
					expr::NodePtr node = expr::compile(raw_value, &err);
					if (node == nullptr)
					{
						_warnings.push_back("[Constants] " + raw_key + " = " + raw_value +
							": parse failed (" + err + ")");
						continue;
					}
					const expr::EvalContext ctx{
						[this](const std::string &n, float &v) { return get_var(n, v); },
						[this](const std::string &n) { return resource_alive(n); }
					};
					float value = 0.0f;
					expr::eval(*node, ctx, value);
					_vars[var] = value;
					if (persist)
						_persist_vars.insert(var);
				}
			}
			else if (istarts_with(lower, "key"))
			{
				KeyBinding key;
				std::vector<std::string> warns;
				if (parse_key_section(sec.name, lines, key, warns))
					_keys.push_back(std::move(key));
				for (const std::string &w : warns)
					_warnings.push_back(w);
			}
			else if (lower == "present")
			{
				std::vector<std::string> warns;
				parse_script_body(sec.name, lines, _present, warns);
				for (const std::string &w : warns)
					_warnings.push_back(w);
			}
			else if (istarts_with(lower, "commandlist") || istarts_with(lower, "customshader"))
			{
				CommandList list;
				list.name = normalize_section_name(sec.name);
				std::vector<std::string> warns;
				parse_script_body(sec.name, lines, list, warns);
				for (const std::string &w : warns)
					_warnings.push_back(w);
				_lists[list.name] = std::move(list);
			}
			else if (istarts_with(lower, "shaderoverride"))
			{
				// Pipeline-hash matching is M4; the bodies parse (and can
				// be run) but never trigger on their own yet.
				CommandList list;
				list.name = normalize_section_name(sec.name);
				std::vector<std::string> warns;
				parse_script_body(sec.name, lines, list, warns);
				for (const std::string &w : warns)
					_warnings.push_back(w);
				_lists[list.name] = std::move(list);
			}
			else if (istarts_with(lower, "textureoverride"))
			{
				CommandList list;
				list.name = normalize_section_name(sec.name);
				std::vector<std::string> warns;
				parse_script_body(sec.name, lines, list, warns);
				for (const std::string &w : warns)
					_warnings.push_back(w);
				_override_scripts[list.name] = std::move(list);
			}
			else if (istarts_with(lower, "resource"))
			{
				ScriptResource res;
				res.section = sec.name;
				res.name = to_lower(sec.name.substr(8)); // strip 'resource'
				for (const auto &[raw_key, raw_value] : lines)
				{
					const std::string key = to_lower(raw_key);
					if (key == "filename")
						res.filename = raw_value;
					else if (key == "type")
						res.type = to_lower(raw_value);
					else if (key == "format")
						res.format = to_lower(raw_value);
					else if (key == "stride")
						res.stride = static_cast<uint32_t>(std::strtoul(raw_value.c_str(), nullptr, 10));
					else if (key == "array")
						res.array = static_cast<uint32_t>(std::strtoul(raw_value.c_str(), nullptr, 10));
					// 'data = "..."' in-mod buffers are not supported yet
				}
				res.index_size = format_index_size(res.format);
				if (res.stride == 0)
					res.stride = format_element_size(res.format);
				_resources[res.name] = std::move(res);
			}
			// everything else ([ShaderOverride] is above; unknown sections
			// are the mod_rules parser's business)
		}

		return true;
	}

	// ---------------------------------------------------------------------
	// Variables
	// ---------------------------------------------------------------------

	bool ScriptRuntime::get_var(const std::string &name, float &out) const
	{
		const auto it = _vars.find(name);
		if (it == _vars.end())
			return false;
		out = it->second;
		return true;
	}

	float ScriptRuntime::var_or(const std::string &name, float fallback) const
	{
		float v = 0.0f;
		return get_var(name, v) ? v : fallback;
	}

	void ScriptRuntime::set_var(const std::string &name, float value)
	{
		_vars[name] = value;
	}

	std::vector<std::pair<std::string, float>> ScriptRuntime::vars_snapshot() const
	{
		std::vector<std::pair<std::string, float>> out;
		out.reserve(_vars.size());
		for (const auto &[name, value] : _vars)
			out.emplace_back(name, value);
		std::sort(out.begin(), out.end(),
			[](const auto &a, const auto &b) { return a.first < b.first; });
		return out;
	}

	void ScriptRuntime::load_persist(const std::filesystem::path &file)
	{
		std::ifstream in(file);
		if (!in.is_open())
			return;
		std::string line;
		while (std::getline(in, line))
		{
			const size_t eq = line.find('=');
			if (eq == std::string::npos)
				continue;
			const std::string name = to_lower(line.substr(0, eq));
			if (_persist_vars.count(name) == 0)
				continue;
			bool ok = false;
			const float v = parse_number(line.substr(eq + 1), &ok);
			if (ok)
				_vars[name] = v;
		}
	}

	void ScriptRuntime::save_persist(const std::filesystem::path &file) const
	{
		std::ofstream out(file, std::ios::trunc);
		if (!out.is_open())
			return;
		char buf[64];
		for (const std::string &name : _persist_vars)
		{
			const auto it = _vars.find(name);
			if (it == _vars.end())
				continue;
			std::snprintf(buf, sizeof(buf), "%.9g", it->second);
			out << name << '=' << buf << '\n';
		}
	}

	// ---------------------------------------------------------------------
	// Keys
	// ---------------------------------------------------------------------

	void ScriptRuntime::on_key(uint32_t vk, bool down)
	{
		for (KeyBinding &key : _keys)
		{
			if (key.vk != vk)
				continue;

			// Edge detection per binding (caller passes the CURRENT state;
			// the previous state lives here).
			const bool was = _key_prev[vk];
			_key_prev[vk] = down;
			if (!down || was)
				continue;

			if (key.condition != nullptr)
			{
				const expr::EvalContext ctx{
					[this](const std::string &n, float &v) { return get_var(n, v); },
					[this](const std::string &n) { return resource_alive(n); }
				};
				float c = 0.0f;
				expr::eval(*key.condition, ctx, c);
				if (c == 0.0f)
					continue;
			}

			// cycle: find the current value, advance (wrap), missing -> 0th
			float cur = 0.0f;
			get_var(key.var, cur);
			size_t idx = key.values.size();
			for (size_t i = 0; i < key.values.size(); ++i)
				if (key.values[i] == cur)
				{
					idx = i;
					break;
				}
			const float next = (idx + 1 >= key.values.size())
				? key.values.front()
				: key.values[idx + 1];
			set_var(key.var, next);
		}
	}

	// ---------------------------------------------------------------------
	// Resources
	// ---------------------------------------------------------------------

	ScriptResource *ScriptRuntime::find_resource(const std::string &name)
	{
		const std::string lower = to_lower(name);
		auto it = _resources.find(lower);
		if (it == _resources.end() && lower.rfind("resource", 0) == 0)
		{
			// References keep the section prefix ('ResourceIndexBuffer'),
			// stored keys are stripped ('indexbuffer').
			it = _resources.find(lower.substr(8));
		}
		return it == _resources.end() ? nullptr : &it->second;
	}

	const ScriptResource *ScriptRuntime::find_resource(const std::string &name) const
	{
		const std::string lower = to_lower(name);
		auto it = _resources.find(lower);
		if (it == _resources.end() && lower.rfind("resource", 0) == 0)
			it = _resources.find(lower.substr(8));
		return it == _resources.end() ? nullptr : &it->second;
	}

	ScriptResource *ScriptRuntime::find_or_create_resource(const std::string &name)
	{
		ScriptResource *res = find_resource(name);
		if (res != nullptr)
			return res;
		std::string key = to_lower(name);
		if (key.rfind("resource", 0) == 0)
			key = key.substr(8);
		if (key.empty())
			return nullptr;
		ScriptResource implicit;
		implicit.name = key;
		_resources[key] = std::move(implicit);
		return &_resources[key];
	}

	bool ScriptRuntime::resource_alive(const std::string &name) const
	{
		const ScriptResource *res = find_resource(name);
		return res != nullptr && res->handle != 0;
	}

	uint64_t ScriptRuntime::ensure_resource(ScriptResource &res, GpuBridge *bridge)
	{
		if (res.handle != 0)
			return res.handle;
		if (res.create_failed || bridge == nullptr)
			return 0;
		if (res.game_capture)
			return 0; // captures resolve at capture time only

		if (!res.filename.empty())
		{
			const std::filesystem::path path = _mod_dir / res.filename;
			const bool is_dds = res.filename.size() > 4 &&
				to_lower(res.filename.substr(res.filename.size() - 4)) == ".dds";
			if (is_dds)
			{
				// Texture resources bind via ps-tN redirection (M4) or the
				// M1 hash-replacement path; nothing to instantiate here.
				res.create_failed = true;
				warn_once("restex:" + res.name,
					"resource '" + res.name + "' is a texture; slot-bound textures need M4 descriptor rebinding");
				return 0;
			}

			std::ifstream file(path, std::ios::binary);
			if (!file.is_open())
			{
				res.create_failed = true;
				warn_once("resopen:" + res.name, "cannot open " + res.filename);
				return 0;
			}
			std::vector<char> bytes((std::istreambuf_iterator<char>(file)),
				std::istreambuf_iterator<char>());

			const bool uav = res.type == "rwbuffer" || res.type == "rwtexture" ||
				res.type == "rwstructured_buffer" || res.type == "buffer_uv";
			res.handle = bridge->create_buffer(bytes.data(), bytes.size(), uav, true);
			res.size = bytes.size();
			if (res.handle == 0)
			{
				res.create_failed = true;
				warn_once("rescreate:" + res.name, "buffer creation failed for " + res.filename);
				return 0;
			}
			return res.handle;
		}

		// Synthesized buffer: array elements of 'format' (zero-filled).
		const uint32_t elem = res.stride != 0 ? res.stride : format_element_size(res.format);
		if (res.array == 0 || elem == 0)
		{
			res.create_failed = true;
			return 0; // pure reference resources stay dead until ref'd
		}
		res.size = static_cast<uint64_t>(res.array) * elem;
		res.handle = bridge->create_buffer(nullptr, res.size, true, true);
		if (res.handle == 0)
		{
			res.create_failed = true;
			return 0;
		}
		return res.handle;
	}

	bool ScriptRuntime::capture_game_slot(ScriptResource &res, const std::string &slot,
		const IaState *ia, GpuBridge *bridge)
	{
		if (ia == nullptr)
		{
			warn_once("cap:" + res.name, "game-slot capture outside a draw context");
			return false;
		}

		uint64_t handle = 0, offset = 0;
		uint32_t stride = 0;
		if (slot == "ib")
		{
			handle = ia->ib_buffer;
			offset = ia->ib_offset;
			stride = ia->ib_index_size;
		}
		else if (slot.size() > 2 && slot[0] == 'v' && slot[1] == 'b')
		{
			const uint32_t n = static_cast<uint32_t>(std::strtoul(slot.c_str() + 2, nullptr, 10));
			if (n < IaState::max_vb_slots && (ia->vb_valid_mask & (1u << n)))
			{
				handle = ia->vbs[n].buffer;
				offset = ia->vbs[n].offset;
				stride = ia->vbs[n].stride;
			}
		}
		if (handle == 0)
		{
			// Frame-start captures before any binding: stay dead, retry later.
			return false;
		}

		res.handle = handle;
		res.offset = offset;
		res.stride = stride;
		res.game_capture = true;
		res.size = 0; // resolved lazily via the bridge when needed
		if (bridge != nullptr)
			bridge->buffer_size(handle, &res.size);
		return true;
	}

	void ScriptRuntime::destroy_resources(GpuBridge *bridge)
	{
		if (bridge == nullptr)
			return;
		for (auto &[name, res] : _resources)
		{
			if (res.handle != 0 && !res.game_capture)
				bridge->destroy_buffer(res.handle);
			res.handle = 0;
			res.create_failed = false;
		}
	}

	const CommandList *ScriptRuntime::find_override_script(const std::string &section) const
	{
		const auto it = _override_scripts.find(normalize_section_name(section));
		return it == _override_scripts.end() ? nullptr : &it->second;
	}

	// ---------------------------------------------------------------------
	// Execution
	// ---------------------------------------------------------------------

	void ScriptRuntime::warn_once(const std::string &key, const std::string &msg)
	{
		if (_warned.insert(key).second)
		{
			_warnings.push_back(msg);
			++commands_warned;
		}
	}

	void ScriptRuntime::run_present(GpuBridge *bridge)
	{
		if (_present.empty())
			return;
		execute(_present, bridge, nullptr);
	}

	bool ScriptRuntime::run_list(const std::string &name, GpuBridge *bridge, const IaState *ia)
	{
		auto it = _lists.find(name);
		if (it == _lists.end())
		{
			// 'run = CommandListX' keeps the section prefix; stored
			// keys are normalized ('x'). Try the normalized form.
			const std::string norm = normalize_section_name(name);
			if (norm == name)
				return false;
			it = _lists.find(norm);
			if (it == _lists.end())
				return false;
		}
		execute(it->second, bridge, ia);
		return true;
	}

	bool ScriptRuntime::run_override(const std::string &section, GpuBridge *bridge, const IaState *ia)
	{
		const CommandList *list = find_override_script(section);
		if (list == nullptr)
			return false;
		_skip = false;
		execute(*list, bridge, ia);
		return _skip;
	}

	void ScriptRuntime::execute(const CommandList &list, GpuBridge *bridge, const IaState *ia)
	{
		std::vector<const Command *> post;

		// M4: blend state lives only inside one top-level execution.
		_blend_active = false;
		_active_blend = BlendConfig{};

		exec_body(list, bridge, ia, 0, post);

		const expr::EvalContext ctx{
			[this](const std::string &n, float &v) { return get_var(n, v); },
			[this](const std::string &n) { return resource_alive(n); }
		};

		// 'post $x = v' phase (after the whole list, in order).
		for (const Command *c : post)
		{
			float v = 0.0f;
			expr::eval(*c->value, ctx, v);
			set_var(c->target, v);
		}

		_blend_active = false;
		_active_blend = BlendConfig{};
	}

	void ScriptRuntime::exec_body(const CommandList &list, GpuBridge *bridge,
		const IaState *ia, int depth, std::vector<const Command *> &post)
	{
		if (depth > 16)
		{
			warn_once("depth:" + list.name, "run-chain too deep at '" + list.name + "'");
			return;
		}
		++lists_executed;

		const expr::EvalContext ctx{
			[this](const std::string &n, float &v) { return get_var(n, v); },
			[this](const std::string &n) { return resource_alive(n); }
		};

		size_t pc = 0;
		while (pc < list.body.size())
		{
			const Command &c = list.body[pc];
			++pc;

			switch (c.kind)
			{
			case Command::Kind::jump:
				pc = c.jump_target;
				break;

			case Command::Kind::jump_false:
			{
				float v = 0.0f;
				expr::eval(*c.value, ctx, v);
				if (v == 0.0f)
					pc = c.jump_target;
				break;
			}

			case Command::Kind::assign:
			{
				if (is_namespaced(c.target))
					break; // framework variable: parse-time warning already
				float v = 0.0f;
				expr::eval(*c.value, ctx, v);
				set_var(c.target, v);
				break;
			}

			case Command::Kind::post_assign:
				post.push_back(&c);
				break;

			case Command::Kind::run:
				if (is_namespaced(c.target))
					break; // framework list: no-op (warned at parse)
				if (!run_list(c.target, bridge, ia))
					warn_once("run:" + c.target, "run target '" + c.target + "' not found");
				break;

			case Command::Kind::blend_state:
				// Merge this line's fields into the active override
				// (each 'blend*' line updates only what it carries).
				_blend_active = true;
				_active_blend.enable = true;
				if (c.blend.has_factors)
			{
				for (int n = 0; n < 4; ++n)
					if (c.blend.factor_mask & (1u << n))
					{
						_active_blend.factors[n] = c.blend.factors[n];
						_active_blend.factor_mask |= static_cast<uint8_t>(1u << n);
					}
				_active_blend.has_factors = true;
			}
				if (c.blend.has_color)
				{
					_active_blend.has_color = true;
					_active_blend.op = c.blend.op;
					_active_blend.src = c.blend.src;
					_active_blend.dst = c.blend.dst;
				}
				if (c.blend.has_alpha)
				{
					_active_blend.has_alpha = true;
					_active_blend.alpha_op = c.blend.alpha_op;
					_active_blend.alpha_src = c.blend.alpha_src;
					_active_blend.alpha_dst = c.blend.alpha_dst;
				}
				break;

			case Command::Kind::drawindexed:
				if (bridge == nullptr)
				{
					warn_once("draw:" + list.name, "drawindexed outside a draw context");
					break;
				}
				if (_blend_active && _active_blend.enable)
				{
					// Per-draw apply/revoke: the next script draw (if any)
					// re-applies, and nothing leaks past this execution.
					const uint64_t token = bridge->apply_blend(_active_blend);
					bridge->draw_indexed(c.index_count, c.first_index, c.base_vertex,
						c.instance_count);
					if (token != 0)
						bridge->revoke_blend(token);
				}
				else
				{
					bridge->draw_indexed(c.index_count, c.first_index, c.base_vertex,
						c.instance_count);
				}
				++draws_issued;
				break;

			case Command::Kind::handling_skip:
				// Requested by the body: the caller (draw interceptor)
				// blocks the original draw. Inside a plain run chain
				// (CustomShader bodies) it is informational only.
				_skip = true;
				break;

			case Command::Kind::bind_vb:
			{
				if (bridge == nullptr)
					break;
				ScriptResource *res = find_resource(c.target);
				const uint64_t h = res != nullptr ? ensure_resource(*res, bridge) : 0;
				if (h == 0)
				{
					warn_once("vb:" + c.target, "vb slot " + std::to_string(c.slot) +
						": resource '" + c.target + "' unavailable");
					break;
				}
				bridge->bind_vb(c.slot, h, res->offset, res->stride);
				break;
			}

			case Command::Kind::bind_ib:
			{
				if (bridge == nullptr)
					break;
				ScriptResource *res = find_resource(c.target);
				const uint64_t h = res != nullptr ? ensure_resource(*res, bridge) : 0;
				if (h == 0)
				{
					warn_once("ib:" + c.target, "ib: resource '" + c.target + "' unavailable");
					break;
				}
				bridge->bind_ib(h, res->offset, res->index_size != 0 ? res->index_size : 4);
				break;
			}

			case Command::Kind::bind_slot:
				warn_once("slot:" + list.name,
					"descriptor-slot rebinding (cs-t/vs-cb/ps-t) needs root-signature tracking (M4)");
				break;

			case Command::Kind::res_copy:
			case Command::Kind::res_copy_desc:
			{
				ScriptResource *src = find_resource(c.target);
				ScriptResource *dst = find_resource(c.dest);
				if (src == nullptr || dst == nullptr)
				{
					warn_once("copy:" + c.dest, "copy: unknown resource in '" +
						c.dest + " = copy " + c.target + "'");
					break;
				}

				if (c.kind == Command::Kind::res_copy_desc)
				{
					// Adopt the source layout; keep instance when sized right.
					dst->stride = src->stride;
					dst->index_size = src->index_size;
					if (dst->handle != 0 && dst->size == src->size)
						break;
					if (!dst->game_capture && dst->handle != 0)
						; // fall through to re-copy below
				}

				const uint64_t sh = ensure_resource(*src, bridge);
				if (sh == 0)
				{
					// Source not live yet: 3DMigoto defers silently.
					break;
				}
				if (src->size == 0 && bridge != nullptr)
					bridge->buffer_size(sh, &src->size); // game captures
				if (src->size != 0 && dst->size != src->size &&
					!dst->game_capture && dst->handle != 0)
				{
					bridge->destroy_buffer(dst->handle);
					dst->handle = 0;
					dst->create_failed = false;
				}
				if (src->size != 0)
					dst->size = src->size;
				const uint64_t dh = ensure_resource(*dst, bridge);
				if (dh == 0 || bridge == nullptr)
				{
					warn_once("copydst:" + c.dest, "copy: destination '" + c.dest + "' unavailable");
					break;
				}
				bridge->copy_buffer(sh, src->offset, dh, 0, src->size != 0 ? src->size : dst->size);
				break;
			}

			case Command::Kind::res_ref:
			{
				ScriptResource *src = find_resource(c.target);
				ScriptResource *dst = find_or_create_resource(c.dest);
				if (src == nullptr || dst == nullptr)
				{
					warn_once("ref:" + c.dest, "ref: unknown resource in '" +
						c.dest + " = ref " + c.target + "'");
					break;
				}
				const uint64_t sh = ensure_resource(*src, bridge);
				dst->handle = sh;
				dst->offset = sh != 0 ? src->offset : 0;
				dst->stride = src->stride;
				dst->index_size = src->index_size;
				dst->size = src->size;
				dst->game_capture = src->game_capture;
				dst->create_failed = false;
				break;
			}

			case Command::Kind::res_capture:
			{
				ScriptResource *dst = find_or_create_resource(c.dest);
				if (dst == nullptr)
					break;
				capture_game_slot(*dst, c.target, ia, bridge);
				break;
			}

			case Command::Kind::res_null:
			{
				ScriptResource *dst = find_resource(c.dest);
				if (dst == nullptr)
					break;
				if (dst->handle != 0 && !dst->game_capture && bridge != nullptr)
					bridge->destroy_buffer(dst->handle);
				dst->handle = 0;
				dst->size = 0;
				dst->offset = 0;
				dst->create_failed = false;
				break;
			}

			case Command::Kind::check_override:
				// The M1 hash-replacement engine already rewrites matched
				// texture slots continuously; nothing to do per draw.
				break;
			}
		}
	}
}

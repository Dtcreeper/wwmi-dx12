// WWMI-DX12: 3DMigoto command-list script model (M3).
//
// The Lynae-class WWMI mods drive everything through ini script
// sections: [Constants], [Key*], [Present], [CommandList*],
// [CustomShader*], [ShaderOverride*] and inline scripts inside
// [TextureOverride*]. This header models the parsed form:
//
//  - Command      one executable line (assignment, run, drawindexed,
//                 vb/ib/slot binding, resource copy/ref/null, ...)
//  - CommandList  a flattened body with if/else/endif compiled to
//                 conditional jumps (parse-time block resolution)
//  - ConstantDef  'global [persist] $var = value'
//  - KeyBinding   '[Key*] key/type/$var = a,b,c' cycling
//  - ShaderOverrideRule  '[ShaderOverride*] hash + body' (parsed;
//                 pipeline-hash matching arrives with M3 runtime)
//
// Semantics mirrored from 3DMigoto:
//  - 'run = CommandListXxx' executes the named command list inline
//  - if/else/endif nest arbitrarily; expressions use wwmi::expr
//  - 'ResourceX = copy/copy_desc/ref ResourceY' moves GPU data
//  - 'ResourceX = ref vb0' captures the game's CURRENT vb0 binding
//    into the resource (3DMigoto ResourceCopySource semantics)
//  - 'vbN = [ref] ResourceX' / 'ib = ResourceX' rebind IA slots for
//    the drawindexed commands that follow in the same list
//  - 'drawindexed = count, firstIndex, baseVertex[, instances]'
//  - 'handling = skip' inside a body blocks the intercepted draw
//  - 'post $var = expr' assignments run after the rest of the list
//  - namespaced references (run = CommandList\WWMIv1\X,
//    $\WWMIv1\y, Resource\WWMIv1\z) target the WWMI framework and
//    are recorded with unsupported flags -- the M3 converter strips
//    them from converted mods
#pragma once

#include "expr.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace wwmi
{
	// ---------------------------------------------------------------------
	// Blend override (M4): 'blend = OP SRC DEST' / 'blendalpha = ...' /
	// 'blend_factor[N] = v' inside a command list (CustomShader-style).
	// Applied to drawindexed commands until the enclosing list ends.
	// ---------------------------------------------------------------------

	// WWMI-DX12 neutral blend codes. Values 0..11 intentionally line up
	// with reshade::api::blend_op / blend_factor; >=12 need the mapping
	// table in the addon bridge (api has gaps there).
	struct BlendConfig
	{
		bool enable = false;
		bool has_color = false;                     // 'blend' present
		uint8_t op = 0, src = 0, dst = 0;          // color: ADD BLEND_FACTOR INV_BLEND_FACTOR
		bool has_alpha = false;                    // 'blendalpha' present
		uint8_t alpha_op = 0, alpha_src = 0, alpha_dst = 0;
		float factors[4] = { 0.0f, 0.0f, 0.0f, 0.0f }; // OMSetBlendFactor
		bool has_factors = false;                  // any blend_factor[N] seen
		uint8_t factor_mask = 0;                   // bit N: factors[N] set here (partial update)
	};

	// 3DMigoto blend tokens -> neutral codes ('ADD', 'BLEND_FACTOR', ...).
	bool parse_blend_op(std::string_view token, uint8_t &out);
	bool parse_blend_factor(std::string_view token, uint8_t &out);

	// ---------------------------------------------------------------------
	// Executable command (one script line)
	// ---------------------------------------------------------------------

	struct Command
	{
		enum class Kind : uint8_t
		{
			// control flow (targets are indices into the body)
			jump,        // unconditional; target = next pc
			jump_false,  // if (!cond) pc = target; value = condition
			// variables
			assign,      // $target = value
			post_assign, // post $target = value (deferred to list end)
			// structure
			run,         // run = <command list name> (target)
			// drawing
			drawindexed, // drawindexed = count, first_index, base_vertex
			handling_skip, // handling = skip
			blend_state, // blend/blendalpha/blend_factor: OM override
			// IA bindings (for re-issued draws)
			bind_vb,     // vbN = [ref] ResourceX (slot, target, is_ref)
			bind_ib,     // ib  = [ref] ResourceX
			bind_slot,   // <stage>-t/u/cb<N> = [ref] ResourceX
			// resource ops
			res_copy,     // ResourceX = copy ResourceY
			res_copy_desc,// ResourceX = copy_desc ResourceY
			res_ref,      // ResourceX = ref ResourceY (alias GPU data)
			res_capture,  // ResourceX = ref vb0/ib/vb1.. (capture game slot)
			res_null,     // ResourceX = null
			// misc
			check_override, // CheckTextureOverride = ps-tN (stage+slot)
		};

		Kind kind = Kind::assign;
		std::string target;        // variable / resource / list name (normalized)
		std::string dest;          // res_* ops: destination resource name
		expr::NodePtr value;       // condition or rhs expression
		uint32_t jump_target = 0;  // jump destination (index)

		// drawindexed = <count>, <first_index>, <base_vertex>[, <instances>]
		uint32_t index_count = 0;
		uint32_t first_index = 0;
		int32_t base_vertex = 0;
		uint32_t instance_count = 1;

		uint32_t slot = 0;         // vbN / tN / uN / cbN index
		char stage = 0;            // 'v','p','c' (vs/ps/cs)
		char slot_type = 0;        // 't' srv, 'u' uav, 'c' cbv (bind_slot)
		bool is_ref = false;       // leading 'ref' keyword

		// blend_state: the config carried by this command
		BlendConfig blend;

		// Original line for diagnostics.
		std::string source;
	};

	// A flattened script body with resolved if/else/endif jumps.
	struct CommandList
	{
		std::string name;          // normalized, lowercase
		std::string section;       // original section header
		std::vector<Command> body;

		bool empty() const { return body.empty(); }
	};

	// [Constants] entry. All variables live in one global namespace
	// (3DMigoto 'global' vs local only affects sharing, which M3
	// approximates by making everything global).
	struct ConstantDef
	{
		std::string name; // normalized, no '$'
		float value = 0.0f;
		bool persist = false;
		std::string source_section;
	};

	// [Key*] binding: cycles a variable through a value list.
	struct KeyBinding
	{
		std::string name;         // section name (diagnostics)
		std::string var;          // normalized variable name
		std::vector<float> values;// cycle list ('0,1,2')
		uint32_t vk = 0;          // Windows virtual-key code
		std::string key_name;     // original key text (diagnostics)
		expr::NodePtr condition;  // optional condition expression
	};

	// NOTE: [ShaderOverride*] rules live in mod_rules.hpp
	// (ShaderOverrideRule); their bodies parse into ordinary CommandLists
	// (see parse_script_body) keyed by the normalized section name.

	// ---------------------------------------------------------------------
	// Parsing
	// ---------------------------------------------------------------------

	// Parses the body of one script section ('[CommandListXxx]',
	// '[Present]', '[CustomShaderXxx]' or the inline part of a
	// '[TextureOverride*]' section). 'lines' are the raw 'key = value'
	// pairs in file order; keys without '=' pass through as raw keys
	// (e.g. 'endif'). Unsupported constructs append to warnings.
	void parse_script_body(const std::string &section,
		const std::vector<std::pair<std::string, std::string>> &lines,
		CommandList &out, std::vector<std::string> &warnings);

	// '[Key*]' section -> KeyBinding. Returns false when the section is
	// not usable (missing key/var); details go to warnings.
	bool parse_key_section(const std::string &section,
		const std::vector<std::pair<std::string, std::string>> &lines,
		KeyBinding &out, std::vector<std::string> &warnings);

	// Parses a key name ('VK_UP', ',', '9', 'ctrl+x' subset) to a
	// virtual-key code. Returns 0 when unknown.
	uint32_t parse_vk(const std::string &key_name);

	// Normalizes a section name: lowercased, with the leading
	// 'commandlist'/'customshader'/'textureoverride' prefix removed.
	std::string normalize_section_name(std::string_view section);

	// True when a name carries a framework namespace ('xx\wwmiv1\yyy').
	bool is_namespaced(std::string_view name);
}

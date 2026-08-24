// WWMI-DX12: 3DMigoto mod.ini rule model (M1 + M2 subset).
//
// Parses the appearance-mod subset of 3DMigoto's ini syntax:
//   [TextureOverride<name>]
//   hash = 0x1234abcd            ; 3DMigoto texture hash (full-token hex)
//   ps-t0 = ResourceBodyDiffuse  ; <stage>-t<slot> = <resource ref>
//   handling = skip              ; M2: skip draws matched by this rule
//   match_index_count = 123      ; M2: draw-context fuzzy filter
//
//   [Resource<name>]
//   filename = Textures/foo.dds  ; relative to the mod root
//
// Semantics mirrored from 3DMigoto:
//  - hash: sscanf("%16llx%n") must consume the entire value (GetIniHash);
//    texture hashes are truncated to 32 bits.
//  - slot keys: '<stage>-t<slot>' with stage in {v,h,d,g,p,c} and
//    slot < 128 (ResourceCopyTarget::ParseTarget).
//  - binding values are lowercased; leading copy-option words
//    (ref/copy/stereo/...) are skipped; the source must be a
//    'resource<name>' reference (custom resource lookup is lowercase).
//  - keys and sections are case-insensitive.
//  - handling (M2): 'skip' blocks the draw call (SkipCommand sets
//    call_info->skip); 'abort' is approximated the same way. Other
//    values (commandlist/next/...) are recorded as unsupported.
//  - match_* (M2): '[op] value' with op in {=,!,<,>,<=,>=}, default '='
//    (parse_fuzzy_numeric_match_expression). Applied to the draw that
//    triggered the rule via matches_draw_info().
#pragma once

#include "ini_file.hpp"

#include <cstdint>
#include <filesystem>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace wwmi
{
	// A '<stage>-t<slot> = resource...' binding line inside a TextureOverride.
	struct ResourceBinding
	{
		char stage = 0;          // 'v','h','d','g','p','c'
		uint32_t slot = 0;       // texture slot index
		std::string resource;    // referenced resource name (lowercase, no 'resource' prefix)
	};

	// -----------------------------------------------------------------------
	// M2: draw-context rule extensions (3DMigoto FuzzyMatch / DrawCallInfo)
	// -----------------------------------------------------------------------

	// 3DMigoto FuzzyMatchOp subset used by the draw-context matchers.
	enum class FuzzyOp : uint8_t
	{
		equal,        // '=' (default)
		not_equal,    // '!'
		less,         // '<'
		less_equal,   // '<='
		greater,      // '>'
		greater_equal // '>='
	};

	// 3DMigoto FuzzyMatch for plain integers: '[op] value'. A matcher only
	// participates in matches_draw_info() when its key was present in the
	// ini (enabled); absent matchers impose no constraint.
	struct FuzzyMatch
	{
		FuzzyOp op = FuzzyOp::equal;
		uint32_t value = 0;
		bool enabled = false;

		bool matches(uint32_t v) const;
	};

	// 3DMigoto DrawCallInfo fields the match_* keys filter on. Field
	// semantics mirror 3DMigoto: non-applicable counts are 0 (e.g.
	// VertexCount on an indexed draw, IndexCount on a non-indexed draw);
	// FirstVertex carries the signed BaseVertexLocation bits.
	struct DrawCallInfo
	{
		uint32_t vertex_count = 0;
		uint32_t index_count = 0;
		uint32_t instance_count = 0;
		uint32_t first_vertex = 0;
		uint32_t first_index = 0;
		uint32_t first_instance = 0;
	};

	// 3DMigoto HandlingMode subset relevant to draw interception.
	enum class HandlingMode : uint8_t
	{
		none,  // rule has no handling command
		skip,  // 'handling = skip': block the draw call
		abort, // 'handling = abort': approximated as skip (cannot abort a
		       // D3D12 command-list recording mid-way)
	};

	struct TextureOverrideRule
	{
		std::string section;     // original section name (diagnostics)
		uint32_t hash = 0;       // 3DMigoto texture hash (low 32 bits)
		bool has_hash = false;
		std::vector<ResourceBinding> bindings; // in file order

		// ---- M2: draw interception ----
		HandlingMode handling = HandlingMode::none;
		FuzzyMatch match_first_vertex;
		FuzzyMatch match_first_index;
		FuzzyMatch match_first_instance;
		FuzzyMatch match_vertex_count;
		FuzzyMatch match_index_count;
		FuzzyMatch match_instance_count;
		bool has_draw_context_match = false; // any match_* key present

		// m1-9: replacement texture resolved by the addon bridge (NOT the
		// parser): the file of the first bound resource, or <hash8>.dds in
		// the mod root. Empty when the rule matches nothing loadable.
		std::filesystem::path texture_path;

		// M3: index into the addon's script-runtime table (set by the
		// loader, not the parser). UINT32_MAX = no script runtime.
		uint32_t runtime = 0xffffffffu;
	};

	// M4: [ShaderOverride<name>] -- keys on the XXH64 of a shader
	// bytecode (3DMigoto ShaderOverride; 64-bit unlike texture hashes).
	// The section body is an inline script owned by the ScriptRuntime
	// (e.g. WWMI attack-latch: '$attack = 1'); the rule parser only
	// records the hash and links the runtime.
	struct ShaderOverrideRule
	{
		std::string section;     // original section name (diagnostics)
		uint64_t hash = 0;       // XXH64 of the shader bytecode
		bool has_hash = false;

		// M4: index into the addon's script-runtime table (set by the
		// loader). UINT32_MAX = no script runtime.
		uint32_t runtime = 0xffffffffu;
	};

	// 3DMigoto matches_draw_info: a rule with any match_* key only applies
	// to draws whose context matches every matcher. A rule without
	// match_* keys applies to every draw.
	bool matches_draw_info(const TextureOverrideRule &rule, const DrawCallInfo &call);

	struct ResourceDef
	{
		std::string section;     // original section name
		std::string filename;    // relative to mod root (original case)
		// Optional overrides (recorded; used when the DDS header is missing
		// or overridden by the mod):
		std::string format;
		int32_t width = -1;
		int32_t height = -1;
		int32_t depth = -1;
		int32_t mips = -1;
		int32_t array = -1;
	};

	struct ModRules
	{
		std::filesystem::path mod_dir;  // mod root = parent dir of mod.ini
		std::vector<TextureOverrideRule> overrides;
		std::vector<ShaderOverrideRule> shader_overrides; // M4
		std::unordered_map<std::string, ResourceDef> resources; // key: lowercase section name
		std::vector<std::string> warnings; // non-fatal parse diagnostics
	};

	// Loads and validates a mod.ini. Returns false only when the file cannot
	// be read; individual bad sections/keys degrade to warnings.
	bool load_mod_rules(const std::filesystem::path &ini_path, ModRules &out);

	// ---- shared low-level parsers (exposed for unit tests) ----

	// 3DMigoto GetIniHash: '%16llx' must consume the whole token.
	bool parse_hash(const std::string &value, uint64_t &out_hash);

	// '<stage>-t<slot>' key parser. Returns false if the key is not a
	// texture-slot binding (caller may then treat it as another command).
	bool parse_texture_slot_key(std::string_view key, char &out_stage, uint32_t &out_slot);

	// Binding value parser: skips leading copy-option words, then resolves
	// 'resource<name>' references (lowercased). Returns false for special
	// sources (this/null/bb/...) which M1 does not support yet.
	bool parse_resource_ref(std::string_view value, std::string &out_resource);

	// 3DMigoto parse_fuzzy_numeric_match_expression subset:
	// '[op] value' with op in {=,!,<,>,<=,>=} (default '='), value a plain
	// unsigned integer. Field-name right-hand sides (width/height/...) are
	// not supported for draw-context matches and parse as no-match.
	// On success out.enabled is set to true.
	bool parse_fuzzy_match(const std::string &value, FuzzyMatch &out);
}

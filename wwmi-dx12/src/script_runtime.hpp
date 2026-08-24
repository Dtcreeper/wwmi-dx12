// WWMI-DX12: 3DMigoto script runtime (M3).
//
// ScriptRuntime owns one mod's script state and executes it:
//
//   vars       [Constants] '$x = v' + persist overlay (saved/loaded as
//              an ini file under the WWMI data root)
//   keys       [Key*] cycle bindings; poll_keys() edge-detects the VK
//   lists      every [CommandList*] / [CustomShader*] body, plus the
//              merged [Present] script and the inline bodies of
//              [TextureOverride*] sections (keyed by normalized name)
//   resources  [Resource*] definitions with lazily instantiated GPU
//              buffers (.buf upload), 'ref vb0/ib' game-slot captures
//              and ref/copy/copy_desc/null semantics
//
// Execution model (execute()): a flattened pc walk over Command bodies
// (if/else/endif already compiled to jumps). 'run = X' recurses with a
// depth limit; namespaced targets (\WWMIv1\...) are parse-time no-ops.
// GPU access goes through GpuBridge so the runtime stays ReShade-free
// and unit-testable with a fake bridge.
//
// Semantics mirrored from 3DMigoto:
//  - variables are floats; undefined reads 0
//  - 'ResourceX == null' tests GPU liveness (no live instance = null)
//  - 'post $x = v' runs after the rest of the list
//  - 'ResourceX = ref vb0' captures the game's CURRENT slot binding;
//    'vb0 = ref ResourceX' rebinds from the capture
//  - drawindexed re-issues draws through the bridge (draw context only)
#pragma once

#include "expr.hpp"
#include "script_model.hpp"

#include <cstdint>
#include <filesystem>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace wwmi
{
	class BufferTracker; // IA game-slot capture source (fwd)
	struct IaState;

	// ---------------------------------------------------------------------
	// GPU bridge (implemented over ReShade in the addon; faked in tests)
	// ---------------------------------------------------------------------

	class GpuBridge
	{
	public:
		virtual ~GpuBridge() = default;

		// Creates a committed buffer and uploads 'data' (initial-data
		// path). want_uav/want_sr add unordered-access / shader-resource
		// usage (RWBuffer resources). Returns 0 on failure.
		virtual uint64_t create_buffer(const void *data, uint64_t size,
			bool want_uav, bool want_sr) = 0;
		virtual void destroy_buffer(uint64_t handle) = 0;

		// Byte size of a live game buffer (for copy sizing).
		virtual bool buffer_size(uint64_t handle, uint64_t *size) = 0;

		// Recording-time operations (draw context) or immediate-command-
		// list operations (present context) -- the bridge instance owns
		// the target; the runtime does not care which.
		virtual void bind_vb(uint32_t slot, uint64_t resource,
			uint64_t offset, uint32_t stride) = 0;
		virtual void bind_ib(uint64_t resource, uint64_t offset,
			uint32_t index_size) = 0;
		virtual void draw_indexed(uint32_t index_count, uint32_t first_index,
			int32_t base_vertex, uint32_t instance_count) = 0;
		virtual void copy_buffer(uint64_t src, uint64_t src_offset,
			uint64_t dst, uint64_t dst_offset, uint64_t size) = 0;

		// M4 OM blend override. Applies the config to the draws that
		// follow (clone of the current graphics PSO with a rewritten
		// blend subobject + OMSetBlendFactor). Returns a non-zero token
		// for revoke_blend, 0 when no override could be applied (e.g. no
		// bound PSO / clone failed) -- the draw then runs unblended.
		virtual uint64_t apply_blend(const BlendConfig &cfg) = 0;
		virtual void revoke_blend(uint64_t token) = 0;
	};

	// ---------------------------------------------------------------------
	// Resource state
	// ---------------------------------------------------------------------

	// One [Resource*] definition + its runtime instance.
	struct ScriptResource
	{
		std::string name;      // normalized (lowercase, no 'resource' prefix)
		std::string section;   // original section name
		std::string filename;  // relative to mod root ('' = synthesized)
		std::string type;      // buffer / rwbuffer / texture... (lowercase)
		std::string format;    // dxgi format name (lowercase)
		uint32_t stride = 0;   // VB element stride (binding hint)
		uint32_t array = 0;    // element count for synthesized buffers

		// Runtime instance (created lazily on first use).
		uint64_t handle = 0;      // bridge buffer handle; 0 = not live
		uint64_t offset = 0;      // captured game-slot bindings only
		uint32_t index_size = 0;  // IB: 2/4 from format
		uint64_t size = 0;        // byte size once known
		bool game_capture = false; // 'ref vb0/ib': aliases a game buffer
		bool create_failed = false;
	};

	// ---------------------------------------------------------------------
	// ScriptRuntime
	// ---------------------------------------------------------------------

	class ScriptRuntime
	{
	public:
		// Parses mod.ini (or any ini) into vars/keys/lists/resources.
		// Non-fatal problems append to warnings (also mirrored into the
		// parser warnings of the individual sections).
		bool load(const std::filesystem::path &ini_path);

		const std::filesystem::path &mod_dir() const { return _mod_dir; }
		const std::string &mod_name() const { return _mod_name; }
		const std::vector<std::string> &warnings() const { return _warnings; }

		// ---- variables ----
		bool get_var(const std::string &name, float &out) const;
		float var_or(const std::string &name, float fallback) const;
		void set_var(const std::string &name, float value);
		size_t var_count() const { return _vars.size(); }

		// All variables, name-sorted (overlay variable viewer).
		std::vector<std::pair<std::string, float>> vars_snapshot() const;

		// persist overlay: '<name>=<value>' lines; only persist-flagged
		// names are restored/written.
		void load_persist(const std::filesystem::path &file);
		void save_persist(const std::filesystem::path &file) const;

		// ---- keys ----
		// Polls one VK (edge detection is the caller's: 'down' = current
		// state). Evaluates the binding condition, then cycles.
		void on_key(uint32_t vk, bool down);
		const std::vector<KeyBinding> &keys() const { return _keys; }

		// ---- execution ----
		// Executes the merged [Present] script (present context: no IA
		// captures possible; drawindexed would warn).
		void run_present(GpuBridge *bridge);

		// Executes one command list by normalized name. Returns false
		// when no such list exists.
		bool run_list(const std::string &name, GpuBridge *bridge, const IaState *ia);

		// Executes the inline script of a [TextureOverride*] section in
		// the intercepted draw's context. Returns true when the body
		// executed 'handling = skip' (the original draw must be
		// blocked); drawindexed commands inside the body have already
		// re-issued the replacement draws through the bridge.
		bool run_override(const std::string &section, GpuBridge *bridge, const IaState *ia);

		// Executes a parsed body (e.g. a TextureOverride inline script).
		void execute(const CommandList &list, GpuBridge *bridge, const IaState *ia);

		// ---- resources ----
		ScriptResource *find_resource(const std::string &name);
		const ScriptResource *find_resource(const std::string &name) const;

		// Finds or implicitly registers a resource ('ResourceX = ref vb0'
		// targets are often declared nowhere; 3DMigoto instantiates them).
		ScriptResource *find_or_create_resource(const std::string &name);

		// True when the resource has a live instance (== null checks).
		bool resource_alive(const std::string &name) const;

		// Releases all created buffers (device teardown).
		void destroy_resources(GpuBridge *bridge);

		// Inline bodies of [TextureOverride*] sections by normalized
		// section name (the addon bridge attaches them to rules).
		const CommandList *find_override_script(const std::string &section) const;

		// Execution stats (overlay).
		uint64_t lists_executed = 0;
		uint64_t draws_issued = 0;
		uint64_t commands_warned = 0;

	private:
		void exec_body(const CommandList &list, GpuBridge *bridge,
			const IaState *ia, int depth, std::vector<const Command *> &post);

		// Ensures a resource has a live GPU instance. File-backed: load +
		// upload once. Synthesized (RWBuffer array=N): zero buffer.
		// Returns 0 when unavailable.
		uint64_t ensure_resource(ScriptResource &res, GpuBridge *bridge);

		// Captures 'vb0'/'vb1'.../'ib' from the game IA state.
		bool capture_game_slot(ScriptResource &res, const std::string &slot,
			const IaState *ia, GpuBridge *bridge);

		void warn_once(const std::string &key, const std::string &msg);

		std::filesystem::path _mod_dir;
		std::string _mod_name; // mod directory name (diagnostics)

		std::unordered_map<std::string, float> _vars;
		std::unordered_set<std::string> _persist_vars;

		std::vector<KeyBinding> _keys;
		CommandList _present; // merged [Present] bodies
		std::unordered_map<std::string, CommandList> _lists;      // commandlist/customshader
		std::unordered_map<std::string, CommandList> _override_scripts; // textureoverride

		std::unordered_map<std::string, ScriptResource> _resources;

		std::vector<std::string> _warnings;
		std::unordered_set<std::string> _warned; // warn_once keys
		std::unordered_map<uint32_t, bool> _key_prev; // VK edge detection
		bool _skip = false; // run_override: 'handling = skip' executed

		// M4: active OM blend inside the currently executing top-level
		// list ('blend = ...' merges in; drawindexed applies it per draw;
		// cleared when the top-level execution returns).
		BlendConfig _active_blend;
		bool _blend_active = false;
	};
}

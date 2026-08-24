// WWMI-DX12: 3DMigoto-compatible texture override layer for DX12 games
// (Wuthering Waves), implemented as a ReShade addon.
//
// m1-9 runtime pipeline:
//
//   track      init_resource -> TextureTracker (candidate 2D textures)
//   learn      initial data hash, or GPU readback at present time
//   match      learned hash -> HashIndex (direct, or via session pairing)
//   activate   rule matched -> load mod DDS, upload, create SRV,
//              OverrideEngine::activate -> rewrite every known slot
//   maintain   update_descriptor_tables (create path) and
//              copy_descriptor_tables (copy path) keep slots overridden
//              while the game (re)writes or copies descriptors
//
// M1 semantics note: a matched rule replaces the SLOTS the matched texture
// is bound to (texture-content replacement). Full 3DMigoto binding commands
// ('ps-t0 = resource' retargeting other slots at draw time) require root
// signature tracking and land in M3.
//
// m2-5 runtime pipeline (draw interception):
//
//   gate       draw_intercept flag: zero per-draw overhead unless a loaded
//              mod actually uses 'handling = skip'
//   bind       bind_index_buffer / bind_vertex_buffers replay the IA state
//              into the command list's IaState and lazily admit the
//              buffer into the pool (the VB/IB role is only known here)
//   learn      present-time GPU readback -> calc_buffer_hash (3DMigoto
//              data+D3D11_BUFFER_DESC formula), budgeted per frame
//   skip       draw / draw_indexed -> find_skip_rule: looks the draw up by
//              the bound IB hash (indexed draws) and VB slot 0 hash against
//              draw-intercepting rules; a match with passing match_* filters
//              returns true and the draw never reaches the driver
// ReShade overlay (ImGui 1.92.5 function table): ImTextureID must be a
// 64-bit type so it matches api::resource_view.
//
// NOTE: imgui.h MUST be included before anything that pulls in reshade.hpp
// (e.g. log.hpp): reshade_overlay.hpp only emits its inline ImGui wrappers
// when IMGUI_VERSION_NUM is already defined, otherwise the ImGui functions
// stay declared-but-undefined and the link fails.
#define ImTextureID ImU64
#include <imgui.h>
#include "reshade.hpp"

#include "log.hpp"
#include "mod_rules.hpp"
#include "texture_hash.hpp"
#include "texture_tracker.hpp"
#include "buffer_hash.hpp"
#include "buffer_tracker.hpp"
#include "override_engine.hpp"
#include "dds_loader.hpp"
#include "script_runtime.hpp"
#include "pipeline_tracker.hpp"
#include "skeleton_chain.hpp"

// M4: direct OMSetBlendFactor for full float precision (ReShade's
// dynamic_state path would round-trip through uint32 bit patterns).
#include <d3d12.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <mutex>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace wwmi
{

	// ---------------------------------------------------------------------
	// Global runtime state
	// ---------------------------------------------------------------------

	// A replacement texture + SRV pair created from a mod DDS. Cached per
// file path so re-activations (game recreating a texture) reuse the GPU
// copy instead of leaking per-match duplicates.
	struct Replacement
	{
		reshade::api::resource texture{};
		reshade::api::resource_view view{};
	};

	// Deferred activation request produced by record_hash().
	struct ActivationReq
	{
		uint64_t resource = 0;
		std::filesystem::path texture_path;
		std::string section;
	};

	// M9: reference to a push-CBV root parameter (param index + binding
	// inside the parameter). UINT32_MAX param = not found.
	struct SkelParamRef
	{
		uint32_t param = UINT32_MAX;
		uint32_t binding = 0;

		bool found() const { return param != UINT32_MAX; }
	};

	struct RuntimeState
	{
		std::mutex lock;

		TextureTracker tracker;
		HashIndex index;
		SessionCache session;
		OverrideEngine engine;

		// ---- M2: draw interception ----
		BufferTracker buffers;

		// ---- M6: pooled index-buffer views ----
		// UE DX12 suballocates mesh IBs in 32 MB pools; DrawWindowIndex
		// (immutable after load_mods) turns the mod's windowed rules
		// into region-probe targets, IndexViewTracker caches per-view
		// verification, view_probes ferries work from the draw thread
		// (enqueue) to present (GPU readback + hash compare).
		DrawWindowIndex windows;
		IndexViewTracker index_views;

		struct IndexProbe
		{
			uint64_t handle = 0;
			uint64_t offset = 0;      // view base inside the pool
			uint32_t index_size = 0;
			uint32_t group_hash = 0;  // expected 3DMigoto hash
			uint32_t min_first_index = 0;
			uint32_t span_bytes = 0;  // region size hashed
			uint32_t retries = 0;
			std::string section;      // rule group name (diagnostics)
		};
		std::vector<IndexProbe> view_probes;

		// ---- M4: pipeline / shader-hash tracking ----
		// init_pipeline fills this; draw-time ShaderOverride matching and
		// the blend PSO clones read from it.
		PipelineTracker pipelines;

		// ShaderOverride rule index: XXH64 shader hash -> runtime +
		// section. Filled once in load_mods() and immutable afterwards,
		// so draw-time lookups need no lock.
		struct ShaderOverrideEntry
		{
			uint32_t runtime = 0xffffffffu;
			std::string section;
		};
		std::unordered_map<uint64_t, ShaderOverrideEntry> shader_overrides;
		std::atomic<bool> shader_override_active{ false };

		// Per-command-list graphics pipeline (bind_pipeline replay).
		// Guarded by <lock> like draw_states.
		std::unordered_map<uint64_t, uint64_t> cl_pipeline;
		std::unordered_map<uint64_t, uint64_t> cl_compute_pipeline;

		// Blend override PSO cache: (src pipeline, config) -> clone.
		// Clones are created by AddonBridge::apply_blend and destroyed
		// lazily (see pending_pipeline_destroys).
		struct BlendClone
		{
			uint64_t src = 0;
			reshade::api::pipeline pipe{};
		};
		std::unordered_map<uint64_t, BlendClone> blend_clones;
		std::mutex blend_lock; // guards blend_clones only
		std::vector<reshade::api::pipeline> pending_pipeline_destroys;

		// ---- M3: script runtimes ----
		// One per loaded mod (mod.ini). The mutex is held while the
		// script executes; bridge calls inside never take <lock>
		// themselves, and the event handlers below never take a runtime
		// mutex while holding <lock> -- lock order is runtime -> lock.
		struct ModRuntime
		{
			ScriptRuntime rt;
			std::mutex lock;
		};
		std::vector<std::unique_ptr<ModRuntime>> runtimes;
		bool has_scripts = false; // any runtime with keys/present/override scripts

		// Per-command-list IA state, keyed by the command list's native
		// handle. Entries appear lazily from the bind events and disappear
		// in destroy_command_list.
		std::unordered_map<uint64_t, IaState> draw_states;

		// ---- M8: root signature tracking ----
		// Graphics root state per command list, maintained from the
		// push_descriptors / bind_descriptor_tables events, plus the
		// parameter layout of every pipeline layout the game created.
		// The skeleton system needs this to find the bone CBs the game's
		// skinning VS reads (DX11's vs-cb3/vs-cb4) and to rebind them
		// for the mod's re-issued draws.
		struct RootCbvBinding
		{
			uint64_t buffer = 0; // resource handle
			uint64_t offset = 0; // byte offset into the buffer
			uint64_t size = UINT64_MAX; // range size from the push (UINT64_MAX = whole)
		};
		struct ClRootState
		{
			uint64_t layout = 0; // current graphics pipeline layout
			std::unordered_map<uint64_t, RootCbvBinding> cbvs;   // (param<<32)|register
			std::unordered_map<uint32_t, uint64_t> tables;       // param -> table base
		};
		std::unordered_map<uint64_t, ClRootState> cl_root;
		std::unordered_map<uint64_t,
			std::vector<reshade::api::pipeline_layout_param>> layouts;

		// ---- M9: skeleton chain ----
		// Config from mod.ini [SkeletonChain] (immutable after load_mods);
		// skel_snap is the latest skeleton-CBV snapshot taken at a component
		// draw (written under <lock>); the upload buffers and region layout
		// are owned by the present thread, only the current slot's resource
		// handle is published for the draw thread.
		SkeletonChain skel;
		struct SkelSnapshot
		{
			uint64_t layout = 0;
			uint32_t cb3_param = UINT32_MAX; // root param bound at vs-cb3 (extra skeleton)
			uint32_t cb3_binding = 0;        // binding index inside that param
			uint32_t cb4_param = UINT32_MAX; // root param bound at vs-cb4 (main skeleton)
			uint32_t cb4_binding = 0;        // binding index inside that param
			uint64_t main_buf = 0, main_off = 0;
			uint64_t extra_buf = 0, extra_off = 0;
			uint64_t frame = 0;
			bool valid = false;
		};
		SkelSnapshot skel_snap;
		// layout handle -> (cb3 param/binding, cb4 param/binding); UINT32_MAX
		// param means "identified as absent" (cached negative result).
		std::unordered_map<uint64_t,
			std::pair<SkelParamRef, SkelParamRef>> skel_layout_params;
		bool skel_dumped = false; // one-time identification/content dump done

		// Published upload state (present thread writes under <lock>).
		uint64_t skel_gpu_buf = 0; // current ping-pong upload buffer handle
		bool skel_gpu_ready = false;
		// Region offsets inside the upload buffer (immutable once built).
		struct SkelRegions
		{
			uint64_t merged_off = 0, extra_off = 0;
			uint64_t remap_off[SkeletonChain::k_remap_tables] = {};
			uint64_t extra_remap_off[SkeletonChain::k_remap_tables] = {};
			uint64_t remap_bytes[SkeletonChain::k_remap_tables] = {};
			uint64_t total = 0;
		};
		SkelRegions skel_regions;
		// Counters for the detach stats line.
		uint64_t skel_captures = 0;
		uint64_t skel_pushes = 0;

		// Set after load_mods() when any loaded rule uses handling =
		// skip/abort. Bind/draw handlers early out on it: texture-only mod
		// sets pay nothing per draw. Every access to index / buffers /
		// draw_states below it happens under <lock> anyway, so a stale
		// false only delays tracking, never corrupts it.
		std::atomic<bool> draw_intercept{ false };

		// ReShade overlay visibility (reshade_open_overlay event). Mod
		// key bindings pause while it is open so typing into the overlay
		// does not cycle mod variables.
		std::atomic<bool> overlay_open{ false };

		// SRV handle -> resource handle, for SRVs of tracked textures (the
		// update event only tells us the view, not the resource). Bounded;
		// pruned when the resource is destroyed.
		std::unordered_map<uint64_t, uint64_t> view_resource;
		static constexpr size_t k_max_view_records = 65536;

		// Replacement textures we created, by mod file path.
		std::unordered_map<std::string, Replacement> replacements;

		// Matched rules waiting for GPU work (processed outside the lock).
		std::vector<ActivationReq> activations;

		// Most recent activations, newest first (overlay display only).
		struct ActivationRecord
		{
			uint64_t resource = 0;
			std::string texture;
			uint64_t frame = 0;
			uint32_t slot_rewrites = 0;
		};
		std::vector<ActivationRecord> activation_log;
		static constexpr size_t k_max_activation_log = 32;

		// ---- M2: draw-skip bookkeeping (overlay display) ----
		struct SkipRecord
		{
			uint64_t count = 0; // draws skipped by this rule
			uint64_t frame = 0; // last skip
		};
		std::unordered_map<std::string, SkipRecord> skip_log; // key: section

		// Current resource states, learned from the game's own barriers
		// (ReShade mirrors every D3D12 ResourceBarrier into the barrier
		// event). Needed to transition a texture to copy_source for
		// readback without guessing.
		std::unordered_map<uint64_t, reshade::api::resource_usage> states;

		// Set in init_device (D3D12 only) before any other event of that
		// device can fire; read lock-free by the event handlers.
		std::atomic<reshade::api::device *> device{ nullptr };

		// The game's graphics command queue (immediate command list entry
		// point). Tracked via init_command_queue; read lock-free.
		std::atomic<reshade::api::command_queue *> queue{ nullptr };
		uint64_t frame = 0;
		bool mods_loaded = false;
		bool session_dirty = false;

		// Actual Mods directory in use (data_root()/Mods), recorded at
		// load_mods() time for the overlay/log so a wrong WWMI_HOME or a
		// missing %APPDATA% tree is immediately visible.
		std::string mods_root;

		// Mod manager catalog: every mod directory discovered under the
		// Mods root at load_mods() time. 'enabled' is the persisted
		// toggle (mods-disabled.txt, absent = enabled); 'loaded' is
		// what actually happened this session. The overlay edits
		// 'enabled' and rewrites the file immediately -- changes apply
		// on the NEXT launch (no hot reload by design).
		struct ModCatalogEntry
		{
			std::string name;
			bool enabled = true;
			bool loaded = false;
		};
		std::vector<ModCatalogEntry> mod_catalog;

		// Aggregate counters (atomic: events fire on multiple threads).
		std::atomic<uint64_t> devices = 0;
		std::atomic<uint64_t> srv_views = 0;
		std::atomic<uint64_t> hash_matches = 0; // learned hash hits a mod rule
		std::atomic<uint64_t> readback_hashes = 0;
		std::atomic<uint64_t> initialdata_hashes = 0;
		std::atomic<uint64_t> descriptor_updates = 0; // SRV-type update events
		std::atomic<uint64_t> descriptor_copies = 0;  // copy events
		std::atomic<uint64_t> overwrites_applied = 0; // descriptor rewrites
		std::atomic<uint64_t> active_replacements = 0;

		// M2 counters
		std::atomic<uint64_t> buffer_readback_hashes = 0;
		std::atomic<uint64_t> draws_skipped = 0;

		// M6 counters
		std::atomic<uint64_t> view_probes_issued = 0;
		std::atomic<uint64_t> views_verified = 0;

		// M3 counters
		std::atomic<uint64_t> script_draws_blocked = 0;  // handling=skip executed
		std::atomic<uint64_t> script_draws_issued = 0;   // drawindexed re-issued via bridge
		std::atomic<uint64_t> script_lists_executed = 0; // command lists run

		// M4 counters
		std::atomic<uint64_t> shader_overrides_hit = 0;  // ShaderOverride bodies executed
		std::atomic<uint64_t> blend_pso_clones = 0;      // PSO clones created
		std::atomic<uint64_t> blend_draws = 0;           // draws issued under a blend override
	};

	static RuntimeState g_state;

	// Set while this thread creates its own replacement resources so the
	// init_resource / init_resource_view hooks do not track or record them
	// (they must never be hashed, matched or replaced).
	static thread_local bool t_own_creation = false;

	struct OwnCreationGuard
	{
		OwnCreationGuard() { t_own_creation = true; }
		~OwnCreationGuard() { t_own_creation = false; }
	};

	// Set while a mod script executes (and while the interceptor restores
	// the IA state afterwards). The bind/draw event handlers early out on
	// it: the script's own vb/ib rebinds and re-issued drawindexed calls
	// must not update IaState snapshots, enter the buffer pool or recurse
	// into draw interception.
	static thread_local bool t_script_exec = false;

	struct ScriptExecGuard
	{
		ScriptExecGuard() { t_script_exec = true; }
		~ScriptExecGuard() { t_script_exec = false; }
	};

	// Data root: EVERYTHING we create (the Mods tree and the session
	// cache) lives outside the game directory on purpose -- only ReShade
	// and the addon binary itself reside there, keeping the game install
	// clean. Resolution order:
	//   1. WWMI_HOME environment variable, when set
	//   2. %APPDATA%\WWMI-DX12
	//   3. exe directory (last resort for portable/test setups)
	static std::filesystem::path data_root()
	{
		wchar_t env[MAX_PATH] = {};
		if (GetEnvironmentVariableW(L"WWMI_HOME", env, MAX_PATH) > 0)
			return std::filesystem::path(env);

		wchar_t appdata[MAX_PATH] = {};
		if (GetEnvironmentVariableW(L"APPDATA", appdata, MAX_PATH) > 0)
			return std::filesystem::path(appdata) / L"WWMI-DX12";

		wchar_t exe[MAX_PATH] = {};
		GetModuleFileNameW(nullptr, exe, MAX_PATH);
		return std::filesystem::path(exe).parent_path();
	}

	// ---------------------------------------------------------------------
	// Mod loading: <data_root>/Mods/*/mod.ini -> HashIndex + resolved
	// textures (data_root lives OUTSIDE the game directory, see above)
	// ---------------------------------------------------------------------

	// Resolves the replacement texture file of a rule. The parser stays
	// pure (syntax only); path resolution is bridge policy:
	//   1. the file of the first '<stage>-t<slot> = resource...' binding
	//   2. 3DMigoto's hash-named file convention: <hash8>.dds in mod root
	static void resolve_rule_textures(ModRules &mod)
	{
		for (TextureOverrideRule &rule : mod.overrides)
		{
			if (!rule.has_hash)
				continue;

			// Draw-intercepting rules (handling = skip/abort) act through
			// the M2/M3 draw pipeline, not the texture-replacement path.
			// A texture warning for them is noise.
			if (rule.handling != HandlingMode::none)
				continue;

			if (!rule.bindings.empty())
			{
				// parse_resource_ref strips the 'resource' prefix
				// ('ResourceTexture0' -> 'texture0') while the resource
				// table keys are full lowercased section names
				// ('resourcetexture0'). Try both spellings.
				const std::string &ref = rule.bindings.front().resource;
				auto rit = mod.resources.find(ref);
				if (rit == mod.resources.end())
					rit = mod.resources.find("resource" + ref);
				if (rit != mod.resources.end() && !rit->second.filename.empty())
					rule.texture_path = mod.mod_dir / rit->second.filename;
			}

			if (rule.texture_path.empty())
			{
				char name[16];
				std::snprintf(name, sizeof(name), "%08x", rule.hash);
				const std::filesystem::path fallback = mod.mod_dir / (std::string(name) + ".dds");
				std::error_code ec;
				if (std::filesystem::is_regular_file(fallback, ec))
					rule.texture_path = fallback;
			}

			if (rule.texture_path.empty())
				log::warn("%s: rule '%s' has no loadable texture (unresolved binding or no %08x.dds)",
					mod.mod_dir.string().c_str(), rule.section.c_str(), rule.hash);
		}
	}

	// ---------------------------------------------------------------------
	// M9: [SkeletonChain] parsing
	//
	//   [SkeletonChain]
	//   forward_table = Meshes/BlendRemapForward.buf
	//   component = TextureOverrideComponent3, 118, 111, 0, 105
	//               <rule section>, <vg_offset>, <vg_count>[, <remap_id>, <remap_count>]
	// ---------------------------------------------------------------------

	static void parse_skeleton_chain(const std::filesystem::path &ini,
		const std::filesystem::path &mod_dir)
	{
		std::ifstream file(ini, std::ios::binary);
		if (!file.is_open())
			return;
		const std::string text((std::istreambuf_iterator<char>(file)),
			std::istreambuf_iterator<char>());

		IniFile parsed;
		if (!parse_ini(text, parsed))
			return;

		const IniSection *sec = nullptr;
		for (const IniSection &s : parsed.sections)
			if (iequals(s.name, "SkeletonChain"))
			{
				sec = &s;
				break;
			}
		if (sec == nullptr)
			return;

		size_t registered = 0;
		for (const IniEntry &e : sec->entries)
		{
			if (iequals(e.key, "forward_table"))
			{
				if (!g_state.skel.forward.empty())
					continue; // first mod that supplies the table wins
				const std::filesystem::path table = mod_dir / e.value;
				std::ifstream tf(table, std::ios::binary);
				if (!tf.is_open())
				{
					log::warn("skeleton chain: forward table '%s' not found",
						table.string().c_str());
					continue;
				}
				std::vector<uint8_t> raw((std::istreambuf_iterator<char>(tf)),
					std::istreambuf_iterator<char>());
				std::vector<uint16_t> forward(raw.size() / 2);
				if (!forward.empty())
					std::memcpy(forward.data(), raw.data(),
						forward.size() * sizeof(uint16_t));
				if (forward.empty() ||
					forward.size() % SkeletonChain::k_remap_stride != 0)
				{
					log::warn("skeleton chain: forward table '%s' has %llu entries (not a multiple of %u): ignored",
						table.string().c_str(),
						static_cast<unsigned long long>(forward.size()),
						SkeletonChain::k_remap_stride);
					continue;
				}
				g_state.skel.forward = std::move(forward);
				log::info("skeleton chain: forward table '%s' loaded (%llu entries)",
					table.string().c_str(),
					static_cast<unsigned long long>(g_state.skel.forward.size()));
			}
			else if (iequals(e.key, "component"))
			{
				// <rule section>, <vg_offset>, <vg_count>[, <remap_id>, <remap_count>]
				std::vector<std::string> parts;
				std::string item;
				std::stringstream ss(e.value);
				while (std::getline(ss, item, ','))
				{
					// trim
					const auto first = item.find_first_not_of(" \t");
					const auto last = item.find_last_not_of(" \t");
					parts.push_back(first == std::string::npos
						? std::string()
						: item.substr(first, last - first + 1));
				}
				if (parts.size() < 3 || parts[0].empty())
				{
					log::warn("skeleton chain: malformed component '%s'",
						e.value.c_str());
					continue;
				}

				SkeletonChain::Component comp;
				char *end = nullptr;
				comp.vg_offset = std::strtoul(parts[1].c_str(), &end, 10);
				comp.vg_count = std::strtoul(parts[2].c_str(), &end, 10);
				if (parts.size() >= 5)
				{
					comp.remap_id = std::strtol(parts[3].c_str(), &end, 10);
					comp.remap_count = std::strtoul(parts[4].c_str(), &end, 10);
				}
				if (comp.vg_count == 0 ||
					comp.vg_offset + comp.vg_count > SkeletonChain::k_merged_bones ||
					comp.remap_id >= static_cast<int32_t>(SkeletonChain::k_remap_tables))
				{
					log::warn("skeleton chain: component '%s' out of range: ignored",
						e.value.c_str());
					continue;
				}

				g_state.skel.rules[parts[0]] = comp;
				++registered;
			}
		}

		if (registered > 0)
			log::info("skeleton chain: %llu component(s) registered from '%S'",
				static_cast<unsigned long long>(registered), mod_dir.c_str());
	}

	// ---------------------------------------------------------------------
	// Mod manager persistence: one mod folder name per line in
	// <data_root>/mods-disabled.txt. Absent = enabled (new mods load by
	// default); listed = skipped at load_mods() time on the next launch.
	// ---------------------------------------------------------------------

	static std::filesystem::path disabled_mods_path()
	{
		return data_root() / "mods-disabled.txt";
	}

	static std::vector<std::string> read_disabled_mods()
	{
		std::vector<std::string> names;
		std::ifstream f(disabled_mods_path());
		if (!f)
			return names;

		std::string line;
		while (std::getline(f, line))
		{
			while (!line.empty() &&
				(line.back() == '\r' || line.back() == ' ' ||
					line.back() == '\t'))
				line.pop_back();
			if (!line.empty())
				names.push_back(line);
		}
		return names;
	}

	// Rewrites the whole file from the catalog (called under g_state.lock
	// on an overlay toggle; a tiny truncating write is fine there).
	static void write_disabled_mods(
		const std::vector<RuntimeState::ModCatalogEntry> &catalog)
	{
		std::ofstream f(disabled_mods_path(), std::ios::trunc);
		if (!f)
		{
			log::warn("mod manager: cannot write %S",
				disabled_mods_path().c_str());
			return;
		}
		for (const RuntimeState::ModCatalogEntry &e : catalog)
			if (!e.enabled)
				f << e.name << '\n';
	}

	static void load_mods()
	{
		std::vector<TextureOverrideRule> rules;
		std::vector<ShaderOverrideRule> shader_rules;

		const std::filesystem::path mods_dir = data_root() / "Mods";
		std::error_code ec;
		if (!std::filesystem::is_directory(mods_dir, ec))
		{
			log::info("no Mods directory at %S; nothing to load", mods_dir.c_str());
			g_state.mods_root = mods_dir.string();
			return;
		}
		g_state.mods_root = mods_dir.string();
		const std::filesystem::path persist_dir = data_root() / "Persist";
		const std::vector<std::string> disabled = read_disabled_mods();

		size_t mods = 0;
		for (const auto &entry : std::filesystem::directory_iterator(mods_dir, ec))
		{
			if (!entry.is_directory())
				continue;

			// Only directories holding a mod.ini are mods; everything
			// else (READMEs, backups) stays invisible to the manager.
			const std::filesystem::path ini = entry.path() / "mod.ini";
			std::error_code ini_ec;
			if (!std::filesystem::is_regular_file(ini, ini_ec))
				continue;

			const std::string name = entry.path().filename().string();

			RuntimeState::ModCatalogEntry cat;
			cat.name = name;
			cat.enabled = std::find(disabled.begin(), disabled.end(),
				name) == disabled.end();

			if (!cat.enabled)
			{
				log::info("mod '%s' disabled by manager: skipped", name.c_str());
				g_state.mod_catalog.push_back(std::move(cat));
				continue;
			}

			ModRules mod;
			if (!load_mod_rules(ini, mod))
			{
				log::warn("mod '%s': mod.ini failed to load: skipped",
					name.c_str());
				g_state.mod_catalog.push_back(std::move(cat));
				continue;
			}

			cat.loaded = true;
			g_state.mod_catalog.push_back(std::move(cat));
			++mods;
			for (const auto &warning : mod.warnings)
				log::warn("%S: %s", entry.path().c_str(), warning.c_str());

			// M3: script runtime for this mod (vars/keys/resources/lists).
			auto mrt = std::make_unique<RuntimeState::ModRuntime>();
			if (mrt->rt.load(ini))
			{
				for (const std::string &w : mrt->rt.warnings())
					log::warn("%S: %s", entry.path().c_str(), w.c_str());

				std::error_code mkdir_ec;
				std::filesystem::create_directories(persist_dir, mkdir_ec);
				mrt->rt.load_persist(persist_dir / (entry.path().filename().string() + ".ini"));

				log::info("mod '%S': %llu var(s), %llu key(s), script-ready",
					entry.path().c_str(),
					static_cast<unsigned long long>(mrt->rt.var_count()),
					static_cast<unsigned long long>(mrt->rt.keys().size()));

				const uint32_t rt_index = static_cast<uint32_t>(g_state.runtimes.size());
				for (TextureOverrideRule &rule : mod.overrides)
					rule.runtime = rt_index;
				for (ShaderOverrideRule &so : mod.shader_overrides)
					so.runtime = rt_index;
				g_state.runtimes.push_back(std::move(mrt));
			}
			else
			{
				log::warn("%S: script runtime unavailable (rules stay static)",
					entry.path().c_str());
			}

			resolve_rule_textures(mod);

			// M9: [SkeletonChain] -- per-component skeleton merge windows
			// and the Forward remap table (see skeleton_chain.hpp). Parsed
			// from the raw ini (the rule parser has no section for it).
			parse_skeleton_chain(ini, entry.path());

			rules.insert(rules.end(), mod.overrides.begin(), mod.overrides.end());
			shader_rules.insert(shader_rules.end(),
				std::make_move_iterator(mod.shader_overrides.begin()),
				std::make_move_iterator(mod.shader_overrides.end()));
		}

		g_state.has_scripts = !g_state.runtimes.empty();

		// M6 fix: build the pooled-IB window index from the same rule
		// set BEFORE index.build moves it away. Without this call
		// find_by_draw() never matches, region probes never start and
		// pooled-IB model replacement is silently dead (the windows
		// log below is the only visible signal).
		size_t windowed_rules = 0;
		for (const TextureOverrideRule &rule : rules)
			if (rule.has_hash && rule.match_first_index.enabled && rule.match_index_count.enabled)
				++windowed_rules;
		g_state.windows.build(rules);
		g_state.index.build(std::move(rules));

		// M4: ShaderOverride hash index (immutable after this point).
		for (const ShaderOverrideRule &so : shader_rules)
		{
			RuntimeState::ShaderOverrideEntry entry;
			entry.runtime = so.runtime;
			entry.section = so.section;
			g_state.shader_overrides[so.hash] = std::move(entry);
		}
		g_state.shader_override_active.store(!g_state.shader_overrides.empty());

		log::info("loaded %llu mod(s), %llu texture override rule(s), %llu script runtime(s), %llu shader override rule(s)",
			static_cast<unsigned long long>(mods),
			static_cast<unsigned long long>(g_state.index.size()),
			static_cast<unsigned long long>(g_state.runtimes.size()),
			static_cast<unsigned long long>(g_state.shader_overrides.size()));

		// M9: skeleton chain summary (config is immutable from here on).
		if (g_state.skel.active())
		{
			g_state.skel.finalize_config();
			log::info("skeleton chain active: %llu component rule(s)%s",
				static_cast<unsigned long long>(g_state.skel.rules.size()),
				g_state.skel.has_remap() ? ", forward remap table loaded" : "");
		}

		if (!g_state.windows.empty())
			log::info("pooled-IB window matching enabled: %llu rule group(s) (region hashing for pooled index buffers)",
				static_cast<unsigned long long>(g_state.windows.groups().size()));
		else if (windowed_rules > 0)
			log::warn("%llu windowed rule(s) (match_first_index + match_index_count) produced no probe group: "
				"rules without handling = skip cannot join region hashing and will only match learned whole buffers",
				static_cast<unsigned long long>(windowed_rules));

		const std::filesystem::path cache_path = data_root() / "session-cache.json";
		if (g_state.session.load(cache_path))
			log::info("session cache restored: %llu pairing(s)", static_cast<unsigned long long>(g_state.session.size()));
	}

	static void save_session()
	{
		const std::filesystem::path cache_path = data_root() / "session-cache.json";
		if (g_state.session.save(cache_path))
			log::info("session cache saved: %llu pairing(s)", static_cast<unsigned long long>(g_state.session.size()));
	}

	// ---------------------------------------------------------------------
	// 3DMigoto desc/hash helpers
	// ---------------------------------------------------------------------

	// Maps a ReShade texture desc onto the D3D11_TEXTURE2D_DESC layout the
	// 3DMigoto hash pipeline expects. Bind/usage flags are the plausible
	// D3D11 mappings for a gpu-only sampleable texture.
	static Tex2DDesc to_tex2d_desc(const TrackedTexture &t)
	{
		Tex2DDesc desc{};
		desc.width = t.width;
		desc.height = t.height;
		desc.mip_levels = t.levels;
		desc.array_size = t.layers;
		desc.format = t.format;
		desc.sample_count = 1;
		desc.sample_quality = 0;
		desc.usage = 0;             // D3D11_USAGE_DEFAULT
		desc.bind_flags = 0x8;      // D3D11_BIND_SHADER_RESOURCE
		desc.cpu_access_flags = 0;
		desc.misc_flags = 0;
		return desc;
	}

	static void mark_unsupported(uint64_t handle)
	{
		std::lock_guard<std::mutex> guard(g_state.lock);
		g_state.tracker.set_unsupported(handle);
	}

	// Records a learned hash and queues an activation when a mod rule
	// matches. Called WITHOUT the runtime lock (all GPU-side hashing paths
	// run outside it); takes the lock internally.
	static void record_hash(uint64_t handle, const Tex2DDesc &desc, uint32_t data_hash)
	{
		const uint32_t texture_hash = calc_texture2d_desc_hash(data_hash, desc);

		bool matched = false;
		std::string matched_section;
		ActivationReq req;
		{
			std::lock_guard<std::mutex> guard(g_state.lock);
			g_state.tracker.set_hash(handle, data_hash, texture_hash);

			// Learned hash -> mod rule? (direct hit, or via a session pairing)
			uint32_t mod_hash = texture_hash;
			g_state.session.lookup(texture_hash, &mod_hash);

			// HashIndex is immutable after load_mods(), so the rule pointer
			// stays valid; copy what the activation needs.
			const TextureOverrideRule *rule = g_state.index.find(mod_hash);
			if (rule != nullptr)
			{
				matched = true;
				matched_section = rule->section; // log after std::move below
				++g_state.hash_matches;
				if (!rule->texture_path.empty())
				{
					req.resource = handle;
					req.texture_path = rule->texture_path;
					req.section = rule->section;
					g_state.activations.push_back(std::move(req));
				}
			}
		}

		if (matched)
			log::info("hash match: texture %p hash=%08x -> rule '%s'",
				reinterpret_cast<void *>(handle), texture_hash, matched_section.c_str());
	}

	// ---------------------------------------------------------------------
	// Learning path A: initial data (game uploaded the texture at creation)
	// ---------------------------------------------------------------------

	static void hash_from_initial_data(uint64_t handle, const TrackedTexture &snap,
		const reshade::api::subresource_data &initial)
	{
		const Tex2DDesc desc = to_tex2d_desc(snap);

		SurfaceInfo si;
		if (!get_surface_info(snap.width, snap.height, snap.format, &si))
			return;

		// Upload data is tight-packed by convention (matches the layouts
		// 3DMigoto hashed on D3D11). Pitch sanity: at least one tight row.
		if (initial.row_pitch < si.row_bytes)
			return;

		SubresourceData sd;
		sd.sys_mem = initial.data;
		sd.sys_mem_pitch = initial.row_pitch;

		record_hash(handle, desc, calc_texture2d_data_hash(desc, sd));
		++g_state.initialdata_hashes;
	}

	// ---------------------------------------------------------------------
	// Learning path B: GPU readback (runtime::get_texture_data pattern)
	// ---------------------------------------------------------------------

	// Flushes the immediate command list and blocks until the GPU finished
	// it (ReShade screenshot pattern: signal a fence at the queue tail and
	// wait for it, falling back to wait_idle). Without this, mapping the
	// readback resource would race the in-flight copy and hash stale data.
	static void submit_and_wait(reshade::api::command_queue *queue)
	{
		reshade::api::device *const device = queue->get_device();

		reshade::api::fence fence{};
		if (!device->create_fence(0, reshade::api::fence_flags::none, &fence) ||
			!queue->signal(fence, 1) || // signal() flushes the immediate list first
			!device->wait(fence, 1))
			queue->wait_idle();
		if (fence.handle != 0)
			device->destroy_fence(fence);
	}

	static void hash_via_readback(reshade::api::command_queue *queue, const TrackedTexture &snap,
		reshade::api::resource_usage current_state)
	{
		reshade::api::device *const device = queue->get_device();
		const reshade::api::resource src{ snap.handle };

		const Tex2DDesc desc = to_tex2d_desc(snap);
		SurfaceInfo si;
		if (!get_surface_info(snap.width, snap.height, snap.format, &si))
		{
			mark_unsupported(snap.handle);
			return;
		}

		reshade::api::resource intermediate{};
		const reshade::api::resource_desc rb_desc(
			snap.width, snap.height, 1, 1,
			static_cast<reshade::api::format>(snap.format), 1,
			reshade::api::memory_heap::readback,
			reshade::api::resource_usage::copy_dest);

		if (!device->create_resource(rb_desc, nullptr,
			reshade::api::resource_usage::copy_dest, &intermediate))
		{
			log::warn("readback texture creation failed for %p (%ux%u)",
				reinterpret_cast<void *>(snap.handle), snap.width, snap.height);
			mark_unsupported(snap.handle);
			return;
		}

		reshade::api::command_list *const cmd = queue->get_immediate_command_list();
	cmd->barrier(src, current_state, reshade::api::resource_usage::copy_source);
	cmd->copy_texture_region(src, 0, nullptr, intermediate, 0, nullptr);
	cmd->barrier(src, reshade::api::resource_usage::copy_source, current_state);
	submit_and_wait(queue);

		reshade::api::subresource_data mapped{};
		if (device->map_texture_region(intermediate, 0, nullptr,
			reshade::api::map_access::read_only, &mapped))
		{
			// D3D12 readback rows are 256-byte aligned; 3DMigoto hashed the
			// game's tight upload layout. Compact, then hash.
			std::vector<uint8_t> tight(si.num_bytes);
			const size_t n = compact_rows(mapped.data, mapped.row_pitch,
				si.num_rows, si.row_bytes, tight.data(), tight.size());

			if (n == si.num_bytes)
			{
				SubresourceData sd;
				sd.sys_mem = tight.data();
				sd.sys_mem_pitch = static_cast<uint32_t>(si.row_bytes);
				record_hash(snap.handle, desc, calc_texture2d_data_hash(desc, sd));
				++g_state.readback_hashes;
			}
			else
			{
				log::warn("readback compaction failed for %p (%ux%u fmt=%u)",
					reinterpret_cast<void *>(snap.handle), snap.width, snap.height, snap.format);
				mark_unsupported(snap.handle);
			}

			device->unmap_texture_region(intermediate, 0);
		}
		else
		{
			log::warn("readback map failed for %p", reinterpret_cast<void *>(snap.handle));
			mark_unsupported(snap.handle);
		}

		device->destroy_resource(intermediate);
	}

	// ---------------------------------------------------------------------
	// M2: buffer (VB/IB) hash learning
	// ---------------------------------------------------------------------

	static void mark_buffer_unsupported(uint64_t handle)
	{
		std::lock_guard<std::mutex> guard(g_state.lock);
		g_state.buffers.set_unsupported(handle);
	}

	// Records a learned buffer hash and reports rules it now feeds.
	static void record_buffer_hash(uint64_t handle, uint32_t data_hash, uint32_t hash,
		uint64_t byte_width, BufferRole role)
	{
		// Rate-limited learning trace: the first few buffers and then every
		// 512th. Makes hash-formula mismatches visible in ReShade.log (the
		// mod's expected hash vs what the DX12 game actually produced).
		// Large index buffers are the draw-intercept target (mesh mods key
		// on the body IB hash) and rare enough to always log.
		static std::atomic<uint64_t> s_learned{ 0 };
		const uint64_t n = s_learned.fetch_add(1);
		const bool is_large_ib = role == BufferRole::index && byte_width >= 256 * 1024;

		std::string section;
		{
			std::lock_guard<std::mutex> guard(g_state.lock);
			g_state.buffers.set_hash(handle, data_hash, hash);

			std::vector<const TextureOverrideRule *> rules;
			g_state.index.find_draw_rules(hash, rules);
			if (!rules.empty())
				section = rules.front()->section; // diagnostic only
		}

		if (!section.empty())
			log::info("buffer hash match: buffer %p hash=%08x -> rule '%s' (matching draws will be skipped)",
				reinterpret_cast<void *>(handle), hash, section.c_str());
		else if (is_large_ib)
			log::info("buffer hash learned #%llu: %p full=%08x data=%08x (index buffer, %llu bytes)",
				static_cast<unsigned long long>(n), reinterpret_cast<void *>(handle),
				hash, data_hash, static_cast<unsigned long long>(byte_width));
		else if (n < 12 || (n % 512) == 0)
			log::info("buffer hash learned #%llu: %p full=%08x data=%08x (%s, %llu bytes)",
				static_cast<unsigned long long>(n), reinterpret_cast<void *>(handle),
				hash, data_hash,
				role == BufferRole::vertex ? "vb" : role == BufferRole::index ? "ib" : "buf",
				static_cast<unsigned long long>(byte_width));
	}

	// Present-time GPU readback of one tracked VB/IB, hashed with the
	// 3DMigoto buffer formula (data + D3D11_BUFFER_DESC). Buffers are
	// implicitly COMMON state in D3D12, so no barriers are needed to copy
	// out of them.
	static void hash_buffer_via_readback(reshade::api::command_queue *queue, const TrackedBuffer &snap)
	{
		reshade::api::device *const device = queue->get_device();
		const reshade::api::resource src{ snap.handle };

		const reshade::api::resource_desc rb_desc(
			snap.byte_width, reshade::api::memory_heap::readback,
			reshade::api::resource_usage::copy_dest);

		reshade::api::resource intermediate{};
		if (!device->create_resource(rb_desc, nullptr,
			reshade::api::resource_usage::copy_dest, &intermediate))
		{
			log::warn("readback buffer creation failed for %p (%llu bytes)",
				reinterpret_cast<void *>(snap.handle),
				static_cast<unsigned long long>(snap.byte_width));
			mark_buffer_unsupported(snap.handle);
			return;
		}

		reshade::api::command_list *const cmd = queue->get_immediate_command_list();
		cmd->copy_buffer_region(src, 0, intermediate, 0, snap.byte_width);
		submit_and_wait(queue);

		void *data = nullptr;
		if (device->map_buffer_region(intermediate, 0, UINT64_MAX,
			reshade::api::map_access::read_only, &data))
		{
			const uint32_t data_hash = calc_buffer_data_hash(data, snap.byte_width);
			record_buffer_hash(snap.handle, data_hash,
				calc_buffer_hash(data_hash, snap.byte_width, snap.role),
				snap.byte_width, snap.role);
			++g_state.buffer_readback_hashes;
			device->unmap_buffer_region(intermediate);
		}
		else
		{
			log::warn("readback map failed for buffer %p", reinterpret_cast<void *>(snap.handle));
			mark_buffer_unsupported(snap.handle);
		}

		device->destroy_resource(intermediate);
	}

	// ---------------------------------------------------------------------
	// M9: skeleton chain -- present-time capture / merge / upload
	// ---------------------------------------------------------------------

	// Reads 'rows' float4s out of a game constant-buffer ring region.
	// Fast path: persistently-mapped upload rings map directly (the game
	// CPU wrote the frame's constants before recording, so no GPU sync is
	// needed). Slow path: default-heap buffers go through a staging copy
	// and a queue wait. Returns false when the handle is stale or the
	// region runs past the buffer end (caller retries with the next
	// frame's snapshot).
	static bool read_game_cb(reshade::api::device *device,
		reshade::api::command_queue *queue, uint64_t buffer, uint64_t offset,
		float *out, uint32_t rows, bool dump, const char *tag)
	{
		const uint64_t bytes = static_cast<uint64_t>(rows) * 4 * sizeof(float);
		if (buffer == 0)
			return false;

		const reshade::api::resource_desc desc =
			device->get_resource_desc(reshade::api::resource{ buffer });
		if (desc.type != reshade::api::resource_type::buffer ||
			desc.buffer.size < offset + bytes)
		{
			if (dump)
				log::warn("skeleton cb %s: buffer %llx +%llu out of range (%llu bytes)",
					tag, static_cast<unsigned long long>(buffer),
					static_cast<unsigned long long>(offset),
					static_cast<unsigned long long>(desc.buffer.size));
			return false;
		}

		void *ptr = nullptr;
		if (device->map_buffer_region(reshade::api::resource{ buffer }, offset,
			bytes, reshade::api::map_access::read_only, &ptr))
		{
			std::memcpy(out, ptr, static_cast<size_t>(bytes));
			device->unmap_buffer_region(reshade::api::resource{ buffer });
			if (dump)
				log::info("skeleton cb %s: buffer %llx +%llu (%llu bytes) direct-mapped",
					tag, static_cast<unsigned long long>(buffer),
					static_cast<unsigned long long>(offset),
					static_cast<unsigned long long>(desc.buffer.size));
			return true;
		}

		const reshade::api::resource_desc rb_desc(bytes,
			reshade::api::memory_heap::readback,
			reshade::api::resource_usage::copy_dest);
		reshade::api::resource intermediate{};
		if (!device->create_resource(rb_desc, nullptr,
			reshade::api::resource_usage::copy_dest, &intermediate))
			return false;

		reshade::api::command_list *const cmd = queue->get_immediate_command_list();
		cmd->copy_buffer_region(reshade::api::resource{ buffer }, offset,
			intermediate, 0, bytes);
		submit_and_wait(queue);

		bool ok = false;
		void *data = nullptr;
		if (device->map_buffer_region(intermediate, 0, UINT64_MAX,
			reshade::api::map_access::read_only, &data))
		{
			std::memcpy(out, data, static_cast<size_t>(bytes));
			device->unmap_buffer_region(intermediate);
			ok = true;
		}
		device->destroy_resource(intermediate);

		if (dump)
		{
			if (ok)
				log::info("skeleton cb %s: buffer %llx +%llu (%llu bytes) staged copy",
					tag, static_cast<unsigned long long>(buffer),
					static_cast<unsigned long long>(offset),
					static_cast<unsigned long long>(desc.buffer.size));
			else
				log::warn("skeleton cb %s: staged copy map failed", tag);
		}
		return ok;
	}

	// One-time content validation of the captured skeleton CBs: non-zero
	// float4-row density (a character skeleton has hundreds of live bones)
	// and a scan for the 3381.7xx marker family the WWMI stack uses to
	// identify skeleton constant buffers.
	static void dump_skeleton_content(const char *tag, const float *cb)
	{
		uint32_t non_zero_rows = 0;
		int marker = -1;
		for (uint32_t row = 0; row < SkeletonChain::k_game_rows; ++row)
		{
			for (uint32_t e = 0; e < 4; ++e)
			{
				const float v = cb[row * 4 + e];
				if (v == 0.0f)
					continue;
				if (e == 0)
					++non_zero_rows;
				if (marker < 0 && v >= 3381.0f && v <= 3382.0f)
					marker = static_cast<int>(row);
			}
		}
		log::info("skeleton cb %s: %u/%u live float4 rows%s head=[%.3f %.3f %.3f %.3f | %.3f %.3f %.3f %.3f]",
			tag, non_zero_rows, SkeletonChain::k_game_rows,
			marker >= 0 ? ", 3381.x marker present" : "",
			cb[0], cb[1], cb[2], cb[3], cb[4], cb[5], cb[6], cb[7]);
	}

	// Uploads this frame's merged skeletons into the ping-pong upload
	// buffer and publishes the current slot for the draw thread. Root
	// CBV bindings require 256-byte aligned offsets, so every region is
	// padded to a 256-byte multiple.
	static void upload_skeleton(reshade::api::device *device)
	{
		// 8-deep ring: the CPU may run MaxFrameLatency (up to 3) frames
		// ahead of the GPU, and this map-write uploads directly into GPU
		// memory -- reusing a slot the GPU is still reading corrupts the
		// bones of in-flight frames. 8 covers any realistic latency.
		static constexpr uint32_t k_slots = 8;
		static reshade::api::resource slots[k_slots] = {};
		static uint32_t slot = 0;
		static bool s_ready_logged = false;

		// Region layout is immutable once built (config is load-time
		// constant); built lazily on the first upload.
		bool need_regions = false;
		{
			std::lock_guard<std::mutex> guard(g_state.lock);
			need_regions = g_state.skel_regions.total == 0;
		}
		if (need_regions)
		{
			RuntimeState::SkelRegions r{};
			uint64_t off = 0;
			const auto place = [&off](uint64_t bytes, uint64_t *out_off) {
				*out_off = off;
				off += (bytes + 255) / 256 * 256;
			};
			const uint64_t merged_bytes =
				SkeletonChain::bone_rows_bytes(SkeletonChain::k_merged_bones);
			// Remapped regions are full game-CB sized (768 float4 rows):
			// the VS declares Skeleton[768] and root CBVs carry no size,
			// so the whole declared range must be valid, zero-padded
			// past the live bones (matches the DX11 resources, which
			// were full-size copies).
			const uint64_t cb_bytes =
				SkeletonChain::bone_rows_bytes(SkeletonChain::k_game_rows / 3);
			place(merged_bytes, &r.merged_off);
			place(merged_bytes, &r.extra_off);
			for (uint32_t i = 0; i < SkeletonChain::k_remap_tables; ++i)
			{
				r.remap_bytes[i] = g_state.skel.remap_bones(i) != 0 ? cb_bytes : 0;
				place(r.remap_bytes[i], &r.remap_off[i]);
				place(r.remap_bytes[i], &r.extra_remap_off[i]);
			}
			r.total = off;
			std::lock_guard<std::mutex> guard(g_state.lock);
			g_state.skel_regions = r;
		}

		RuntimeState::SkelRegions r{};
		{
			std::lock_guard<std::mutex> guard(g_state.lock);
			r = g_state.skel_regions;
		}
		if (r.total == 0)
			return;

		// Lazy creation of the ping-pong upload buffers.
		{
			OwnCreationGuard guard;
			for (uint32_t i = 0; i < k_slots; ++i)
			{
				if (slots[i].handle != 0)
					continue;
				const reshade::api::resource_desc desc(r.total,
					reshade::api::memory_heap::cpu_to_gpu,
					reshade::api::resource_usage::general);
				if (!device->create_resource(desc, nullptr,
					reshade::api::resource_usage::general, &slots[i]))
				{
					log::warn("skeleton upload buffer creation failed (%llu bytes)",
						static_cast<unsigned long long>(r.total));
					return;
				}
			}
		}

		slot = (slot + 1) % k_slots;
		const reshade::api::resource dst = slots[slot];

		void *ptr = nullptr;
		if (!device->map_buffer_region(dst, 0, UINT64_MAX,
			reshade::api::map_access::write_only, &ptr))
		{
			log::warn("skeleton upload map failed (slot %u)", slot);
			return;
		}

		const uint64_t merged_bytes =
			SkeletonChain::bone_rows_bytes(SkeletonChain::k_merged_bones);
		auto *dst_bytes = static_cast<uint8_t *>(ptr);
		std::memcpy(dst_bytes + r.merged_off, g_state.skel.merged(), merged_bytes);
		std::memcpy(dst_bytes + r.extra_off, g_state.skel.extra(), merged_bytes);
		for (uint32_t i = 0; i < SkeletonChain::k_remap_tables; ++i)
		{
			if (r.remap_bytes[i] == 0)
				continue;
			std::memcpy(dst_bytes + r.remap_off[i],
				g_state.skel.remapped(i), r.remap_bytes[i]);
			std::memcpy(dst_bytes + r.extra_remap_off[i],
				g_state.skel.extra_remapped(i), r.remap_bytes[i]);
		}
		device->unmap_buffer_region(dst);

		{
			std::lock_guard<std::mutex> guard(g_state.lock);
			g_state.skel_gpu_buf = dst.handle;
			g_state.skel_gpu_ready = true;
		}

		if (!s_ready_logged)
		{
			s_ready_logged = true;
			log::info("skeleton upload ready: %u x %llu byte ping-pong buffers, %u main / %u extra bone(s) merged",
				k_slots,
				static_cast<unsigned long long>(r.total),
				g_state.skel.merged_bones_written(),
				g_state.skel.extra_bones_written());
		}
	}

	// Present-time skeleton processing: capture the game's two skeleton
	// CBs (snapshot taken at the last component draw), merge them into
	// the mod's skeleton layout and upload the result -- the NEXT frame's
	// mod draws bind it (1-frame delay, like the WWMI Merged skeleton
	// mode).
	static void process_skeleton(reshade::api::command_queue *queue)
	{
		reshade::api::device *const device = queue->get_device();

		RuntimeState::SkelSnapshot snap;
		bool dump = false;
		{
			std::lock_guard<std::mutex> guard(g_state.lock);
			if (!g_state.skel.active() || !g_state.skel_snap.valid)
				return;
			snap = g_state.skel_snap;
			dump = !g_state.skel_dumped;
		}

		// M9 diagnostics: log each DISTINCT capture source (layout,
		// buffer, offset pairs). Different render passes bind different
		// root params / ring offsets; the LAST component draw of the
		// frame wins the capture, so this log answers "whose skeleton
		// did we actually capture" when components misbehave.
		static uint32_t s_src_logs = 0;
		static uint64_t s_prev_main_buf = 0, s_prev_main_off = 0;
		static uint64_t s_prev_extra_buf = 0, s_prev_extra_off = 0;
		static uint64_t s_prev_layout = 0;
		if (s_src_logs < 30 &&
			(snap.main_buf != s_prev_main_buf || snap.main_off != s_prev_main_off ||
				snap.extra_buf != s_prev_extra_buf || snap.extra_off != s_prev_extra_off ||
				snap.layout != s_prev_layout))
		{
			++s_src_logs;
			log::info("skeleton capture #%u: layout %llx cb4/main=%llx+%llu cb3/extra=%llx+%llu",
				s_src_logs,
				static_cast<unsigned long long>(snap.layout),
				static_cast<unsigned long long>(snap.main_buf),
				static_cast<unsigned long long>(snap.main_off),
				static_cast<unsigned long long>(snap.extra_buf),
				static_cast<unsigned long long>(snap.extra_off));
			s_prev_main_buf = snap.main_buf;
			s_prev_main_off = snap.main_off;
			s_prev_extra_buf = snap.extra_buf;
			s_prev_extra_off = snap.extra_off;
			s_prev_layout = snap.layout;
		}

		std::vector<float> main_cb(static_cast<size_t>(SkeletonChain::k_game_rows) * 4);
		std::vector<float> extra_cb(static_cast<size_t>(SkeletonChain::k_game_rows) * 4);
		if (!read_game_cb(device, queue, snap.main_buf, snap.main_off,
				main_cb.data(), SkeletonChain::k_game_rows, dump, "cb4/main") ||
			!read_game_cb(device, queue, snap.extra_buf, snap.extra_off,
				extra_cb.data(), SkeletonChain::k_game_rows, dump, "cb3/extra"))
			return; // stale handles: the next snapshot retries

		if (dump)
		{
			g_state.skel_dumped = true;
			dump_skeleton_content("cb4/main", main_cb.data());
			dump_skeleton_content("cb3/extra", extra_cb.data());
		}

		g_state.skel.merge(main_cb.data(), extra_cb.data());
		++g_state.skel_captures;

		upload_skeleton(device);
	}

	// M6 diagnostics: on a view's FIRST mismatch, persist the probed
	// region (plus slack up to the pool tail) so the exact 3DMigoto
	// hash hypothesis -- original ByteWidth and D3D11 desc fields such
	// as Usage=IMMUTABLE vs DEFAULT -- can be verified OFFLINE with
	// the dump-probe tool instead of guessing from mismatch lines.
	static constexpr uint64_t k_view_dump_slack = 4ull * 1024 * 1024;
	static std::atomic<uint32_t> g_view_dumps{ 0 };

	static void dump_probe_region(const void *data, uint64_t len,
		const RuntimeState::IndexProbe &p)
	{
		const uint32_t slot = g_view_dumps.fetch_add(1);
		if (slot >= 16)
			return; // budget: the first 16 mismatching views only

		std::error_code ec;
		const std::filesystem::path dir = data_root() / "Dumps";
		std::filesystem::create_directories(dir, ec);

		char name[96];
		sprintf_s(name, "iv_%016llx_off%llu_is%u.bin",
			static_cast<unsigned long long>(p.handle),
			static_cast<unsigned long long>(p.offset), p.index_size);
		const std::filesystem::path path = dir / name;

		std::ofstream f(path, std::ios::binary);
		if (!f)
		{
			log::warn("view dump failed: cannot open %S", path.c_str());
			return;
		}
		f.write(static_cast<const char *>(data),
			static_cast<std::streamsize>(len));

		// Eyeball line: sane index data is smallish u32 values; huge
		// or random-looking words point at a wrong region base.
		const uint32_t *u = static_cast<const uint32_t *>(data);
		const size_t n = std::min<size_t>(static_cast<size_t>(len) / 4, 4096);
		uint32_t mx = 0;
		for (size_t i = 0; i < n; ++i)
			mx = std::max(mx, u[i]);
		log::info("view dump #%u: %S (%llu bytes; span=%u want=%08x) first=%08x %08x %08x %08x max[first %llu u32]=%u",
			slot + 1, path.c_str(),
			static_cast<unsigned long long>(len), p.span_bytes,
			p.group_hash,
			n > 0 ? u[0] : 0, n > 1 ? u[1] : 0,
			n > 2 ? u[2] : 0, n > 3 ? u[3] : 0,
			static_cast<unsigned long long>(n), mx);
	}

	// M6: region readback + hash compare for one pooled-IB probe. The
	// region [view_offset + min_first_index*is, span) inside the pool is
	// exactly the bytes the DX11-era dedicated index buffer contained, so
	// hashing it with desc ByteWidth = span reproduces the mod's hash.
	static void verify_index_view_probe(reshade::api::command_queue *queue,
		const RuntimeState::IndexProbe &p)
	{
		reshade::api::device *const device = queue->get_device();
		const reshade::api::resource src{ p.handle };
		const uint64_t region_offset =
			p.offset + uint64_t(p.min_first_index) * p.index_size;

		// The span comes from mod window rules and is unrelated to the
		// live pool size: a draw-signature collision on a small buffer
		// (or a mesh view near the pool tail) makes the region read
		// past the committed allocation, which faults the GPU and
		// removes the device (DXGI_ERROR_DEVICE_HUNG on scene entry).
		// Validate against the live resource before queueing the copy.
		const reshade::api::resource_desc src_desc =
			device->get_resource_desc(src);
		const bool src_is_buffer =
			src_desc.type == reshade::api::resource_type::buffer;
		const uint64_t src_size =
			src_is_buffer ? src_desc.buffer.size : 0;
		if (!src_is_buffer || region_offset + p.span_bytes > src_size)
		{
			log::info("index view probe out of range: buffer %p off=%llu span=%u size=%llu (rule '%s', attempt %u)",
				reinterpret_cast<void *>(p.handle),
				static_cast<unsigned long long>(p.offset), p.span_bytes,
				static_cast<unsigned long long>(src_size),
				p.section.c_str(), p.retries + 1);
			std::lock_guard<std::mutex> guard(g_state.lock);
			g_state.index_views.set_failed(p.handle, p.offset);
			return;
		}

		// Read span + slack (clamped to the pool tail): the slack feeds
		// the mismatch dump so offline tools can also test
		// ByteWidth-larger-than-span hash hypotheses.
		const uint64_t region_avail = src_size - region_offset;
		const uint64_t read_len = std::min<uint64_t>(region_avail,
			uint64_t(p.span_bytes) + k_view_dump_slack);

		const reshade::api::resource_desc rb_desc(
			read_len, reshade::api::memory_heap::readback,
			reshade::api::resource_usage::copy_dest);

		reshade::api::resource intermediate{};
		if (!device->create_resource(rb_desc, nullptr,
			reshade::api::resource_usage::copy_dest, &intermediate))
		{
			log::warn("view-probe buffer creation failed for %p (%llu bytes)",
				reinterpret_cast<void *>(p.handle),
				static_cast<unsigned long long>(read_len));
			std::lock_guard<std::mutex> guard(g_state.lock);
			g_state.index_views.set_failed(p.handle, p.offset);
			return;
		}

		reshade::api::command_list *const cmd = queue->get_immediate_command_list();
		cmd->copy_buffer_region(src, region_offset, intermediate, 0, read_len);
		submit_and_wait(queue);

		uint32_t full_hash = 0;
		bool mapped = false;
		void *data = nullptr;
		if (device->map_buffer_region(intermediate, 0, UINT64_MAX,
			reshade::api::map_access::read_only, &data))
		{
			const uint32_t data_hash = calc_buffer_data_hash(data, p.span_bytes);
			full_hash = calc_buffer_hash(data_hash, p.span_bytes, BufferRole::index);
			mapped = true;
			if (full_hash != p.group_hash && p.retries == 0)
				dump_probe_region(data, read_len, p);
			device->unmap_buffer_region(intermediate);
		}
		device->destroy_resource(intermediate);

		if (!mapped)
		{
			log::warn("view-probe map failed for buffer %p off=%llu",
				reinterpret_cast<void *>(p.handle),
				static_cast<unsigned long long>(p.offset));
			std::lock_guard<std::mutex> guard(g_state.lock);
			g_state.index_views.set_failed(p.handle, p.offset);
			return;
		}

		if (full_hash == p.group_hash)
		{
			{
				std::lock_guard<std::mutex> guard(g_state.lock);
				g_state.index_views.set_verified(p.handle, p.offset, full_hash);
			}
			++g_state.views_verified;
			log::info("index view verified: buffer %p off=%llu span=%u -> hash %08x (rule group '%s'; draws route through it now)",
				reinterpret_cast<void *>(p.handle),
				static_cast<unsigned long long>(p.offset),
				p.span_bytes, full_hash, p.section.c_str());
		}
		else
		{
			{
				std::lock_guard<std::mutex> guard(g_state.lock);
				g_state.index_views.set_failed(p.handle, p.offset);
			}
			// Retry noise control: full detail on the first attempts.
			if (p.retries < 3)
				log::info("index view mismatch: buffer %p off=%llu is=%u first=%u span=%u got=%08x want=%08x (rule '%s', attempt %u)",
					reinterpret_cast<void *>(p.handle),
					static_cast<unsigned long long>(p.offset),
					p.index_size, p.min_first_index, p.span_bytes,
					full_hash, p.group_hash, p.section.c_str(), p.retries + 1);
		}
	}

	// ---------------------------------------------------------------------
	// M3: GpuBridge over ReShade (script runtime -> device/command list)
	// ---------------------------------------------------------------------

	// Records buffer binds/draws/copies into the game's CURRENT command
	// list recording (draw context) or the immediate command list
	// (present context, cmd == nullptr: binds/draws no-op there).
	// create/destroy run under OwnCreationGuard so our own resources stay
	// invisible to the tracking events. No method takes <lock>.
	// ---------------------------------------------------------------------
	// M4: blend override plumbing
	// ---------------------------------------------------------------------

	// Neutral blend factor code -> reshade::api::blend_factor. Codes
	// 0..11 are identical; the neutral ordering inserts no codes for
	// constant_alpha / one_minus_constant_alpha (12/13 in the API), so
	// everything >= 12 shifts by +2 (src_alpha_sat 12->14, src1_color
	// 13->15, ...). blend_op needs no mapping (0..4 identical).
	static reshade::api::blend_factor to_api_blend_factor(uint8_t neutral)
	{
		return static_cast<reshade::api::blend_factor>(neutral <= 11 ? neutral : neutral + 2);
	}

	// Cache key for one (source pipeline, blend config) clone. Factors
	// are excluded: they ride OMSetBlendFactor, not the PSO.
	static uint64_t blend_clone_key(uint64_t src, const BlendConfig &cfg)
	{
		uint64_t k = src * 0x9E3779B97F4A7C15ull;
		if (cfg.has_color)
			k ^= 0x100000000ull | (uint64_t(cfg.op) << 16) |
				(uint64_t(cfg.src) << 8) | uint64_t(cfg.dst);
		if (cfg.has_alpha)
			k ^= 0x200000000ull | (uint64_t(cfg.alpha_op) << 16) |
				(uint64_t(cfg.alpha_src) << 8) | uint64_t(cfg.alpha_dst);
		// final mix (splitmix64-style) to spread the low bits
		k ^= k >> 30; k *= 0xBF5846D80CE4D5B5ull;
		k ^= k >> 27; k *= 0x94D049BB133111EBull;
		return k ^ (k >> 31);
	}

	// Rewrites the blend portion of a cloned blend_desc. 3DMigoto
	// semantics: 'blend'/'blendalpha' imply BlendEnable for every render
	// target; everything else (write mask, logic op, alpha-to-coverage)
	// stays as the game authored it.
	static void patch_blend_desc(reshade::api::blend_desc &bd, const BlendConfig &cfg)
	{
		for (uint32_t rt = 0; rt < 8; ++rt)
		{
			if (cfg.has_color)
			{
				bd.blend_enable[rt] = true;
				bd.source_color_blend_factor[rt] = to_api_blend_factor(cfg.src);
				bd.dest_color_blend_factor[rt] = to_api_blend_factor(cfg.dst);
				bd.color_blend_op[rt] = static_cast<reshade::api::blend_op>(cfg.op);
			}
			if (cfg.has_alpha)
			{
				bd.blend_enable[rt] = true;
				bd.source_alpha_blend_factor[rt] = to_api_blend_factor(cfg.alpha_src);
				bd.dest_alpha_blend_factor[rt] = to_api_blend_factor(cfg.alpha_dst);
				bd.alpha_blend_op[rt] = static_cast<reshade::api::blend_op>(cfg.alpha_op);
			}
		}
	}

	// Direct OMSetBlendFactor on the native command list (float-exact).
	static void set_blend_factor(reshade::api::command_list *cmd, const float factors[4])
	{
		auto *d3d = reinterpret_cast<ID3D12GraphicsCommandList *>(
			static_cast<uintptr_t>(cmd->get_native()));
		if (d3d != nullptr)
			d3d->OMSetBlendFactor(factors);
	}

	class AddonBridge final : public GpuBridge
	{
	public:
		AddonBridge(reshade::api::device *device, reshade::api::command_list *cmd)
			: _device(device), _cmd(cmd) {}

		uint64_t create_buffer(const void *data, uint64_t size,
			bool want_uav, bool /*want_sr*/) override
		{
			OwnCreationGuard guard;

			const reshade::api::resource_usage usage = want_uav
				? (reshade::api::resource_usage::unordered_access |
					reshade::api::resource_usage::general)
				: reshade::api::resource_usage::general;
			const reshade::api::resource_desc desc(size,
				reshade::api::memory_heap::gpu_only, usage);

			reshade::api::subresource_data init{};
			reshade::api::subresource_data *init_ptr = nullptr;
			if (data != nullptr)
			{
				init.data = const_cast<void *>(data);
				init_ptr = &init;
			}

			reshade::api::resource res{};
			if (!_device->create_resource(desc, init_ptr,
				reshade::api::resource_usage::general, &res))
				return 0;
			return res.handle;
		}

		void destroy_buffer(uint64_t handle) override
		{
			OwnCreationGuard guard;
			_device->destroy_resource(reshade::api::resource{ handle });
		}

		bool buffer_size(uint64_t handle, uint64_t *size) override
		{
			const reshade::api::resource_desc desc =
				_device->get_resource_desc(reshade::api::resource{ handle });
			if (desc.type != reshade::api::resource_type::buffer)
				return false;
			*size = desc.buffer.size;
			return true;
		}

		void bind_vb(uint32_t slot, uint64_t resource, uint64_t offset, uint32_t stride) override
		{
			if (_cmd == nullptr)
				return;
			_cmd->bind_vertex_buffer(slot, reshade::api::resource{ resource }, offset, stride);
		}

		void bind_ib(uint64_t resource, uint64_t offset, uint32_t index_size) override
		{
			if (_cmd == nullptr)
				return;
			_cmd->bind_index_buffer(reshade::api::resource{ resource }, offset, index_size);
		}

		void draw_indexed(uint32_t index_count, uint32_t first_index,
			int32_t base_vertex, uint32_t instance_count) override
		{
			if (_cmd == nullptr)
				return;
			// M9: the skeleton CBV replacement applies lazily at the first
			// re-issued draw -- a body that never skips (mod disabled)
			// leaves the game's bindings untouched.
			apply_skeleton_pushes();
			_cmd->draw_indexed(index_count, instance_count, first_index, base_vertex, 0);
			++g_state.script_draws_issued;
		}

		void copy_buffer(uint64_t src, uint64_t src_offset,
			uint64_t dst, uint64_t dst_offset, uint64_t size) override
		{
			if (_cmd == nullptr || size == 0)
				return;
			_cmd->copy_buffer_region(reshade::api::resource{ src }, src_offset,
				reshade::api::resource{ dst }, dst_offset, size);
		}

		// ---- M9: skeleton CBV replacement ----
		// Armed by evaluate_draw before the rule body runs; the pushes are
		// applied lazily at the first re-issued draw (skip-gated) and the
		// game's original bindings are pushed back afterwards. All pushes
		// happen inside the ScriptExecGuard window, so they do not pollute
		// the root-state tracking.
		void arm_skeleton(uint64_t layout, const SkelParamRef &cb3,
			const RuntimeState::RootCbvBinding &orig_cb3, const SkelParamRef &cb4,
			const RuntimeState::RootCbvBinding &orig_cb4, uint64_t gpu_buffer,
			uint64_t main_off, uint64_t main_size,
			uint64_t extra_off, uint64_t extra_size)
		{
			_skel_armed = true;
			_skel_applied = false;
			_skel_layout = layout;
			_skel_cb3 = cb3;
			_skel_cb4 = cb4;
			_skel_orig_cb3 = orig_cb3;
			_skel_orig_cb4 = orig_cb4;
			_skel_gpu_buffer = gpu_buffer;
			_skel_main_off = main_off;
			_skel_main_size = main_size;
			_skel_extra_off = extra_off;
			_skel_extra_size = extra_size;
		}

		bool skeleton_applied() const { return _skel_applied; }

		void restore_skeleton()
		{
			if (!_skel_applied || _cmd == nullptr)
			{
				_skel_armed = false;
				return;
			}
			push_cbv(_cmd, _skel_layout, _skel_cb4,
				_skel_orig_cb4.buffer, _skel_orig_cb4.offset, _skel_orig_cb4.size);
			push_cbv(_cmd, _skel_layout, _skel_cb3,
				_skel_orig_cb3.buffer, _skel_orig_cb3.offset, _skel_orig_cb3.size);
			_skel_applied = false;
			_skel_armed = false;
		}

	private:
		void apply_skeleton_pushes()
		{
			if (!_skel_armed || _skel_applied || _cmd == nullptr)
				return;
			_skel_applied = true;
			// cb4 (main skeleton) <- merged, cb3 (extra) <- extra merged.
			push_cbv(_cmd, _skel_layout, _skel_cb4,
				_skel_gpu_buffer, _skel_main_off, _skel_main_size);
			push_cbv(_cmd, _skel_layout, _skel_cb3,
				_skel_gpu_buffer, _skel_extra_off, _skel_extra_size);
			++g_state.skel_pushes;
		}

		static void push_cbv(reshade::api::command_list *cmd, uint64_t layout,
			const SkelParamRef &slot, uint64_t buffer, uint64_t offset, uint64_t size)
		{
			if (buffer == 0 || !slot.found())
				return;
			reshade::api::buffer_range range{};
			range.buffer = reshade::api::resource{ buffer };
			range.offset = offset;
			range.size = size;
			reshade::api::descriptor_table_update update{};
			update.type = reshade::api::descriptor_type::constant_buffer;
			update.binding = slot.binding;
			update.count = 1;
			update.descriptors = &range;
			cmd->push_descriptors(reshade::api::shader_stage::all,
				reshade::api::pipeline_layout{ layout }, slot.param, update);
		}

	public:

		// M4: applies the script's merged BlendConfig for the draw that
		// follows. Returns a revoke token (the bound source pipeline) or
		// 0 when nothing could be applied. Blend op/src/dst live in the
		// PSO -> clone + rebind; factors ride OMSetBlendFactor directly.
		uint64_t apply_blend(const BlendConfig &cfg) override
		{
			if (_cmd == nullptr)
				return 0;

			uint64_t src = 0;
			{
				std::lock_guard<std::mutex> guard(g_state.lock);
				const auto it = g_state.cl_pipeline.find(_cmd->get_native());
				if (it != g_state.cl_pipeline.end())
					src = it->second;
			}
			if (src == 0)
				return 0; // no tracked graphics pipeline on this list

			// Factors first (they survive the PSO rebind in revoke).
			if (cfg.has_factors)
			{
				// Partial 'blend_factor[N]' updates: unlisted components
				// fall back to the D3D12 default (1,1,1,1) -- the current
				// OM factor is not queryable in D3D12.
				float f[4] = { 1.0f, 1.0f, 1.0f, 1.0f };
				for (uint32_t n = 0; n < 4; ++n)
					if (cfg.factor_mask & (1u << n))
						f[n] = cfg.factors[n];
				set_blend_factor(_cmd, f);
			}

			if (!cfg.has_color && !cfg.has_alpha)
				return src; // factor-only override: the PSO stays bound

			// Cached clone?
			const uint64_t key = blend_clone_key(src, cfg);
			reshade::api::pipeline cloned{};
			{
				std::lock_guard<std::mutex> guard(g_state.blend_lock);
				const auto it = g_state.blend_clones.find(key);
				if (it != g_state.blend_clones.end())
					cloned = it->second.pipe;
			}

			if (cloned.handle == 0)
			{
				const std::vector<pipeline_subobject> *source =
					g_state.pipelines.clone_source(src);
				if (source == nullptr)
					return 0; // cache miss + evicted subobjects: give up

				// Shallow copy is enough: the tracker owns the deep
				// storage; only the blend desc is replaced locally.
				std::vector<pipeline_subobject> tmp(source->begin(), source->end());
				reshade::api::blend_desc patched{};
				bool has_blend = false;
				for (pipeline_subobject &s : tmp)
				{
					if (s.type == pipeline_subobject_type::blend_state && s.data != nullptr)
					{
						patched = *static_cast<const reshade::api::blend_desc *>(s.data);
						patch_blend_desc(patched, cfg);
						s.data = &patched;
						has_blend = true;
						break;
					}
				}
				if (!has_blend)
					return 0;

				{
					OwnCreationGuard guard;
					if (!_device->create_pipeline(
						reshade::api::pipeline_layout{ g_state.pipelines.clone_source_layout(src) },
						static_cast<uint32_t>(tmp.size()), tmp.data(), &cloned) || cloned.handle == 0)
						return 0;
				}

				{
					std::lock_guard<std::mutex> guard(g_state.blend_lock);
					g_state.blend_clones[key] = { src, cloned };
				}
				++g_state.blend_pso_clones;
			}

			// t_script_exec is set here: our own bind_pipeline replay must
			// not overwrite the tracked source pipeline of this list.
			_cmd->bind_pipeline(reshade::api::pipeline_stage::all_graphics, cloned);
			++g_state.blend_draws;
			return src;
		}

		// Restores the pipeline recorded in the token (blend op/src/dst
		// live inside the PSO, so rebinding the original undoes them).
		// The blend factor is reset to the D3D12 default; D3D12 offers no
		// way to query the pre-override value.
		void revoke_blend(uint64_t token) override
		{
			if (_cmd == nullptr || token == 0)
				return;
			_cmd->bind_pipeline(reshade::api::pipeline_stage::all_graphics,
				reshade::api::pipeline{ token });
			const float defaults[4] = { 1.0f, 1.0f, 1.0f, 1.0f };
			set_blend_factor(_cmd, defaults);
		}

	private:
		reshade::api::device *_device;
		reshade::api::command_list *_cmd;

		// M9 skeleton replacement state (see arm_skeleton).
		bool _skel_armed = false;
		bool _skel_applied = false;
		uint64_t _skel_layout = 0;
		SkelParamRef _skel_cb3{};
		SkelParamRef _skel_cb4{};
		RuntimeState::RootCbvBinding _skel_orig_cb3{};
		RuntimeState::RootCbvBinding _skel_orig_cb4{};
		uint64_t _skel_gpu_buffer = 0;
		uint64_t _skel_main_off = 0, _skel_main_size = 0;
		uint64_t _skel_extra_off = 0, _skel_extra_size = 0;
	};

	// Re-applies the pre-script IA bindings to the recording. The game's
	// own RHI state cache does not know the script changed anything, so
	// without this restore it would skip re-binding its buffers for the
	// draws that follow and render with the mod's geometry.
	static void restore_ia(reshade::api::command_list *cmd, const IaState &ia)
	{
		for (uint32_t i = 0; i < IaState::max_vb_slots; ++i)
			if (ia.vb_valid_mask & (1u << i))
				cmd->bind_vertex_buffer(i, reshade::api::resource{ ia.vbs[i].buffer },
					ia.vbs[i].offset, ia.vbs[i].stride);
		if (ia.ib_buffer != 0)
			cmd->bind_index_buffer(reshade::api::resource{ ia.ib_buffer },
				ia.ib_offset, ia.ib_index_size);
	}

	// ---------------------------------------------------------------------
	// M2/M3: draw-time skip decision + script execution
	// ---------------------------------------------------------------------

	// M6: schedule a region probe for the pooled-IB view this draw uses
	// when the draw signature matches a mod window group. Caller must
	// hold g_state.lock. Cheap on the hot path: one linear scan over
	// the window groups (mesh mods use a handful) and a hash lookup.
	//
	// Diagnostics: a window hit also marks the view 'interesting' so
	// the draw-indexed hook tallies EVERY subsequent signature on it --
	// the real section layout, which decides whether byte-hash probes
	// are even the right verification (UE DX12 re-indexes meshes).
	static void schedule_index_view_probe(const IaState &ia, const DrawCallInfo &call)
	{
		if (ia.ib_buffer == 0 || ia.ib_index_size == 0)
			return;

		// Diagnostics: tally EVERY indexed draw signature on an
		// already-interesting view (window hits and misses alike) --
		// the view's real section layout.
		if (TrackedIndexView *tv = g_state.index_views.find(ia.ib_buffer, ia.ib_offset))
			if (tv->interesting)
				tv->tally(call.first_index, call.index_count);

		const DrawRuleGroup *grp =
			g_state.windows.find_by_draw(call.first_index, call.index_count);
		if (grp == nullptr)
			return;

		TrackedIndexView *v = g_state.index_views.track(
			ia.ib_buffer, ia.ib_offset, ia.ib_index_size, g_state.frame);
		if (v == nullptr)
			return;
		if (!v->interesting)
		{
			// First window hit on this view: start the tally.
			v->interesting = true;
			v->tally(call.first_index, call.index_count);
		}

		// M7: signature-coverage verification -- once every window of
		// the group has been drawn on this view, the layout proves the
		// view IS the mod's mesh pool region (collision-proof for
		// multi-window tilings). This bypasses the byte-hash probe,
		// which can never succeed on UE DX12 (vertex re-indexing).
		if (v->state != TrackedIndexView::State::verified &&
			signature_coverage_complete(*v, *grp))
		{
			g_state.index_views.set_verified(
				ia.ib_buffer, ia.ib_offset, grp->hash);
			log::info("index view verified by signature coverage: buffer %p off=%llu is=%u -> hash %08x (%u window(s) drawn; byte-hash probe bypassed)",
				reinterpret_cast<void *>(ia.ib_buffer),
				static_cast<unsigned long long>(ia.ib_offset),
				ia.ib_index_size, grp->hash,
				static_cast<unsigned>(grp->windows.size()));
			return;
		}

		if (!g_state.index_views.probe_due(*v, g_state.frame))
			return;

		const uint64_t span_bytes =
			uint64_t(grp->max_end_index - grp->min_first_index) * ia.ib_index_size;
		if (span_bytes == 0 || span_bytes > 64ull * 1024 * 1024)
			return;

		// Early bounds filter against the bind-time pool size: a view
		// whose window cannot fit is structurally unprovable, so mark
		// it failed (retry cooldown applies) instead of ferrying the
		// probe to the present thread. verify re-checks the live
		// resource as the authoritative guard.
		if (const TrackedBuffer *pool = g_state.buffers.find(ia.ib_buffer))
		{
			const uint64_t end = ia.ib_offset +
				uint64_t(grp->max_end_index) * ia.ib_index_size;
			if (end > pool->byte_width)
			{
				g_state.index_views.set_failed(ia.ib_buffer, ia.ib_offset);
				v->probe_frame = g_state.frame; // start the cooldown now
				return;
			}
		}

		v->state = TrackedIndexView::State::probing;
		v->probe_frame = g_state.frame;
		v->expected_hash = grp->hash;

		RuntimeState::IndexProbe probe;
		probe.handle = ia.ib_buffer;
		probe.offset = ia.ib_offset;
		probe.index_size = ia.ib_index_size;
		probe.group_hash = grp->hash;
		probe.min_first_index = grp->min_first_index;
		probe.span_bytes = static_cast<uint32_t>(span_bytes);
		probe.retries = v->retries;
		probe.section = grp->section;
		g_state.view_probes.push_back(std::move(probe));
		++g_state.view_probes_issued;
	}

	// M8 diagnostics: short readable name for the vertex formats the
	// game's input layouts actually use (unknown values fall back to
	// the numeric enum so nothing is silently hidden).
	static const char *vertex_format_name(reshade::api::format f)
	{
		using reshade::api::format;
		switch (f)
		{
		case format::r32g32b32a32_float: return "r32g32b32a32f";
		case format::r32g32b32_float: return "r32g32b32f";
		case format::r32g32_float: return "r32g32f";
		case format::r32g32_uint: return "r32g32u";
		case format::r32_uint: return "r32u";
		case format::r16g16b16a16_float: return "r16g16b16a16f";
		case format::r16g16b16a16_snorm: return "r16g16b16a16sn";
		case format::r16g16_float: return "r16g16f";
		case format::r16g16_uint: return "r16g16u";
		case format::r16g16_snorm: return "r16g16sn";
		case format::r16_uint: return "r16u";
		case format::r11g11b10_float: return "r11g11b10f";
		case format::r10g10b10a2_unorm: return "r10g10b10a2un";
		case format::r8g8b8a8_unorm: return "r8g8b8a8un";
		case format::r8g8b8a8_uint: return "r8g8b8a8u";
		case format::r8g8b8a8_snorm: return "r8g8b8a8sn";
		case format::r8_uint: return "r8u";
		default: return nullptr;
		}
	}

	// ---------------------------------------------------------------------
	// M9: skeleton chain -- draw-time snapshot + bridge arming
	// ---------------------------------------------------------------------

	// Finds the root parameter of a pipeline layout that carries the
	// vertex-stage constant buffer bound at 'register_index' in register
	// space 0 -- the DX12 mapping of the DX11 mod's vs-cb3/vs-cb4
	// replacement. Push parameters with multiple ranges are scanned in
	// full (the register can live beyond ranges[0]); the matching range's
	// binding index is reported for the push_descriptors update.
	// UINT32_MAX param = layout has no such push param (the skeleton may
	// then live in descriptor tables, which this implementation does not
	// intercept).
	static SkelParamRef find_push_cbv_param(
		const std::vector<reshade::api::pipeline_layout_param> &params,
		uint32_t register_index)
	{
		SkelParamRef out{};
		for (uint32_t i = 0; i < params.size(); ++i)
		{
			const reshade::api::pipeline_layout_param &p = params[i];
			const reshade::api::descriptor_range *ranges = nullptr;
			uint32_t count = 1;
			switch (p.type)
			{
			case reshade::api::pipeline_layout_param_type::push_descriptors:
				ranges = &p.push_descriptors;
				count = 1;
				break;
			case reshade::api::pipeline_layout_param_type::push_descriptors_with_ranges:
			case reshade::api::pipeline_layout_param_type::push_descriptors_with_ranges_and_flags:
				ranges = p.descriptor_table_with_flags.count != 0
					? p.descriptor_table_with_flags.ranges
					: nullptr;
				count = p.descriptor_table_with_flags.count;
				break;
			default:
				continue;
			}
			if (ranges == nullptr)
				continue;

			for (uint32_t e = 0; e < count; ++e)
			{
				const reshade::api::descriptor_range &r = ranges[e];
				if (r.type != reshade::api::descriptor_type::constant_buffer)
					continue;
				if (r.dx_register_space != 0)
					continue;
				if (r.dx_register_index != register_index)
					continue;
				if ((static_cast<uint32_t>(r.visibility) &
						static_cast<uint32_t>(reshade::api::shader_stage::vertex)) == 0)
					continue;
				out.param = i;
				out.binding = r.binding;
				return out;
			}
		}
		return out; // not found
	}

	// Snapshots the game's skeleton CBVs (root params bound at the
	// vs-cb3/vs-cb4 registers of this draw's layout) for the present-time
	// capture, and arms the bridge's lazy CBV replacement when the merged
	// skeleton is already live. No-op unless the rule declared a
	// [SkeletonChain] component.
	static void prepare_skeleton(reshade::api::command_list *cmd_list,
		const TextureOverrideRule &rule, AddonBridge *bridge)
	{
		if (!g_state.skel.active())
			return;

		uint64_t layout = 0;
		SkelParamRef cb3{}, cb4{};
		RuntimeState::RootCbvBinding orig3{}, orig4{};
		SkeletonChain::Component comp{};
		uint64_t gpu_buf = 0;
		RuntimeState::SkelRegions regions{};
		{
			std::lock_guard<std::mutex> guard(g_state.lock);

			const auto cit = g_state.skel.rules.find(rule.section);
			if (cit == g_state.skel.rules.end())
				return;
			comp = cit->second;

			const auto it = g_state.cl_root.find(cmd_list->get_native());
			if (it == g_state.cl_root.end())
				return;
			layout = it->second.layout;
			if (layout == 0)
				return;

			// Identify the root params bound at vs-cb3/vs-cb4 (cached per
			// layout; the negative result stays cached so an unmatched
			// layout logs once instead of every draw).
			auto idit = g_state.skel_layout_params.find(layout);
			if (idit == g_state.skel_layout_params.end())
			{
				const auto lit = g_state.layouts.find(layout);
				if (lit == g_state.layouts.end())
					return; // layout destroyed before the draw: skip quietly
				cb3 = find_push_cbv_param(lit->second, 3);
				cb4 = find_push_cbv_param(lit->second, 4);
				idit = g_state.skel_layout_params.emplace(layout,
					std::make_pair(cb3, cb4)).first;
				log::info("skeleton chain: layout %llx -> cb3=param%u.%u cb4=param%u.%u%s",
					static_cast<unsigned long long>(layout),
					cb3.param, cb3.binding, cb4.param, cb4.binding,
					(!cb3.found() || !cb4.found())
						? " (missing push CBV at register 3/4: skeleton CBs may live in descriptor tables)"
						: "");
			}
			cb3 = idit->second.first;
			cb4 = idit->second.second;
			if (!cb3.found() || !cb4.found())
				return;

			const auto b3 = it->second.cbvs.find(
				(static_cast<uint64_t>(cb3.param) << 32) | cb3.binding);
			const auto b4 = it->second.cbvs.find(
				(static_cast<uint64_t>(cb4.param) << 32) | cb4.binding);
			if (b3 == it->second.cbvs.end() || b4 == it->second.cbvs.end())
				return; // params exist but not bound yet on this list
			orig3 = b3->second;
			orig4 = b4->second;

			// The latest component draw of the frame wins: passes record
			// the same character with equal skeleton content.
			g_state.skel_snap.layout = layout;
			g_state.skel_snap.cb3_param = cb3.param;
			g_state.skel_snap.cb3_binding = cb3.binding;
			g_state.skel_snap.cb4_param = cb4.param;
			g_state.skel_snap.cb4_binding = cb4.binding;
			g_state.skel_snap.main_buf = orig4.buffer;
			g_state.skel_snap.main_off = orig4.offset;
			g_state.skel_snap.extra_buf = orig3.buffer;
			g_state.skel_snap.extra_off = orig3.offset;
			g_state.skel_snap.frame = g_state.frame;
			g_state.skel_snap.valid = true;

			gpu_buf = g_state.skel_gpu_ready ? g_state.skel_gpu_buf : 0;
			regions = g_state.skel_regions;
		}

		if (bridge == nullptr || gpu_buf == 0)
			return;

		// Regions this component binds: the per-component remapped
		// skeleton (comp3-6) or the full merged skeleton (comp0-2/7).
		uint64_t main_off = regions.merged_off;
		uint64_t main_size = SkeletonChain::bone_rows_bytes(SkeletonChain::k_merged_bones);
		uint64_t extra_off = regions.extra_off;
		uint64_t extra_size = main_size;
		bool remapped = false;
		if (comp.remap_id >= 0 && comp.remap_id < static_cast<int32_t>(SkeletonChain::k_remap_tables) &&
			regions.remap_bytes[comp.remap_id] != 0)
		{
			main_off = regions.remap_off[comp.remap_id];
			main_size = regions.remap_bytes[comp.remap_id];
			extra_off = regions.extra_remap_off[comp.remap_id];
			extra_size = regions.remap_bytes[comp.remap_id];
			remapped = true;
		}

		// One-time per component: the exact CBV region the draws bind.
		{
			static std::unordered_set<std::string> s_logged;
			if (s_logged.insert(rule.section).second)
				log::info("skeleton arm '%s': layout %llx cb4=param%u.%u ->%s%llu +%llu, cb3=param%u.%u ->%s%llu +%llu (%u bone(s), remap_id=%d)",
					rule.section.c_str(),
					static_cast<unsigned long long>(layout),
					cb4.param, cb4.binding,
					remapped ? "remap" : "merged",
					static_cast<unsigned long long>(main_off),
					static_cast<unsigned long long>(main_size),
					cb3.param, cb3.binding,
					remapped ? "remap" : "merged",
					static_cast<unsigned long long>(extra_off),
					static_cast<unsigned long long>(extra_size),
					comp.vg_count, comp.remap_id);
		}

		bridge->arm_skeleton(layout, cb3, orig3, cb4, orig4,
			gpu_buf, main_off, main_size, extra_off, extra_size);
	}

	static bool evaluate_draw(reshade::api::command_list *cmd_list,
		const DrawCallInfo &call, bool indexed)
	{
		const TextureOverrideRule *rule = nullptr;
		RuntimeState::ModRuntime *mrt = nullptr;
		IaState ia{};
		bool have_ia = false;
		bool first_skip = false;
		{
			std::lock_guard<std::mutex> guard(g_state.lock);

			const auto it = g_state.draw_states.find(cmd_list->get_native());
			if (it == g_state.draw_states.end())
				return false; // no IA bindings seen for this recording
			ia = it->second;
			have_ia = true;

			rule = find_skip_rule(g_state.index, g_state.buffers,
				&g_state.index_views, it->second, call, indexed);
			if (rule != nullptr)
			{
				// Skip stats are advisory; keys are bounded by the draw
				// rule count of the loaded mods.
				RuntimeState::SkipRecord &rec = g_state.skip_log[rule->section];
				first_skip = (rec.count == 0);
				++rec.count;
				rec.frame = g_state.frame;

				if (rule->runtime < g_state.runtimes.size())
					mrt = g_state.runtimes[rule->runtime].get();
			}
			else if (indexed)
			{
				// M6: no rule matched -- if this draw's signature hits a
				// mod window, probe the pool view so later draws from it
				// can be routed (see verify_index_view_probe).
				schedule_index_view_probe(it->second, call);
			}
		}

		if (rule == nullptr)
			return false;

		// M7 diagnostics: the FIRST skip of a rule dumps the game's
		// real IA layout next to the mod's assumed strides
		// (vb0..vb4 = 12/8/16/4/16). A mismatch means the GPU fetches
		// vertices with the wrong layout (deformation, spring wobble)
		// and can read past the mod buffers' ends (DEVICE_HUNG). Also
		// shows the game's own draw parameters (base vertex, instance
		// count) the re-issued draws must be sane against.
		if (first_skip && have_ia)
		{
			char slots[320];
			size_t used = 0;
			slots[0] = '\0';
			for (uint32_t i = 0; i < IaState::max_vb_slots &&
				used + 48 < sizeof(slots); ++i)
				if (ia.vb_valid_mask & (1u << i))
					used += static_cast<size_t>(sprintf_s(
						slots + used, sizeof(slots) - used,
						"[%u]=%llx+%llx/%u ", i,
						static_cast<unsigned long long>(ia.vbs[i].buffer),
						static_cast<unsigned long long>(ia.vbs[i].offset),
						ia.vbs[i].stride));
			log::info("first skip '%s': draw idx=(%u,%u) inst=%u base=%d ib=%llx+%llu is=%u vb slots (handle+offset/stride): %s",
				rule->section.c_str(), call.first_index, call.index_count,
				call.instance_count,
				static_cast<int32_t>(call.first_vertex),
				static_cast<unsigned long long>(ia.ib_buffer),
				static_cast<unsigned long long>(ia.ib_offset),
				ia.ib_index_size, slots);
		}

		// M8 diagnostics: dump the input layout semantics of the
		// pipeline the game used for this draw. UE's DX12 runtime
		// re-packs vertex streams, so this is the ground truth for
		// which slot really carries BLENDINDICES/BLENDWEIGHT (the
		// skin data whose absence causes the comps-3-6 deformation
		// and spring wobble) and in which format.
		if (first_skip)
		{
			uint64_t pipe = 0;
			{
				std::lock_guard<std::mutex> guard(g_state.lock);
				const auto pit = g_state.cl_pipeline.find(cmd_list->get_native());
				if (pit != g_state.cl_pipeline.end())
					pipe = pit->second;
			}
			const std::vector<reshade::api::pipeline_subobject> *subs =
				pipe != 0 ? g_state.pipelines.clone_source(pipe) : nullptr;
			if (subs != nullptr)
			{
				for (const reshade::api::pipeline_subobject &sub : *subs)
				{
					if (sub.type != reshade::api::pipeline_subobject_type::input_layout ||
						sub.data == nullptr || sub.count == 0)
						continue;
					const auto *els =
						static_cast<const reshade::api::input_element *>(sub.data);
					char elems[768];
					size_t used = 0;
					elems[0] = '\0';
					for (uint32_t e = 0; e < sub.count && used + 80 < sizeof(elems); ++e)
					{
						const char *fname = vertex_format_name(els[e].format);
						char fbuf[16];
						if (fname == nullptr)
						{
							sprintf_s(fbuf, "f%u", static_cast<uint32_t>(els[e].format));
							fname = fbuf;
						}
						used += static_cast<size_t>(sprintf_s(elems + used,
							sizeof(elems) - used, "[%u]%s%u %s off=%u st=%u; ",
							els[e].buffer_binding,
							els[e].semantic != nullptr ? els[e].semantic : "?",
							els[e].semantic_index, fname,
							els[e].offset, els[e].stride));
					}
					log::info("pipe %llx input layout: %s",
						static_cast<unsigned long long>(pipe), elems);
				}
			}
			else
			{
				log::info("pipe %llx input layout: not cached (clone source evicted)",
					static_cast<unsigned long long>(pipe));
			}
		}

		// M8 diagnostics: same first skip also dumps the graphics root
		// state -- layout parameter types and every bound root CBV with
		// its buffer size. The skeleton CBs are the large (>= ~4KB)
		// vertex-stage CBs; identifying them is step one of the bone
		// capture/rebind system (fixes comps 3-6 deformation).
		if (first_skip)
		{
			reshade::api::device *const dev = cmd_list->get_device();
			RuntimeState::ClRootState root;
			const std::vector<reshade::api::pipeline_layout_param> *params = nullptr;
			{
				std::lock_guard<std::mutex> guard(g_state.lock);
				const auto it = g_state.cl_root.find(cmd_list->get_native());
				if (it != g_state.cl_root.end())
				{
					root = it->second;
					const auto lit = g_state.layouts.find(root.layout);
					if (lit != g_state.layouts.end())
						params = &lit->second;
				}
			}

			if (params != nullptr)
		{
			// M9: dump each parameter with its descriptor class and, for
			// push descriptors, register/space/visibility -- the mapping
			// data the skeleton chain keys on (push CBV b3/b4 = the DX11
			// mod's vs-cb3/vs-cb4).
			char desc[1024];
			size_t used = 0;
			desc[0] = '\0';
			for (size_t i = 0; i < params->size() && used + 72 < sizeof(desc); ++i)
			{
				const reshade::api::pipeline_layout_param &p = (*params)[i];
				switch (p.type)
				{
				case reshade::api::pipeline_layout_param_type::descriptor_table:
				case reshade::api::pipeline_layout_param_type::descriptor_table_with_flags:
					used += static_cast<size_t>(sprintf_s(desc + used,
						sizeof(desc) - used, "[%zu]table ", i));
					break;
				case reshade::api::pipeline_layout_param_type::push_constants:
					used += static_cast<size_t>(sprintf_s(desc + used,
						sizeof(desc) - used, "[%zu]const ", i));
					break;
				default:
				{
					const reshade::api::descriptor_range *r = nullptr;
					const char *cls = "?";
					if (p.type == reshade::api::pipeline_layout_param_type::push_descriptors)
						r = &p.push_descriptors;
					else if (p.type == reshade::api::pipeline_layout_param_type::push_descriptors_with_ranges ||
						p.type == reshade::api::pipeline_layout_param_type::push_descriptors_with_ranges_and_flags)
					{
						if (p.descriptor_table_with_flags.count >= 1 &&
							p.descriptor_table_with_flags.ranges != nullptr)
							r = &p.descriptor_table_with_flags.ranges[0];
					}
					const char *rtype = "?";
					if (r != nullptr)
					{
						if (r->type == reshade::api::descriptor_type::constant_buffer)
							rtype = "b";
						else if (r->type == reshade::api::descriptor_type::shader_resource_view)
							rtype = "t";
						else if (r->type == reshade::api::descriptor_type::unordered_access_view)
							rtype = "u";
						else
							rtype = "s";
						used += static_cast<size_t>(sprintf_s(desc + used,
							sizeof(desc) - used, "[%zu]push %s%u/s%u/%s ", i, rtype,
							r->dx_register_index, r->dx_register_space,
							(static_cast<uint32_t>(r->visibility) &
								static_cast<uint32_t>(reshade::api::shader_stage::vertex))
								? "v" : "-"));
						(void)cls;
						continue;
					}
					used += static_cast<size_t>(sprintf_s(desc + used,
						sizeof(desc) - used, "[%zu]push %s ", i, cls));
					break;
				}
				}
			}
			log::info("root layout %llx: %s",
				static_cast<unsigned long long>(root.layout), desc);
		}
			else
			{
				log::info("root layout: no push/table events recorded for this command list (bindings may live in descriptor heaps)");
			}

			for (const auto &kv : root.cbvs)
			{
				const uint32_t param = static_cast<uint32_t>(kv.first >> 32);
				const uint32_t reg = static_cast<uint32_t>(kv.first & 0xffffffffu);
				uint64_t size = 0;
				if (dev != nullptr && kv.second.buffer != 0)
				{
					const reshade::api::resource_desc d =
						dev->get_resource_desc({ kv.second.buffer });
					if (d.type == reshade::api::resource_type::buffer)
						size = d.buffer.size;
				}
				log::info("root cbv param=%u reg=%u: buffer %llx +%llu (%llu bytes)",
					param, reg,
					static_cast<unsigned long long>(kv.second.buffer),
					static_cast<unsigned long long>(kv.second.offset),
					static_cast<unsigned long long>(size));
			}
			for (const auto &kv : root.tables)
				log::info("root table param=%u: base %llx", kv.first,
					static_cast<unsigned long long>(kv.second));
		}

		// Script path: run the rule body; its 'handling = skip' execution
		// is the actual skip decision. IA bindings the script captured are
		// re-applied afterwards (see restore_ia).
		if (mrt != nullptr)
		{
			reshade::api::device *const device = g_state.device.load();
			if (device != nullptr)
			{
				std::lock_guard<std::mutex> guard(mrt->lock);

				const uint64_t before_draws = g_state.script_draws_issued.load();
				const uint64_t before_lists = mrt->rt.lists_executed;

				bool skip = false;
			{
				ScriptExecGuard exec_guard;
				AddonBridge bridge(device, cmd_list);
				// M9: snapshot the skeleton CBVs and arm the lazy CBV
				// replacement (applies at the first re-issued draw, is
				// pushed back after the body -- both inside this guard so
				// the pushes never pollute the root-state tracking).
				prepare_skeleton(cmd_list, *rule, &bridge);
				skip = mrt->rt.run_override(rule->section, &bridge, have_ia ? &ia : nullptr);
				if (have_ia)
					restore_ia(cmd_list, ia);
				bridge.restore_skeleton();
			}

				g_state.script_lists_executed.fetch_add(
					mrt->rt.lists_executed - before_lists);
				if (skip)
				{
					++g_state.script_draws_blocked;
					++g_state.draws_skipped;
					if (first_skip)
						log::info("script skip engaged: rule '%s' (%llu draw(s) re-issued)",
							rule->section.c_str(),
							static_cast<unsigned long long>(
								g_state.script_draws_issued.load() - before_draws));
					return true;
				}
				return false; // body ran without skip: original draw proceeds
			}
		}

		// Static fallback (M2): no runtime or no device -> plain skip.
		++g_state.draws_skipped;
		if (first_skip)
			log::info("draw skip engaged: rule '%s' (hash=%08x)", rule->section.c_str(), rule->hash);
		return true;
	}

	// ---------------------------------------------------------------------
	// M4: ShaderOverride -- draw-time shader-hash matching
	// ---------------------------------------------------------------------

	// Runs the [ShaderOverride*] bodies whose hash matches any stage of
	// the pipeline currently bound on this command list (3DMigoto
	// matches per draw; typical use is the '$attack = 1' latch). The
	// bodies never block the draw (a 'handling = skip' inside a
	// ShaderOverride body is a known M4 limitation).
	static void run_shader_overrides(reshade::api::command_list *cmd_list)
	{
		// Snapshot under <lock>, execute outside it (lock order:
		// runtime -> lock, and the bodies may take the runtime lock).
		struct Match
		{
			uint32_t runtime;
			std::string section;
		};
		Match matches[PipelineShaders::stage_count];
		uint32_t match_count = 0;
		{
			std::lock_guard<std::mutex> guard(g_state.lock);

			const auto cl = g_state.cl_pipeline.find(cmd_list->get_native());
			if (cl == g_state.cl_pipeline.end())
				return;
			const PipelineShaders *sh = g_state.pipelines.find(cl->second);
			if (sh == nullptr)
				return;

			const uint64_t hashes[PipelineShaders::stage_count] = {
				sh->vs, sh->ps, sh->gs, sh->ds, sh->hs, sh->cs
			};
			for (uint64_t h : hashes)
			{
				if (h == 0)
					continue;
				const auto it = g_state.shader_overrides.find(h);
				if (it == g_state.shader_overrides.end() ||
					it->second.runtime >= g_state.runtimes.size())
					continue;

				// de-dup (a hash can hit twice only when stages collide)
				bool dup = false;
				for (uint32_t i = 0; i < match_count; ++i)
					if (matches[i].runtime == it->second.runtime &&
						matches[i].section == it->second.section)
					{
						dup = true;
						break;
					}
				if (!dup)
					matches[match_count++] = { it->second.runtime, it->second.section };
			}
		}

		for (uint32_t i = 0; i < match_count; ++i)
		{
			RuntimeState::ModRuntime *mrt = g_state.runtimes[matches[i].runtime].get();
			reshade::api::device *const device = g_state.device.load();
			if (device == nullptr)
				return;

			std::lock_guard<std::mutex> guard(mrt->lock);

			const uint64_t before_lists = mrt->rt.lists_executed;
			{
				ScriptExecGuard exec_guard;
				AddonBridge bridge(device, cmd_list);
				mrt->rt.run_list(matches[i].section, &bridge, nullptr);
			}
			g_state.script_lists_executed.fetch_add(
				mrt->rt.lists_executed - before_lists);
			++g_state.shader_overrides_hit;
		}
	}

	// ---------------------------------------------------------------------
	// Replacement pipeline (m1-9)
	// ---------------------------------------------------------------------

	// Rewrites one descriptor slot to the replacement view. update tables
	// with a texture SRV descriptor copies the view into the slot, exactly
	// what the create path did originally.
	static void apply_overwrite(reshade::api::device *device, const Overwrite &ow)
	{
		reshade::api::resource_view view{ ow.view };
		reshade::api::descriptor_table_update upd;
		upd.table = reshade::api::descriptor_table{ ow.table };
		upd.binding = ow.binding;
		upd.array_offset = 0;
		upd.type = reshade::api::descriptor_type::texture_shader_resource_view;
		upd.count = 1;
		upd.descriptors = &view;
		device->update_descriptor_tables(1, &upd);
		++g_state.overwrites_applied;
	}

	static void apply_overwrites(reshade::api::device *device, const std::vector<Overwrite> &ows)
	{
		for (const Overwrite &ow : ows)
			apply_overwrite(device, ow);
	}

	// Loads (or reuses) the replacement texture of a mod file and returns
	// its SRV. The pair stays cached for the device lifetime: freeing a
	// texture that in-flight game frames may still sample would need queue
	// idle waits per eviction, which is not worth it for appearance mods.
	static bool get_or_create_replacement(reshade::api::command_queue *queue,
		const std::filesystem::path &path, Replacement *out)
	{
		reshade::api::device *const device = queue->get_device();
		const std::string key = path.string();
		{
			std::lock_guard<std::mutex> guard(g_state.lock);
			const auto it = g_state.replacements.find(key);
			if (it != g_state.replacements.end())
			{
				*out = it->second;
				return true;
			}
		}

		std::vector<uint8_t> blob;
		DdsTexture dds;
		std::string err;
		if (!load_dds_file(key.c_str(), &blob, &dds, &err))
		{
			log::warn("cannot load replacement %s (%s)", key.c_str(), err.c_str());
			return false;
		}

		reshade::api::resource texture{};
		reshade::api::resource_view view{};

		{
			// Our own resources: keep them out of tracking/hashing.
			OwnCreationGuard guard;

			const reshade::api::resource_flags flags = dds.is_cubemap
				? reshade::api::resource_flags::cube_compatible
				: reshade::api::resource_flags::none;
			const reshade::api::resource_desc td(
				dds.width, dds.height, static_cast<uint16_t>(dds.array_size),
				static_cast<uint16_t>(dds.mip_levels),
				static_cast<reshade::api::format>(dds.format), 1,
				reshade::api::memory_heap::gpu_only,
				reshade::api::resource_usage::shader_resource | reshade::api::resource_usage::copy_dest,
				flags);

			if (!device->create_resource(td, nullptr, reshade::api::resource_usage::copy_dest, &texture))
			{
				log::warn("replacement texture creation failed for %s", key.c_str());
				return false;
			}

			// Upload every subresource (slice-major, matching ReShade's
			// 'level + layer * levels' indexing).
			for (uint32_t slice = 0; slice < dds.array_size; ++slice)
			{
				for (uint32_t mip = 0; mip < dds.mip_levels; ++mip)
				{
					const DdsSubresource *sub = dds.subresource(slice, mip);
					if (sub == nullptr)
						continue;

					reshade::api::subresource_data data;
					data.data = const_cast<uint8_t *>(dds.data) + sub->offset;
					data.row_pitch = static_cast<uint32_t>(sub->row_pitch);
					data.slice_pitch = static_cast<uint32_t>(sub->slice_pitch);
					device->update_texture_region(data, texture, mip + slice * dds.mip_levels, nullptr);
				}
			}

			// Cubemaps must present a cube view or cubemap-sampling shaders
			// would read a 2D array SRV instead.
			const reshade::api::resource_view_desc vd(
				dds.is_cubemap ? reshade::api::resource_view_type::texture_cube
					: reshade::api::resource_view_type::texture_2d,
				static_cast<reshade::api::format>(dds.format),
				0, dds.mip_levels, 0,
				dds.is_cubemap ? dds.array_size / 6 : dds.array_size);

			if (!device->create_resource_view(texture, reshade::api::resource_usage::shader_resource, vd, &view))
			{
				log::warn("replacement SRV creation failed for %s", key.c_str());
				device->destroy_resource(texture);
				return false;
			}
		}

		// Transition to shader_resource and submit BEFORE any descriptor
		// overwrite is applied: queue order then guarantees the upload is
		// complete before a draw that reads the overwritten slot.
		reshade::api::command_list *const cmd = queue->get_immediate_command_list();
		cmd->barrier(texture, reshade::api::resource_usage::copy_dest,
			reshade::api::resource_usage::shader_resource);
		queue->flush_immediate_command_list();

		{
			std::lock_guard<std::mutex> guard(g_state.lock);
			g_state.replacements[key] = Replacement{ texture, view };
		}

		log::info("replacement texture ready: %s (%ux%u, %u mips, fmt=%u)",
			key.c_str(), dds.width, dds.height, dds.mip_levels, dds.format);

		*out = Replacement{ texture, view };
		return true;
	}

	static void activate_replacement(reshade::api::command_queue *queue, const ActivationReq &req)
	{
		{
			std::lock_guard<std::mutex> guard(g_state.lock);
			if (g_state.engine.is_replaced(req.resource))
				return; // already overridden
		}

		Replacement rep;
		if (!get_or_create_replacement(queue, req.texture_path, &rep))
			return;

		reshade::api::device *const device = queue->get_device();

		std::vector<Overwrite> overwrites;
		{
			std::lock_guard<std::mutex> guard(g_state.lock);
			overwrites = g_state.engine.activate(req.resource, rep.view.handle);

			RuntimeState::ActivationRecord rec;
			rec.resource = req.resource;
			rec.texture = req.texture_path.filename().string();
			rec.slot_rewrites = static_cast<uint32_t>(overwrites.size());
			rec.frame = g_state.frame; // only mutated under the lock (on_present)
			g_state.activation_log.insert(g_state.activation_log.begin(), std::move(rec));
			if (g_state.activation_log.size() > RuntimeState::k_max_activation_log)
				g_state.activation_log.resize(RuntimeState::k_max_activation_log);
		}
		apply_overwrites(device, overwrites);

		++g_state.active_replacements;
		log::info("replacement active: texture %p -> %S (%llu slot rewrite(s))",
			reinterpret_cast<void *>(req.resource), req.texture_path.c_str(),
			static_cast<unsigned long long>(overwrites.size()));
	}

	static void process_activations(reshade::api::command_queue *queue)
	{
		if (queue == nullptr)
			return; // no graphics queue yet: requests stay queued

		std::vector<ActivationReq> reqs;
		{
			std::lock_guard<std::mutex> guard(g_state.lock);
			reqs.swap(g_state.activations);
		}

		for (const ActivationReq &req : reqs)
		activate_replacement(queue, req);
	}

	// ---------------------------------------------------------------------
	// Prefilter shared by the tracker bridge
	// ---------------------------------------------------------------------

	static bool is_hash_candidate(const reshade::api::resource_desc &desc)
	{
		// Mod targets are plain sampled material textures. Anything that can
		// be rendered into / written by the GPU is a post-process target or
		// an atlas the game regenerates: its content hash is meaningless for
		// mod matching, and copying it out mid-use risks a wrong-state
		// barrier (observed as DXGI_ERROR_DEVICE_HUNG on NVIDIA).
		constexpr reshade::api::resource_usage k_active_write =
			reshade::api::resource_usage::render_target |
			reshade::api::resource_usage::depth_stencil |
			reshade::api::resource_usage::unordered_access;

		return desc.type == reshade::api::resource_type::texture_2d &&
			desc.heap == reshade::api::memory_heap::gpu_only &&
			(desc.usage & reshade::api::resource_usage::shader_resource) != 0 &&
			(desc.usage & k_active_write) == reshade::api::resource_usage::undefined &&
			desc.texture.samples <= 1 &&
			desc.texture.depth_or_layers <= 8; // cubemaps+small arrays only
	}

	// Readback safety gate: the present-time state snapshot must describe a
	// resource at rest. States like render_target / depth_stencil /
	// unordered_access mean the game's parallel command lists may still be
	// mid-use; transitioning from a guessed state is what hangs the device.
	static bool state_safe_for_readback(reshade::api::resource_usage state)
	{
		using u = reshade::api::resource_usage;
		switch (state)
		{
		case u::undefined:
		case u::general:
		case u::copy_source:
		case u::copy_dest:
		case u::shader_resource:
		case u::shader_resource_pixel:
		case u::shader_resource_non_pixel:
			return true;
		default:
			return false;
		}
	}

} // namespace wwmi

// ---------------------------------------------------------------------------
// Event handlers
// ---------------------------------------------------------------------------

static void on_init_device(reshade::api::device *device)
{
	wwmi::g_state.devices.fetch_add(1);

	if (device->get_api() != reshade::api::device_api::d3d12)
	{
		wwmi::log::warn("non-D3D12 device (api %u); texture overrides disabled",
			static_cast<uint32_t>(device->get_api()));
		return;
	}

	{
		std::lock_guard<std::mutex> guard(wwmi::g_state.lock);
		wwmi::g_state.frame = 0;

		// Load mods BEFORE publishing the device: command-list events can
		// fire from other threads the moment the device is visible, and
		// draw_intercept must not miss the rules of this load.
		if (!wwmi::g_state.mods_loaded)
		{
			wwmi::load_mods();
			wwmi::g_state.mods_loaded = true;
		}
	}

	wwmi::g_state.device.store(device);
	wwmi::g_state.draw_intercept.store(wwmi::g_state.index.has_draw_rules());
	wwmi::log::info("draw interception %s (%llu draw rule(s), VB/IB tracking %s)",
		wwmi::g_state.index.has_draw_rules() ? "enabled" : "disabled",
		static_cast<unsigned long long>(wwmi::g_state.index.draw_rule_count()),
		wwmi::g_state.index.has_draw_rules() ? "on" : "off");

	wwmi::log::info("WWMI-DX12 attached (d3d12).");
}

static void on_destroy_device(reshade::api::device *device)
{
	// M3: release script-owned GPU buffers and flush persist files while
	// the device is still alive. Runs BEFORE taking <lock>: the bridge
	// never takes locks itself (lock order stays runtime -> lock).
	{
		reshade::api::device *const live = wwmi::g_state.device.load();
		if (live != nullptr)
		{
			const std::filesystem::path persist_dir = wwmi::data_root() / "Persist";
			wwmi::AddonBridge bridge(live, nullptr);
			for (const auto &mrt : wwmi::g_state.runtimes)
			{
				std::lock_guard<std::mutex> guard(mrt->lock);
				mrt->rt.destroy_resources(&bridge);
				std::error_code ec;
				std::filesystem::create_directories(persist_dir, ec);
				mrt->rt.save_persist(
					persist_dir / (mrt->rt.mod_dir().filename().string() + ".ini"));
			}
		}
	}

	{
		std::lock_guard<std::mutex> guard(wwmi::g_state.lock);

		if (wwmi::g_state.session_dirty)
			wwmi::save_session();

		// Free our replacement textures while the device is still alive.
		if (device == wwmi::g_state.device.load())
		{
			for (const auto &[path, rep] : wwmi::g_state.replacements)
			{
				device->destroy_resource_view(rep.view);
				device->destroy_resource(rep.texture);
			}
			wwmi::g_state.replacements.clear();
			wwmi::g_state.activations.clear();

			// M2 pools are keyed by device-specific handles.
			wwmi::g_state.buffers.clear();
			wwmi::g_state.draw_states.clear();
			wwmi::g_state.skip_log.clear();

			// M6: pool-view state and queued probes reference the same
			// device-specific handles.
			wwmi::g_state.index_views.clear();
			wwmi::g_state.view_probes.clear();
		}
	}

	wwmi::g_state.device.store(nullptr);

	const auto &st = wwmi::g_state.tracker.stats();
	const auto &bs = wwmi::g_state.buffers.stats();
	const auto &vs = wwmi::g_state.index_views.stats();
	wwmi::log::info("WWMI-DX12 detaching. stats: tracked=%llu hashed=%llu (initial=%llu readback=%llu) "
		"unsupported=%llu evicted=%llu matches=%llu srv_views=%llu | "
		"desc_updates=%llu desc_copies=%llu overwrites=%llu replacements=%llu | "
		"M2 buffers: tracked=%llu binds=%llu hashed=%llu unsupported=%llu evicted=%llu skipped_draws=%llu | "
		"M6 views: tracked=%llu verified=%llu failed=%llu evicted=%llu probes=%llu | "
		"M9 skeleton: captures=%llu pushes=%llu",
		static_cast<unsigned long long>(st.tracked),
		static_cast<unsigned long long>(st.hashed),
		static_cast<unsigned long long>(wwmi::g_state.initialdata_hashes.load()),
		static_cast<unsigned long long>(wwmi::g_state.readback_hashes.load()),
		static_cast<unsigned long long>(st.unsupported),
		static_cast<unsigned long long>(st.evicted),
		static_cast<unsigned long long>(wwmi::g_state.hash_matches.load()),
		static_cast<unsigned long long>(wwmi::g_state.srv_views.load()),
		static_cast<unsigned long long>(wwmi::g_state.descriptor_updates.load()),
		static_cast<unsigned long long>(wwmi::g_state.descriptor_copies.load()),
		static_cast<unsigned long long>(wwmi::g_state.overwrites_applied.load()),
		static_cast<unsigned long long>(wwmi::g_state.active_replacements.load()),
		static_cast<unsigned long long>(bs.tracked),
		static_cast<unsigned long long>(bs.binds),
		static_cast<unsigned long long>(bs.hashed),
		static_cast<unsigned long long>(bs.unsupported),
		static_cast<unsigned long long>(bs.evicted),
		static_cast<unsigned long long>(wwmi::g_state.draws_skipped.load()),
		static_cast<unsigned long long>(vs.tracked),
		static_cast<unsigned long long>(vs.verified),
		static_cast<unsigned long long>(vs.failed),
		static_cast<unsigned long long>(vs.evicted),
		static_cast<unsigned long long>(wwmi::g_state.view_probes_issued.load()),
		static_cast<unsigned long long>(wwmi::g_state.skel_captures),
		static_cast<unsigned long long>(wwmi::g_state.skel_pushes));
}

static void on_init_resource(reshade::api::device *device, const reshade::api::resource_desc &desc,
	const reshade::api::subresource_data *initial_data, reshade::api::resource_usage initial_state, reshade::api::resource resource)
{
	if (wwmi::t_own_creation)
		return; // our replacement / readback resources
	if (!wwmi::is_hash_candidate(desc))
		return;

	wwmi::TrackedTexture snap;
	bool do_initial = false;
	{
		std::lock_guard<std::mutex> guard(wwmi::g_state.lock);

		wwmi::TrackedTexture *t = wwmi::g_state.tracker.track(resource.handle,
			desc.texture.width, desc.texture.height,
			static_cast<uint32_t>(desc.texture.format),
			desc.texture.levels, desc.texture.depth_or_layers,
			wwmi::g_state.frame);
		if (t == nullptr)
			return;

		wwmi::g_state.states[resource.handle] = initial_state;
		snap = *t; // copy out: hashing runs outside the lock

		// Learning path A: the game uploaded the texture at creation time.
		if (initial_data != nullptr && initial_data->data != nullptr && t->levels == 1 && t->layers == 1)
			do_initial = true;
	}

	if (do_initial)
		wwmi::hash_from_initial_data(resource.handle, snap, *initial_data);

	// Initial-data hashes may have queued replacements; the graphics queue
	// does not exist yet during early startup, requests simply stay queued.
	wwmi::process_activations(wwmi::g_state.queue.load());
}

static void on_destroy_resource(reshade::api::device *device, reshade::api::resource resource)
{
	std::lock_guard<std::mutex> guard(wwmi::g_state.lock);
	wwmi::g_state.tracker.untrack(resource.handle);
	wwmi::g_state.buffers.untrack(resource.handle); // M2 (no-op for non-buffers)
	wwmi::g_state.states.erase(resource.handle);
	wwmi::g_state.engine.on_resource_destroyed(resource.handle);

	// Drop view records pointing at the destroyed resource. Replacement
	// textures stay cached (keyed by file path) for reuse.
	for (auto it = wwmi::g_state.view_resource.begin(); it != wwmi::g_state.view_resource.end();)
	{
		if (it->second == resource.handle)
			it = wwmi::g_state.view_resource.erase(it);
		else
			++it;
	}
}

static void on_barrier(reshade::api::command_list *cmd_list, uint32_t count,
	const reshade::api::resource *resources, const reshade::api::resource_usage *old_states,
	const reshade::api::resource_usage *new_states)
{
	std::lock_guard<std::mutex> guard(wwmi::g_state.lock);
	for (uint32_t i = 0; i < count; ++i)
	{
		if (resources[i].handle == 0)
			continue; // global / non-resource barrier
		wwmi::g_state.states[resources[i].handle] = new_states[i];
	}
}

static void on_init_resource_view(reshade::api::device *device, reshade::api::resource resource,
	reshade::api::resource_usage usage_type, const reshade::api::resource_view_desc &desc, reshade::api::resource_view view)
{
	if (usage_type != reshade::api::resource_usage::shader_resource)
		return;
	++wwmi::g_state.srv_views;

	if (wwmi::t_own_creation)
		return;

	std::lock_guard<std::mutex> guard(wwmi::g_state.lock);

	// Only remember views of tracked textures; the update event resolves
	// its descriptors through this table.
	if (wwmi::g_state.tracker.find(resource.handle) == nullptr)
		return;
	if (wwmi::g_state.view_resource.size() >= wwmi::RuntimeState::k_max_view_records)
		return;
	wwmi::g_state.view_resource[view.handle] = resource.handle;
}

// Create path: the game wrote an SRV descriptor (CreateShaderResourceView
// and friends). Records the slot for its resource and rewrites it right
// away when that resource is replaced.
//
// NOTE: on D3D12 this event fires AFTER the original call already ran, so
// returning true would not undo it -- always return false and fix the slot
// up via update_descriptor_tables instead.
static bool on_update_descriptor_tables(reshade::api::device *device, uint32_t count,
	const reshade::api::descriptor_table_update *updates)
{
	if (device != wwmi::g_state.device.load())
		return false;

	bool has_srv = false;
	for (uint32_t i = 0; i < count && !has_srv; ++i)
		has_srv = updates[i].type == reshade::api::descriptor_type::texture_shader_resource_view;
	if (!has_srv)
		return false;

	std::vector<wwmi::Overwrite> overwrites;
	{
		std::lock_guard<std::mutex> guard(wwmi::g_state.lock);

		// Nothing tracked -> nothing to record or override.
		if (wwmi::g_state.view_resource.empty() && wwmi::g_state.engine.replaced_count() == 0)
			return false;

		for (uint32_t i = 0; i < count; ++i)
		{
			const reshade::api::descriptor_table_update &u = updates[i];
			if (u.type != reshade::api::descriptor_type::texture_shader_resource_view ||
				u.descriptors == nullptr || u.count == 0 || u.array_offset != 0)
				continue;

			// Canonicalize the first slot: (descriptor heap, index).
			reshade::api::descriptor_heap heap{};
			uint32_t base_index = 0;
			device->get_descriptor_heap_offset(u.table, u.binding, 0, &heap, &base_index);
			if (heap.handle == 0)
				continue; // table not resolvable (should not happen on D3D12)

			const reshade::api::resource_view *views =
				reinterpret_cast<const reshade::api::resource_view *>(u.descriptors);

			for (uint32_t k = 0; k < u.count; ++k)
			{
				const auto vit = wwmi::g_state.view_resource.find(views[k].handle);
				if (vit == wwmi::g_state.view_resource.end())
					continue; // not a tracked texture's SRV

				const wwmi::SlotId id{ heap.handle, base_index + k };
				const wwmi::SlotWrite write{ u.table.handle, u.binding + k };

				std::vector<wwmi::Overwrite> ows =
					wwmi::g_state.engine.on_srv_slot(vit->second, id, write);
				overwrites.insert(overwrites.end(), ows.begin(), ows.end());
			}
		}
	}

	++wwmi::g_state.descriptor_updates;
	wwmi::apply_overwrites(device, overwrites);
	return false;
}

// Copy path: the game copies descriptors (CPU staging -> shader-visible
// heap). When a replaced texture's slot is copied, the original call would
// propagate the ORIGINAL descriptor and undo the override, so block it,
// re-issue verbatim, then rewrite the affected dest slots.
static bool on_copy_descriptor_tables(reshade::api::device *device, uint32_t count,
	const reshade::api::descriptor_table_copy *copies)
{
	if (device != wwmi::g_state.device.load())
		return false;
	if (count == 0 || copies == nullptr)
		return false;

	wwmi::CopyPlan plan;
	{
		std::lock_guard<std::mutex> guard(wwmi::g_state.lock);

		// Without recorded slots a copy can neither be tracked nor undone.
		if (wwmi::g_state.engine.slot_count() == 0)
			return false;

		std::vector<wwmi::CopySegment> segments;
		segments.reserve(count);

		for (uint32_t i = 0; i < count; ++i)
		{
			const reshade::api::descriptor_table_copy &c = copies[i];
			if (c.count == 0)
				continue;

			reshade::api::descriptor_heap src_heap{}, dst_heap{};
			uint32_t src_index = 0, dst_index = 0;
			device->get_descriptor_heap_offset(c.source_table, c.source_binding, 0, &src_heap, &src_index);
			device->get_descriptor_heap_offset(c.dest_table, c.dest_binding, 0, &dst_heap, &dst_index);
			if (src_heap.handle == 0 || dst_heap.handle == 0)
				continue;

			wwmi::CopySegment seg;
			seg.source = { src_heap.handle, src_index };
			seg.dest = { dst_heap.handle, dst_index };
			seg.dest_table = c.dest_table.handle;
			seg.dest_binding = c.dest_binding;
			seg.count = c.count;
			segments.push_back(seg);
		}

		plan = wwmi::g_state.engine.on_copy(segments.data(), segments.size());
	}

	++wwmi::g_state.descriptor_copies;

	if (!plan.block)
		return false; // original CopyDescriptors proceeds untouched

	// Re-issue the copy exactly as the game intended (goes straight to the
	// original device, no re-entrant event), then fix up replaced slots.
	device->copy_descriptor_tables(count, copies);
	wwmi::apply_overwrites(device, plan.overwrites);
	return true; // block the original call
}

// Track the game's graphics command queue: entry point for the immediate
// command list used by readback and replacement uploads.
static void on_init_command_queue(reshade::api::command_queue *queue)
{
	if ((queue->get_type() & reshade::api::command_queue_type::graphics) == 0)
		return;
	wwmi::g_state.queue.store(queue);
}

static void on_destroy_command_queue(reshade::api::command_queue *queue)
{
	if (wwmi::g_state.queue.load() == queue)
		wwmi::g_state.queue.store(nullptr);
}

// M2: command list going away -> drop its IA state snapshot.
static void on_destroy_command_list(reshade::api::command_list *cmd_list)
{
	std::lock_guard<std::mutex> guard(wwmi::g_state.lock);
	wwmi::g_state.draw_states.erase(cmd_list->get_native());
	// M4: pipeline records die with the command list
	wwmi::g_state.cl_pipeline.erase(cmd_list->get_native());
	wwmi::g_state.cl_compute_pipeline.erase(cmd_list->get_native());
	// M8: root state dies with the command list
	wwmi::g_state.cl_root.erase(cmd_list->get_native());
}

// M2: IA index buffer binding. Replays into the command list's IaState
// and lazily admits the buffer into the pool -- the IB role (and with it
// the D3D11 bind flags of the 3DMigoto hash) is only known here.
static void on_bind_index_buffer(reshade::api::command_list *cmd_list,
	reshade::api::resource buffer, uint64_t offset, uint32_t index_size)
{
	if (wwmi::t_script_exec)
		return; // the script runtime's own rebind: IaState stays untouched
	if (!wwmi::g_state.draw_intercept.load(std::memory_order_relaxed))
		return; // no draw-intercepting mod loaded: zero overhead

	uint64_t byte_width = 0;
	if (buffer.handle != 0)
	{
		const reshade::api::resource_desc desc =
			cmd_list->get_device()->get_resource_desc(buffer);
		if (desc.type == reshade::api::resource_type::buffer)
			byte_width = desc.buffer.size;
	}

	std::lock_guard<std::mutex> guard(wwmi::g_state.lock);
	wwmi::g_state.draw_states[cmd_list->get_native()].bind_index_buffer(
		buffer.handle, offset, index_size);
	if (buffer.handle != 0 && byte_width != 0)
		wwmi::g_state.buffers.track(buffer.handle, byte_width,
			wwmi::BufferRole::index, wwmi::g_state.frame);
}

// M2: IA vertex buffer bindings. Same replay + admission as above; slots
// beyond the D3D12 IA slot count (or with a null buffer) only clear state.
static void on_bind_vertex_buffers(reshade::api::command_list *cmd_list, uint32_t first,
	uint32_t count, const reshade::api::resource *buffers, const uint64_t *offsets, const uint32_t *strides)
{
	if (wwmi::t_script_exec)
		return; // the script runtime's own rebind: IaState stays untouched
	if (!wwmi::g_state.draw_intercept.load(std::memory_order_relaxed))
		return;
	if (count == 0 || buffers == nullptr)
		return;

	uint64_t slot_buffers[wwmi::IaState::max_vb_slots] = {};
	uint64_t slot_offsets[wwmi::IaState::max_vb_slots] = {};
	uint32_t slot_strides[wwmi::IaState::max_vb_slots] = {};
	uint64_t slot_widths[wwmi::IaState::max_vb_slots] = {};

	const uint32_t n = (first < wwmi::IaState::max_vb_slots)
		? std::min(count, wwmi::IaState::max_vb_slots - first)
		: 0;

	reshade::api::device *const device = cmd_list->get_device();
	for (uint32_t i = 0; i < n; ++i)
	{
		slot_buffers[i] = buffers[i].handle;
		slot_offsets[i] = offsets != nullptr ? offsets[i] : 0;
		slot_strides[i] = strides != nullptr ? strides[i] : 0;

		if (buffers[i].handle != 0)
		{
			const reshade::api::resource_desc desc =
				device->get_resource_desc(buffers[i]);
			if (desc.type == reshade::api::resource_type::buffer)
				slot_widths[i] = desc.buffer.size;
		}
	}

	std::lock_guard<std::mutex> guard(wwmi::g_state.lock);
	wwmi::g_state.draw_states[cmd_list->get_native()].bind_vertex_buffers(
		first, n, slot_buffers, slot_offsets, slot_strides);
	for (uint32_t i = 0; i < n; ++i)
		if (slot_buffers[i] != 0 && slot_widths[i] != 0)
			wwmi::g_state.buffers.track(slot_buffers[i], slot_widths[i],
				wwmi::BufferRole::vertex, wwmi::g_state.frame);
}

// ---- M8: root signature tracking ----

// Pipeline layout (root signature) created: remember its parameters so
// the component-draw diagnostics can name each root parameter (table vs
// push descriptor, register index, shader visibility).
static void on_init_pipeline_layout(reshade::api::device * /*device*/,
	uint32_t param_count, const reshade::api::pipeline_layout_param *params,
	reshade::api::pipeline_layout layout)
{
	// Layouts are bounded by the game's root signature count; recording
	// is cheap enough to do unconditionally (before load_mods too).
	//
	// M9 fix: ReShade's params array -- and the descriptor-range arrays
	// its table/range parameters point to -- is only valid DURING this
	// callback. A shallow copy leaves every descriptor_table /
	// push_descriptors_with_ranges entry with a dangling 'ranges'
	// pointer, so reading register indices later returned heap garbage
	// (the skeleton identification saw 'sampler / space 0xCC0B...').
	// Deep-copy the range arrays here; on_destroy_pipeline_layout frees
	// them again.
	std::lock_guard<std::mutex> guard(wwmi::g_state.lock);
	auto &stored = wwmi::g_state.layouts[layout.handle];
	stored.assign(params, params + param_count);

	for (reshade::api::pipeline_layout_param &p : stored)
	{
		using pt = reshade::api::pipeline_layout_param_type;
		switch (p.type)
		{
		case pt::descriptor_table:
		{
			const uint32_t count = p.descriptor_table.count;
			const reshade::api::descriptor_range *src = p.descriptor_table.ranges;
			if (count == 0 || src == nullptr)
				continue;
			auto *copy = new reshade::api::descriptor_range[count];
			for (uint32_t i = 0; i < count; ++i)
				copy[i] = src[i]; // POD copy (static_samplers may alias the game's; unused)
			p.descriptor_table.ranges = copy;
			break;
		}
		case pt::descriptor_table_with_flags:
		case pt::push_descriptors_with_ranges:
		case pt::push_descriptors_with_ranges_and_flags:
		{
			const uint32_t count = p.descriptor_table_with_flags.count;
			const reshade::api::descriptor_range_with_flags *src =
				p.descriptor_table_with_flags.ranges;
			if (count == 0 || src == nullptr)
				continue;
			auto *copy = new reshade::api::descriptor_range_with_flags[count];
			for (uint32_t i = 0; i < count; ++i)
				copy[i] = src[i];
			p.descriptor_table_with_flags.ranges = copy;
			break;
		}
		default:
			break; // push_descriptors (inline range) / push_constants: value copy above is complete
		}
	}
}

static void on_destroy_pipeline_layout(reshade::api::device * /*device*/,
	reshade::api::pipeline_layout layout)
{
	std::lock_guard<std::mutex> guard(wwmi::g_state.lock);
	const auto it = wwmi::g_state.layouts.find(layout.handle);
	if (it == wwmi::g_state.layouts.end())
		return;

	// Free the deep-copied range arrays (see on_init_pipeline_layout).
	for (reshade::api::pipeline_layout_param &p : it->second)
	{
		using pt = reshade::api::pipeline_layout_param_type;
		switch (p.type)
		{
		case pt::descriptor_table:
			delete[] p.descriptor_table.ranges;
			p.descriptor_table.ranges = nullptr;
			break;
		case pt::descriptor_table_with_flags:
		case pt::push_descriptors_with_ranges:
		case pt::push_descriptors_with_ranges_and_flags:
			delete[] p.descriptor_table_with_flags.ranges;
			p.descriptor_table_with_flags.ranges = nullptr;
			break;
		default:
			break;
		}
	}
	wwmi::g_state.layouts.erase(it);
}

// SetGraphicsRootConstantBufferView (and SRV/UAV): record every root CBV
// binding. The event only fires for CBVs here -- descriptors are
// buffer_range {buffer, offset, size}. Keyed (param<<32)|register so the
// latest binding per slot wins, exactly like the command list does.
static void on_push_descriptors(reshade::api::command_list *cmd_list,
	reshade::api::shader_stage /*stages*/, reshade::api::pipeline_layout layout,
	uint32_t layout_param, const reshade::api::descriptor_table_update &update)
{
	if (wwmi::t_script_exec)
		return; // our own bridge bindings must not pollute game state
	if (!wwmi::g_state.draw_intercept.load(std::memory_order_relaxed))
		return;
	if (update.type != reshade::api::descriptor_type::constant_buffer)
		return;
	if (update.count == 0 || update.descriptors == nullptr)
		return;

	const auto *ranges =
		static_cast<const reshade::api::buffer_range *>(update.descriptors);

	std::lock_guard<std::mutex> guard(wwmi::g_state.lock);
	wwmi::RuntimeState::ClRootState &state =
		wwmi::g_state.cl_root[cmd_list->get_native()];
	// M9: a pipeline layout switch invalidates every root argument of the
	// command list (D3D12 semantics: the app must rebind all params after
	// changing the root signature). Stale entries from the previous layout
	// would otherwise alias unrelated params -- the skeleton rebind would
	// push garbage CBVs into the game's draws.
	if (state.layout != layout.handle)
	{
		state.cbvs.clear();
		state.tables.clear();
		state.layout = layout.handle;
	}
	for (uint32_t i = 0; i < update.count; ++i)
	{
		if (ranges[i].buffer.handle == 0)
			continue;
		state.cbvs[(static_cast<uint64_t>(layout_param) << 32) |
			(update.binding + i)] = { ranges[i].buffer.handle,
				ranges[i].offset, ranges[i].size };
	}
}

// SetGraphicsRootDescriptorTable: record table bases per parameter.
static void on_bind_descriptor_tables(reshade::api::command_list *cmd_list,
	reshade::api::shader_stage /*stages*/, reshade::api::pipeline_layout layout,
	uint32_t first, uint32_t count, const reshade::api::descriptor_table *tables,
	uint32_t /*dynamic_offset_count*/, const uint32_t * /*dynamic_offsets*/)
{
	if (wwmi::t_script_exec)
		return;
	if (!wwmi::g_state.draw_intercept.load(std::memory_order_relaxed))
		return;
	if (count == 0 || tables == nullptr)
		return;

	std::lock_guard<std::mutex> guard(wwmi::g_state.lock);
	wwmi::RuntimeState::ClRootState &state =
		wwmi::g_state.cl_root[cmd_list->get_native()];
	if (state.layout != layout.handle)
	{
		state.cbvs.clear();
		state.tables.clear();
		state.layout = layout.handle;
	}
	for (uint32_t i = 0; i < count; ++i)
		state.tables[first + i] = tables[i].handle;
}

// M2: non-indexed draw. 3DMigoto DrawCallInfo semantics: IndexCount /
// FirstIndex stay 0 for non-indexed draws.
static bool on_draw(reshade::api::command_list *cmd_list, uint32_t vertex_count,
	uint32_t instance_count, uint32_t first_vertex, uint32_t first_instance)
{
	// M4: ShaderOverride bodies run on every draw when any mod uses one
	// (independent of the draw_intercept gate below).
	if (wwmi::g_state.shader_override_active.load(std::memory_order_relaxed))
		wwmi::run_shader_overrides(cmd_list);

	if (!wwmi::g_state.draw_intercept.load(std::memory_order_relaxed))
		return false; // never block draws without a draw-intercepting mod

	wwmi::DrawCallInfo call;
	call.vertex_count = vertex_count;
	call.instance_count = instance_count;
	call.first_vertex = first_vertex;
	call.first_instance = first_instance;

	return wwmi::evaluate_draw(cmd_list, call, false);
}

// M2: indexed draw. VertexCount stays 0 and FirstVertex carries the signed
// BaseVertexLocation bits, mirroring 3DMigoto's DrawCallInfo.
static bool on_draw_indexed(reshade::api::command_list *cmd_list, uint32_t index_count,
	uint32_t instance_count, uint32_t first_index, int32_t vertex_offset, uint32_t first_instance)
{
	if (wwmi::g_state.shader_override_active.load(std::memory_order_relaxed))
		wwmi::run_shader_overrides(cmd_list);

	if (!wwmi::g_state.draw_intercept.load(std::memory_order_relaxed))
		return false;

	wwmi::DrawCallInfo call;
	call.index_count = index_count;
	call.instance_count = instance_count;
	call.first_index = first_index;
	call.first_vertex = static_cast<uint32_t>(vertex_offset);
	call.first_instance = first_instance;

	return wwmi::evaluate_draw(cmd_list, call, true);
}

// M4: pipeline creation -> hash + (cached) subobjects for cloning.
static void on_init_pipeline(reshade::api::device * /*device*/,
	reshade::api::pipeline_layout layout, uint32_t subobject_count,
	const reshade::api::pipeline_subobject *subobjects, reshade::api::pipeline pipeline)
{
	if (wwmi::t_own_creation)
		return; // our own blend-clone PSOs are not tracked
	wwmi::g_state.pipelines.on_init_pipeline(layout.handle,
		subobject_count, subobjects, pipeline.handle);
}

// M4: pipeline destruction. May be called from inside other API calls, so
// the clone cleanup only marks entries -- the actual destroy_pipeline of
// our clones happens at present time (see on_present).
static void on_destroy_pipeline(reshade::api::device *device, reshade::api::pipeline pipeline)
{
	wwmi::g_state.pipelines.on_destroy_pipeline(pipeline.handle);

	std::vector<reshade::api::pipeline> orphans;
	{
		std::lock_guard<std::mutex> guard(wwmi::g_state.blend_lock);
		for (auto it = wwmi::g_state.blend_clones.begin();
			it != wwmi::g_state.blend_clones.end();)
		{
			if (it->second.src == pipeline.handle)
			{
				orphans.push_back(it->second.pipe);
				it = wwmi::g_state.blend_clones.erase(it);
			}
			else
				++it;
		}
	}
	if (!orphans.empty())
	{
		std::lock_guard<std::mutex> guard(wwmi::g_state.lock);
		for (reshade::api::pipeline p : orphans)
			wwmi::g_state.pending_pipeline_destroys.push_back(p);
	}
}

// M4: current-pipeline replay per command list. Graphics and compute go
// to separate maps (a dispatch binds compute, a draw binds graphics; the
// last bind of the right kind must survive). Reset arrives as handle 0
// and clears both records.
static void on_bind_pipeline(reshade::api::command_list *cmd_list,
	reshade::api::pipeline_stage stages, reshade::api::pipeline pipeline)
{
	if (wwmi::t_script_exec)
		return; // blend-clone rebinds must not replace the tracked source

	const uint64_t cl = cmd_list->get_native();

	if (pipeline.handle == 0)
	{
		std::lock_guard<std::mutex> guard(wwmi::g_state.lock);
		wwmi::g_state.cl_pipeline.erase(cl);
		wwmi::g_state.cl_compute_pipeline.erase(cl);
		return;
	}

	const bool compute_only =
		(stages & reshade::api::pipeline_stage::all_compute) != reshade::api::pipeline_stage{} &&
		(stages & reshade::api::pipeline_stage::all_graphics) == reshade::api::pipeline_stage{};
	const bool graphics =
		(stages & reshade::api::pipeline_stage::all_graphics) != reshade::api::pipeline_stage{};
	if (!compute_only && !graphics)
		return; // raytracing / unknown: leave the records alone

	std::lock_guard<std::mutex> guard(wwmi::g_state.lock);
	if (compute_only)
		wwmi::g_state.cl_compute_pipeline[cl] = pipeline.handle;
	else
		wwmi::g_state.cl_pipeline[cl] = pipeline.handle;
}

// Present = frame boundary: spend the readback budget on the oldest pending
// textures. GPU work is idle here, so the sync stall is minimal.
static void on_present(reshade::api::command_queue *queue, reshade::api::swapchain *swapchain,
	const reshade::api::rect *source_rect, const reshade::api::rect *dest_rect,
	uint32_t dirty_rect_count, const reshade::api::rect *dirty_rects)
{
	reshade::api::device *const device = wwmi::g_state.device.load();
	if (device == nullptr)
		return;

	// M3: mod key bindings (edge detection inside the runtime). Polled
	// once per frame; paused while the ReShade overlay has focus.
	if (wwmi::g_state.has_scripts && !wwmi::g_state.overlay_open.load())
	{
		for (const auto &mrt : wwmi::g_state.runtimes)
		{
			std::lock_guard<std::mutex> guard(mrt->lock);
			for (const wwmi::KeyBinding &kb : mrt->rt.keys())
			{
				const bool down = (GetAsyncKeyState(static_cast<int>(kb.vk)) & 0x8000) != 0;
				mrt->rt.on_key(kb.vk, down);
			}
		}
	}

	// M3: [Present] scripts (variable housekeeping like attack-frame
	// latches). No draw context exists here, so the bridge carries no
	// command list: vb/ib/drawindexed no-op, resources still work.
	if (wwmi::g_state.has_scripts)
	{
		for (const auto &mrt : wwmi::g_state.runtimes)
		{
			std::lock_guard<std::mutex> guard(mrt->lock);
			wwmi::AddonBridge bridge(device, nullptr);
			mrt->rt.run_present(&bridge);
		}
	}

	std::vector<wwmi::TrackedTexture> pending;
	std::vector<reshade::api::resource_usage> state_snapshot;
	std::vector<reshade::api::pipeline> retired_pipelines;

	{
		std::lock_guard<std::mutex> guard(wwmi::g_state.lock);
		++wwmi::g_state.frame;
		pending = wwmi::g_state.tracker.pick_pending(
			wwmi::g_state.tracker.config().hash_budget_per_frame);
		// M4: blend-clone PSOs whose source was destroyed (deferred here
		// because destroy_pipeline fires inside other API calls).
		retired_pipelines = std::move(wwmi::g_state.pending_pipeline_destroys);
		wwmi::g_state.pending_pipeline_destroys.clear();

		state_snapshot.reserve(pending.size());
		for (const wwmi::TrackedTexture &t : pending)
		{
			const auto it = wwmi::g_state.states.find(t.handle);
			state_snapshot.push_back(it != wwmi::g_state.states.end()
				? it->second
				: reshade::api::resource_usage::undefined);
		}
	}

	for (size_t i = 0; i < pending.size(); ++i)
	{
		if (state_snapshot[i] == reshade::api::resource_usage::undefined)
		{
			// Never saw a barrier for this texture (e.g. bundle-only use);
			// guessing states would corrupt the GPU timeline. Skip it.
			wwmi::mark_unsupported(pending[i].handle);
			continue;
		}

		// Present-time state gate: an actively-used state (render target,
		// depth write, UAV, present) means the game's in-flight command
		// lists may disagree with our snapshot; a wrong StateBefore barrier
		// hangs the device (DXGI_ERROR_DEVICE_HUNG). Requeue instead of
		// risking it; hashes settle once the texture is at rest.
		if (!wwmi::state_safe_for_readback(state_snapshot[i]))
		{
			std::lock_guard<std::mutex> guard(wwmi::g_state.lock);
			wwmi::g_state.tracker.requeue(pending[i].handle);
			continue;
		}

		wwmi::hash_via_readback(queue, pending[i], state_snapshot[i]);
	}

	// M2: spend the buffer readback budget (buffers are large; the budget
	// defaults to one per frame).
	std::vector<wwmi::TrackedBuffer> pending_buffers;
	{
		std::lock_guard<std::mutex> guard(wwmi::g_state.lock);
		pending_buffers = wwmi::g_state.buffers.pick_pending(
			wwmi::g_state.buffers.config().hash_budget_per_frame);
	}
	for (const wwmi::TrackedBuffer &snap : pending_buffers)
		wwmi::hash_buffer_via_readback(queue, snap);

	// M9: skeleton chain -- capture the game's skeleton CBs (snapshot from
	// this frame's component draws), merge into the mod's skeleton layout
	// and upload; the next frame's mod draws bind the result.
	wwmi::process_skeleton(queue);

	// M6: pooled-IB region probes. The region is copied out of the
	// pool, hashed with the ordinary 3DMigoto buffer formula (desc
	// ByteWidth = span) and compared against the mod's rule hash. A
	// verified view routes draws from then on like a learned dedicated
	// buffer (find_skip_rule); a mismatch marks the view failed and a
	// cooldown applies before it may be probed again (pool regions get
	// recycled between scenes).
	if (!wwmi::g_state.view_probes.empty())
	{
		std::vector<wwmi::RuntimeState::IndexProbe> probes;
		{
			std::lock_guard<std::mutex> guard(wwmi::g_state.lock);
			probes.swap(wwmi::g_state.view_probes);
		}
		for (const wwmi::RuntimeState::IndexProbe &p : probes)
			wwmi::verify_index_view_probe(queue, p);
	}

	// M6 diagnostics: periodically dump the draw-signature tallies of
	// interesting views -- the REAL section layout the game renders
	// with, independent of mod.ini's window guesses.
	if (wwmi::g_state.frame % 600 == 0)
	{
		std::lock_guard<std::mutex> guard(wwmi::g_state.lock);
		size_t dumped = 0;
		for (const auto &kv : wwmi::g_state.index_views.views())
		{
			if (dumped >= wwmi::IndexViewTracker::k_diag_views)
				break;
			const wwmi::TrackedIndexView &v = kv.second;
			if (!v.interesting || v.sigs.empty())
				continue;
			++dumped;

			std::string line;
			char one[48];
			for (const auto &s : v.sigs)
			{
				sprintf_s(one, "(%u,%u)x%u ", s.first, s.count, s.hits);
				if (line.size() + strlen(one) > 900)
				{
					line += "...";
					break;
				}
				line += one;
			}
			wwmi::log::info("view draws: buffer %p off=%llu is=%u: %s",
				reinterpret_cast<void *>(v.handle),
				static_cast<unsigned long long>(v.offset),
				v.index_size, line.c_str());
		}
	}

	// M4: retire blend-clone PSOs whose source pipeline was destroyed.
	if (!retired_pipelines.empty())
	{
		for (const reshade::api::pipeline p : retired_pipelines)
			device->destroy_pipeline(p);
	}

	// Hashes learned above may have queued replacements.
	wwmi::process_activations(queue);
}

// ---------------------------------------------------------------------------
// Overlay panel (tab "WWMI-DX12" inside the ReShade overlay)
// ---------------------------------------------------------------------------

static void draw_overlay(reshade::api::effect_runtime *runtime)
{
	(void)runtime;
	using wwmi::g_state;
	using wwmi::RuntimeState;
	using wwmi::TextureTracker;
	using wwmi::BufferTracker;
	using wwmi::TextureOverrideRule;

	const bool attached = g_state.device.load() != nullptr;

	// Snapshot everything the panel shows under one lock; draw outside it.
	TextureTracker::Stats stats{};
	TextureTracker::StateCounts counts{};
	BufferTracker::Stats buffer_stats{};
	BufferTracker::StateCounts buffer_counts{};
	std::vector<RuntimeState::ActivationRecord> log_snapshot;
	std::vector<std::pair<std::string, RuntimeState::SkipRecord>> skip_snapshot;
	std::vector<std::array<char, 3 * 64>> rule_rows; // hash/section/filename
	std::string mods_root_snapshot;
	size_t live = 0, buffers_live = 0, session_pairs = 0, rules_total = 0;
	uint32_t budget = 0, buffer_budget = 0;
	uint64_t frame = 0, collisions = 0, draw_rules = 0, draws_skipped = 0;

	if (attached)
	{
		std::lock_guard<std::mutex> guard(g_state.lock);
		stats = g_state.tracker.stats();
		counts = g_state.tracker.count_by_state();
		buffer_stats = g_state.buffers.stats();
		buffer_counts = g_state.buffers.count_by_state();
		live = g_state.tracker.size();
		buffers_live = g_state.buffers.size();
		session_pairs = g_state.session.size();
		rules_total = g_state.index.size();
		collisions = g_state.index.collision_count();
		draw_rules = g_state.index.draw_rule_count();
		draws_skipped = g_state.draws_skipped.load();
		budget = g_state.tracker.config().hash_budget_per_frame;
		buffer_budget = g_state.buffers.config().hash_budget_per_frame;
		frame = g_state.frame;
		log_snapshot = g_state.activation_log;
		skip_snapshot.assign(g_state.skip_log.begin(), g_state.skip_log.end());
		mods_root_snapshot = g_state.mods_root;

		for (const TextureOverrideRule &rule : g_state.index.rules())
		{
			if (!rule.has_hash)
				continue;
			std::array<char, 3 * 64> row{};
			std::snprintf(row.data(), row.size(), "%08x  %s  %s", rule.hash,
				rule.section.c_str(),
				rule.texture_path.empty() ? "(no texture)"
					: rule.texture_path.filename().string().c_str());
			rule_rows.push_back(row);
		}
	}

	if (!attached)
	{
		ImGui::TextDisabled("waiting for a D3D12 device...");
		return;
	}

	ImGui::Text("frame %llu", static_cast<unsigned long long>(frame));

	ImGui::SeparatorText("Tracking");
	ImGui::Text("live %llu  pending %llu  learning %llu  done %llu  unsupported %llu",
		static_cast<unsigned long long>(live),
		static_cast<unsigned long long>(counts.pending),
		static_cast<unsigned long long>(counts.learning),
		static_cast<unsigned long long>(counts.done),
		static_cast<unsigned long long>(counts.unsupported));
	ImGui::Text("hashed %llu (initial %llu / readback %llu)  evicted %llu  rejected %llu",
		static_cast<unsigned long long>(stats.hashed),
		static_cast<unsigned long long>(g_state.initialdata_hashes.load()),
		static_cast<unsigned long long>(g_state.readback_hashes.load()),
		static_cast<unsigned long long>(stats.evicted),
		static_cast<unsigned long long>(stats.rejected));
	ImGui::TextDisabled("readback budget %u texture(s) per frame", budget);

	ImGui::SeparatorText("Overrides");
	ImGui::Text("hash matches %llu  active replacements %llu",
		static_cast<unsigned long long>(g_state.hash_matches.load()),
		static_cast<unsigned long long>(g_state.active_replacements.load()));
	ImGui::Text("desc updates %llu  desc copies %llu  overwrites applied %llu",
		static_cast<unsigned long long>(g_state.descriptor_updates.load()),
		static_cast<unsigned long long>(g_state.descriptor_copies.load()),
		static_cast<unsigned long long>(g_state.overwrites_applied.load()));

	ImGui::SeparatorText("Draw interception");
	ImGui::Text("draw rules %llu  draws skipped %llu",
		static_cast<unsigned long long>(draw_rules),
		static_cast<unsigned long long>(draws_skipped));
	ImGui::Text("buffers: live %llu  pending %llu  learning %llu  done %llu  unsupported %llu",
		static_cast<unsigned long long>(buffers_live),
		static_cast<unsigned long long>(buffer_counts.pending),
		static_cast<unsigned long long>(buffer_counts.learning),
		static_cast<unsigned long long>(buffer_counts.done),
		static_cast<unsigned long long>(buffer_counts.unsupported));
	ImGui::Text("buffer hashes %llu (readback)  binds %llu  evicted %llu  rejected %llu",
		static_cast<unsigned long long>(g_state.buffer_readback_hashes.load()),
		static_cast<unsigned long long>(buffer_stats.binds),
		static_cast<unsigned long long>(buffer_stats.evicted),
		static_cast<unsigned long long>(buffer_stats.rejected));
	ImGui::TextDisabled("buffer readback budget %u per frame (skips engage once hashes are learned)",
		buffer_budget);

	if (!skip_snapshot.empty())
	{
		ImGui::SeparatorText("Active skips");
		for (const auto &[section, rec] : skip_snapshot)
			ImGui::BulletText("%s: %llu draw(s), last @ frame %llu", section.c_str(),
				static_cast<unsigned long long>(rec.count),
				static_cast<unsigned long long>(rec.frame));
	}

	ImGui::SeparatorText("Mods");
	ImGui::TextDisabled("mods root: %s", mods_root_snapshot.c_str());
	ImGui::Text("%llu rule(s)  %llu hash collision(s)  session cache %llu pairing(s)",
		static_cast<unsigned long long>(rules_total),
		static_cast<unsigned long long>(collisions),
		static_cast<unsigned long long>(session_pairs));

	// ---- Mod manager: per-mod enable toggle, persisted for the next
	// launch (no hot reload: rules, runtimes and resources are wired
	// through immutable indexes after load_mods()).
	ImGui::SeparatorText("Mod manager");
	ImGui::TextDisabled("toggles are saved immediately and take effect on the next launch");

	{
		std::vector<RuntimeState::ModCatalogEntry> catalog;
		{
			std::lock_guard<std::mutex> guard(g_state.lock);
			catalog = g_state.mod_catalog;
		}

		if (catalog.empty())
		{
			ImGui::TextDisabled("no mods found under the mods root");
		}
		else if (ImGui::BeginTable("modmanager", 2, ImGuiTableFlags_Borders))
		{
			ImGui::TableSetupColumn("mod");
			ImGui::TableSetupColumn("this session");
			ImGui::TableHeadersRow();

			for (const RuntimeState::ModCatalogEntry &entry : catalog)
			{
				ImGui::TableNextRow();
				ImGui::TableNextColumn();

				bool enabled = entry.enabled;
				if (ImGui::Checkbox(entry.name.c_str(), &enabled))
				{
					std::lock_guard<std::mutex> guard(g_state.lock);
					for (RuntimeState::ModCatalogEntry &live : g_state.mod_catalog)
						if (live.name == entry.name)
							live.enabled = enabled;
					write_disabled_mods(g_state.mod_catalog);
					wwmi::log::info("mod manager: '%s' %s (applies on next launch)",
						entry.name.c_str(), enabled ? "enabled" : "disabled");
				}

				ImGui::TableNextColumn();
				if (!entry.enabled)
					ImGui::TextUnformatted(entry.loaded
						? "loaded (skipped after restart)"
						: "skipped");
				else
					ImGui::TextUnformatted(entry.loaded
						? "loaded"
						: "mod.ini failed to load");
			}
			ImGui::EndTable();
		}
	}

	if (!log_snapshot.empty())
	{
		ImGui::SeparatorText("Recent activations");
		for (const RuntimeState::ActivationRecord &rec : log_snapshot)
			ImGui::Text("%p  %s  (%u slot rewrite(s) @ frame %llu)",
				reinterpret_cast<const void *>(rec.resource), rec.texture.c_str(),
				rec.slot_rewrites, static_cast<unsigned long long>(rec.frame));
	}

	if (!rule_rows.empty())
	{
		ImGui::SeparatorText("Loaded rules");
		for (const auto &row : rule_rows)
			ImGui::BulletText("%s", row.data());
	}

	if (ImGui::SmallButton("reset learned hashes"))
	{
		std::lock_guard<std::mutex> guard(g_state.lock);
		g_state.tracker.reset_hashes();
		g_state.buffers.reset_hashes(); // M2: relearn VB/IB hashes too
	}
	ImGui::SameLine(0.0f, -1.0f);
	if (ImGui::SmallButton("save session cache"))
	{
		std::lock_guard<std::mutex> guard(g_state.lock);
		wwmi::save_session();
	}

	// ---- M3: script runtimes (locks taken per mod, outside <lock>) ----
	ImGui::SeparatorText("Scripting");
	ImGui::Text("lists run %llu  draws re-issued %llu  draws blocked %llu  keys paused: %s",
		static_cast<unsigned long long>(g_state.script_lists_executed.load()),
		static_cast<unsigned long long>(g_state.script_draws_issued.load()),
		static_cast<unsigned long long>(g_state.script_draws_blocked.load()),
		g_state.overlay_open.load() ? "yes (overlay open)" : "no");

	for (const auto &mrt : g_state.runtimes)
	{
		std::vector<std::pair<std::string, float>> vars;
		std::vector<std::string> key_desc;
		{
			std::lock_guard<std::mutex> guard(mrt->lock);
			vars = mrt->rt.vars_snapshot();
			key_desc.reserve(mrt->rt.keys().size());
			for (const wwmi::KeyBinding &kb : mrt->rt.keys())
			{
				char buf[96];
				std::snprintf(buf, sizeof(buf), "%s = %s (VK_%02X)",
					kb.var.c_str(), kb.key_name.c_str(), kb.vk);
				key_desc.emplace_back(buf);
			}
		}

		if (ImGui::TreeNode("%s (%llu var, %llu key)",
				mrt->rt.mod_name().c_str(),
				static_cast<unsigned long long>(vars.size()),
				static_cast<unsigned long long>(key_desc.size())))
		{
			for (const std::string &k : key_desc)
				ImGui::BulletText("%s", k.c_str());

			if (ImGui::BeginTable("vars", 2, ImGuiTableFlags_Borders))
			{
				for (const auto &[name, value] : vars)
				{
					ImGui::TableNextRow();
					ImGui::TableNextColumn();
					ImGui::TextUnformatted(name.c_str());
					ImGui::TableNextColumn();
					ImGui::Text("%.6g", value);
				}
				ImGui::EndTable();
			}
			ImGui::TreePop();
		}
	}
}

// ReShade overlay visibility: mod key bindings pause while it is open.
static bool on_open_overlay(reshade::api::effect_runtime *runtime, bool open,
	reshade::api::input_source source)
{
	(void)runtime;
	(void)source;
	wwmi::g_state.overlay_open.store(open);
	return false; // never block the state change
}

// ---------------------------------------------------------------------------
// Addon entry point
// ---------------------------------------------------------------------------

BOOL WINAPI DllMain(HMODULE hModule, DWORD fdwReason, LPVOID lpReserved)
{
	switch (fdwReason)
	{
	case DLL_PROCESS_ATTACH:
		DisableThreadLibraryCalls(hModule);

		if (!reshade::register_addon(hModule))
			return FALSE; // ReShade not present or API version mismatch

		reshade::register_event<reshade::addon_event::init_device>(&on_init_device);
		reshade::register_event<reshade::addon_event::destroy_device>(&on_destroy_device);
		reshade::register_event<reshade::addon_event::init_command_queue>(&on_init_command_queue);
		reshade::register_event<reshade::addon_event::destroy_command_queue>(&on_destroy_command_queue);
		reshade::register_event<reshade::addon_event::init_resource>(&on_init_resource);
		reshade::register_event<reshade::addon_event::destroy_resource>(&on_destroy_resource);
		reshade::register_event<reshade::addon_event::init_resource_view>(&on_init_resource_view);
		reshade::register_event<reshade::addon_event::barrier>(&on_barrier);
		reshade::register_event<reshade::addon_event::update_descriptor_tables>(&on_update_descriptor_tables);
		reshade::register_event<reshade::addon_event::copy_descriptor_tables>(&on_copy_descriptor_tables);
		reshade::register_event<reshade::addon_event::present>(&on_present);

		// M2: draw interception
		reshade::register_event<reshade::addon_event::destroy_command_list>(&on_destroy_command_list);
		reshade::register_event<reshade::addon_event::bind_index_buffer>(&on_bind_index_buffer);
		reshade::register_event<reshade::addon_event::bind_vertex_buffers>(&on_bind_vertex_buffers);
		reshade::register_event<reshade::addon_event::draw>(&on_draw);
		reshade::register_event<reshade::addon_event::draw_indexed>(&on_draw_indexed);

		// M4: pipeline tracking + ShaderOverride
		reshade::register_event<reshade::addon_event::init_pipeline>(&on_init_pipeline);
		reshade::register_event<reshade::addon_event::destroy_pipeline>(&on_destroy_pipeline);
		reshade::register_event<reshade::addon_event::bind_pipeline>(&on_bind_pipeline);

		// M8: root signature tracking (skeleton system)
		reshade::register_event<reshade::addon_event::init_pipeline_layout>(&on_init_pipeline_layout);
		reshade::register_event<reshade::addon_event::destroy_pipeline_layout>(&on_destroy_pipeline_layout);
		reshade::register_event<reshade::addon_event::push_descriptors>(&on_push_descriptors);
		reshade::register_event<reshade::addon_event::bind_descriptor_tables>(&on_bind_descriptor_tables);

		reshade::register_overlay("WWMI-DX12", &draw_overlay);
		break;

	case DLL_PROCESS_DETACH:
		if (lpReserved == nullptr) // Process is terminating: do not touch the loader lock
		{
			reshade::unregister_event<reshade::addon_event::init_device>(&on_init_device);
			reshade::unregister_event<reshade::addon_event::destroy_device>(&on_destroy_device);
			reshade::unregister_event<reshade::addon_event::init_command_queue>(&on_init_command_queue);
			reshade::unregister_event<reshade::addon_event::destroy_command_queue>(&on_destroy_command_queue);
			reshade::unregister_event<reshade::addon_event::init_resource>(&on_init_resource);
			reshade::unregister_event<reshade::addon_event::destroy_resource>(&on_destroy_resource);
			reshade::unregister_event<reshade::addon_event::init_resource_view>(&on_init_resource_view);
			reshade::unregister_event<reshade::addon_event::barrier>(&on_barrier);
			reshade::unregister_event<reshade::addon_event::update_descriptor_tables>(&on_update_descriptor_tables);
			reshade::unregister_event<reshade::addon_event::copy_descriptor_tables>(&on_copy_descriptor_tables);
			reshade::unregister_event<reshade::addon_event::present>(&on_present);
			reshade::unregister_event<reshade::addon_event::reshade_open_overlay>(&on_open_overlay);

			// M2: draw interception
			reshade::unregister_event<reshade::addon_event::destroy_command_list>(&on_destroy_command_list);
			reshade::unregister_event<reshade::addon_event::bind_index_buffer>(&on_bind_index_buffer);
			reshade::unregister_event<reshade::addon_event::bind_vertex_buffers>(&on_bind_vertex_buffers);
			reshade::unregister_event<reshade::addon_event::draw>(&on_draw);
			reshade::unregister_event<reshade::addon_event::draw_indexed>(&on_draw_indexed);

			// M4: pipeline tracking + ShaderOverride
			reshade::unregister_event<reshade::addon_event::init_pipeline>(&on_init_pipeline);
			reshade::unregister_event<reshade::addon_event::destroy_pipeline>(&on_destroy_pipeline);
			reshade::unregister_event<reshade::addon_event::bind_pipeline>(&on_bind_pipeline);

			// M8: root signature tracking (skeleton system)
			reshade::unregister_event<reshade::addon_event::init_pipeline_layout>(&on_init_pipeline_layout);
			reshade::unregister_event<reshade::addon_event::destroy_pipeline_layout>(&on_destroy_pipeline_layout);
			reshade::unregister_event<reshade::addon_event::push_descriptors>(&on_push_descriptors);
			reshade::unregister_event<reshade::addon_event::bind_descriptor_tables>(&on_bind_descriptor_tables);

			reshade::unregister_overlay("WWMI-DX12", &draw_overlay);

			reshade::unregister_addon(hModule);
		}
		break;
	}

	return TRUE;
}

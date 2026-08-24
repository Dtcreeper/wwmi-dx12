// WWMI-DX12 M3: ScriptRuntime tests (vars, keys, run chains, resources,
// draw execution) using a fake GPU bridge.
#include "script_runtime.hpp"
#include "buffer_tracker.hpp"
#include "mod_rules.hpp"
#include "test_framework.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>

using namespace wwmi;

// wwmi-tests does not pull in Windows.h.
#ifndef VK_RIGHT
#define VK_RIGHT 0x27
#endif

namespace
{
	// Fake bridge: records every operation; buffers are opaque handles.
	struct FakeBridge final : GpuBridge
	{
		struct Op
		{
			enum class Kind { create, destroy, bind_vb, bind_ib, draw, copy,
				apply_blend, revoke_blend } kind;
			uint64_t a = 0, b = 0, c = 0, d = 0, e = 0;
			uint32_t n = 0;
		};
		std::vector<Op> ops;
		uint64_t sizes[64] = {};
		uint64_t next_handle = 0x1000;
		// last applied blend config (checked by the M4 tests)
		bool blend_applied = false;
		BlendConfig last_blend{};
		uint64_t blend_token = 0x7; // nonzero: "override applied"

		uint64_t create_buffer(const void *data, uint64_t size, bool, bool) override
		{
			const uint64_t h = next_handle++;
			sizes[h - 0x1000] = size;
			Op op; op.kind = Op::Kind::create; op.a = h; op.b = size;
			op.c = data != nullptr ? 1 : 0;
			ops.push_back(op);
			return h;
		}
		void destroy_buffer(uint64_t h) override
		{
			Op op; op.kind = Op::Kind::destroy; op.a = h;
			ops.push_back(op);
		}
		bool buffer_size(uint64_t h, uint64_t *size) override
		{
			if (h < 0x1000 || h >= next_handle)
				return false;
			*size = sizes[h - 0x1000];
			return true;
		}
		void bind_vb(uint32_t slot, uint64_t res, uint64_t off, uint32_t stride) override
		{
			Op op; op.kind = Op::Kind::bind_vb; op.n = slot; op.a = res;
			op.b = off; op.c = stride;
			ops.push_back(op);
		}
		void bind_ib(uint64_t res, uint64_t off, uint32_t index_size) override
		{
			Op op; op.kind = Op::Kind::bind_ib; op.a = res; op.b = off;
			op.c = index_size;
			ops.push_back(op);
		}
		void draw_indexed(uint32_t count, uint32_t first, int32_t base, uint32_t inst) override
		{
			Op op; op.kind = Op::Kind::draw; op.a = count; op.b = first;
			op.c = static_cast<uint64_t>(static_cast<int64_t>(base)); op.d = inst;
			ops.push_back(op);
		}
		void copy_buffer(uint64_t src, uint64_t so, uint64_t dst, uint64_t doff, uint64_t size) override
		{
			Op op; op.kind = Op::Kind::copy; op.a = src; op.b = so; op.c = dst;
			op.d = doff; op.e = size;
			ops.push_back(op);
		}
		uint64_t apply_blend(const BlendConfig &cfg) override
		{
			Op op; op.kind = Op::Kind::apply_blend;
			op.a = cfg.has_color ? 1 : 0;
			op.b = cfg.has_alpha ? 1 : 0;
			op.c = cfg.has_factors ? 1 : 0;
			ops.push_back(op);
			blend_applied = true;
			last_blend = cfg;
			return blend_token;
		}
		void revoke_blend(uint64_t token) override
		{
			Op op; op.kind = Op::Kind::revoke_blend; op.a = token;
			ops.push_back(op);
		}

		size_t count(Op::Kind k) const
		{
			size_t n = 0;
			for (const Op &o : ops)
				if (o.kind == k)
					++n;
			return n;
		}
	};

	// Writes a temp mod tree and loads it.
	struct TempMod
	{
		std::filesystem::path dir;
		std::filesystem::path ini;

		TempMod(const std::string &name, const std::string &ini_content,
			const std::vector<std::pair<std::string, std::string>> &files = {})
		{
			dir = std::filesystem::temp_directory_path() /
				("wwmi-test-" + name);
			std::error_code ec;
			std::filesystem::remove_all(dir, ec);
			std::filesystem::create_directories(dir, ec);
			ini = dir / "mod.ini";
			{ std::ofstream f(ini, std::ios::binary); f << ini_content; }
			for (const auto &[rel, bytes] : files)
			{
				const std::filesystem::path p = dir / rel;
				std::filesystem::create_directories(p.parent_path(), ec);
				std::ofstream f(p, std::ios::binary);
				f.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
			}
		}
		~TempMod() { std::error_code ec; std::filesystem::remove_all(dir, ec); }
	};
}

WWMI_TEST(runtime_constants_and_persist_flags)
{
	TempMod mod("consts", R"INI(
[Constants]
global $required_wwmi_version = 0.96
global $mesh_vertex_count = 141645
global persist $mega = 0
global persist $cloth = 0
$plain = 3
global $derived = $plain + 1
)INI");

	ScriptRuntime rt;
	EXPECT(rt.load(mod.ini));
	EXPECT_EQ(rt.var_count(), static_cast<size_t>(6));
	float v = 0;
	EXPECT(rt.get_var("mesh_vertex_count", v) && v == 141645.0f);
	EXPECT(rt.get_var("derived", v) && v == 4.0f);
	// persist round-trip only touches flagged names
	rt.set_var("mega", 1);
	rt.set_var("plain", 9);
	rt.save_persist(mod.dir / "persist.ini");
	ScriptRuntime rt2;
	rt2.load(mod.ini);
	rt2.load_persist(mod.dir / "persist.ini");
	EXPECT(rt2.get_var("mega", v) && v == 1.0f);
	EXPECT(rt2.get_var("plain", v) && v == 3.0f); // untouched by overlay
}

WWMI_TEST(runtime_key_cycle_and_condition)
{
	TempMod mod("keys", R"INI(
[Constants]
global persist $leg = 0
global $object_detected = 0
[Keyleg]
condition = $object_detected
key = VK_RIGHT
type = cycle
$leg = 0,1,2
)INI");

	ScriptRuntime rt;
	EXPECT(rt.load(mod.ini));
	EXPECT_EQ(rt.keys().size(), static_cast<size_t>(1));

	// Condition blocks the cycle while the object is undetected.
	rt.on_key(VK_RIGHT, true);
	float v = -1;
	rt.get_var("leg", v);
	EXPECT_EQ(v, 0.0f);
	rt.on_key(VK_RIGHT, false); // release

	rt.set_var("object_detected", 1);
	rt.on_key(VK_RIGHT, true);
	rt.get_var("leg", v);
	EXPECT_EQ(v, 1.0f);
	rt.on_key(VK_RIGHT, false);
	rt.on_key(VK_RIGHT, true); // 1 -> 2
	rt.get_var("leg", v);
	EXPECT_EQ(v, 2.0f);
	rt.on_key(VK_RIGHT, false);
	rt.on_key(VK_RIGHT, true); // 2 -> wraps to 0
	rt.get_var("leg", v);
	EXPECT_EQ(v, 0.0f);
}

WWMI_TEST(runtime_run_chain_and_post_assign)
{
	TempMod mod("runchain", R"INI(
[Constants]
global $state_id = 1
global $marks = 0
[CommandListInner]
$marks = $marks + 10
[CommandListOuter]
run = CommandListInner
post $state_id = 7
$marks = $marks + 1
)INI");

	ScriptRuntime rt;
	EXPECT(rt.load(mod.ini));
	rt.run_list("commandlistouter", nullptr, nullptr);

	float v = 0;
	EXPECT(rt.get_var("marks", v) && v == 11.0f);   // inner ran first
	EXPECT(rt.get_var("state_id", v) && v == 7.0f); // post applied last
}

WWMI_TEST(runtime_draw_execution_and_bindings)
{
	TempMod mod("draws", R"INI(
[CommandListOverrideSharedResources]
ResourceBypassVB0 = ref vb0
ib = ResourceIndexBuffer
vb0 = ResourcePositionBuffer
vb1 = ResourceVectorBuffer
drawindexed = 17970, 0, 0
drawindexed = 22299, 161709, 0
vb0 = ref ResourceBypassVB0
)INI",
		{
			{"Meshes/Index.buf", std::string(40, '\x01')},
			{"Meshes/Position.buf", std::string(30, '\x02')},
			{"Meshes/Vector.buf", std::string(20, '\x03')},
		});

	ScriptRuntime rt;
	EXPECT(rt.load(mod.ini));

	// The list also needs the resource definitions.
	TempMod res("draws2", R"INI(
[ResourceIndexBuffer]
type = Buffer
format = DXGI_FORMAT_R32_UINT
stride = 12
filename = Meshes/Index.buf

[ResourcePositionBuffer]
type = Buffer
format = DXGI_FORMAT_R32G32B32_FLOAT
stride = 12
filename = Meshes/Position.buf

[ResourceVectorBuffer]
type = Buffer
format = DXGI_FORMAT_R8G8B8A8_SNORM
stride = 8
filename = Meshes/Vector.buf
)INI");
	// (merge: reload with both sections in one file)
	{
		std::ofstream f(mod.ini, std::ios::app | std::ios::binary);
		std::ifstream g(res.ini);
		f << g.rdbuf();
	}

	ScriptRuntime rt2;
	EXPECT(rt2.load(mod.ini));

	// Game IA state: vb0 = game buffer 0x99, stride 12; ib = 0x88 (4-byte).
	IaState ia;
	ia.vbs[0] = { 0x99, 0, 12 };
	ia.vb_valid_mask = 1u;
	ia.ib_buffer = 0x88;
	ia.ib_offset = 0;
	ia.ib_index_size = 4;

	FakeBridge bridge;
	rt2.run_list("commandlistoverridesharedresources", &bridge, &ia);

	using K = FakeBridge::Op::Kind;
	// 3 file buffers created (index, position, vector)
	EXPECT_EQ(bridge.count(K::create), static_cast<size_t>(3));
	// capture did not create anything but aliased vb0 = 0x99
	// vb1 bind, ib bind, vb0 bind (custom), restore vb0 bind (capture)
	EXPECT_EQ(bridge.count(K::bind_vb), static_cast<size_t>(3));
	EXPECT_EQ(bridge.count(K::bind_ib), static_cast<size_t>(1));
	EXPECT_EQ(bridge.count(K::draw), static_cast<size_t>(2));

	// drawindexed order and parameters
	const FakeBridge::Op *d0 = nullptr, *d1 = nullptr;
	for (const FakeBridge::Op &o : bridge.ops)
	{
		if (o.kind != K::draw)
			continue;
		if (d0 == nullptr)
			d0 = &o;
		else
			d1 = &o;
	}
	EXPECT(d0 != nullptr && d0->a == 17970 && d0->b == 0);
	EXPECT(d1 != nullptr && d1->a == 22299 && d1->b == 161709);

	// The restored vb0 comes from the capture (0x99, stride 12).
	const FakeBridge::Op *restore = nullptr;
	for (const FakeBridge::Op &o : bridge.ops)
		if (o.kind == K::bind_vb && o.a == 0x99)
			restore = &o;
	EXPECT(restore != nullptr && restore->c == 12 && restore->n == 0);

	// IB bind uses the format-derived 4-byte index size.
	for (const FakeBridge::Op &o : bridge.ops)
		if (o.kind == K::bind_ib)
			EXPECT_EQ(o.c, static_cast<uint64_t>(4));
}

WWMI_TEST(runtime_if_blocks_and_resource_null)
{
	TempMod mod("resnull", R"INI(
[Constants]
global $mod_enabled = 1
[CommandListBody]
if ResourceMergedSkeleton !== null
drawindexed = 100, 0, 0
endif
if ResourceGhost == null
drawindexed = 200, 0, 0
endif
if $mod_enabled == 0
drawindexed = 400, 0, 0
endif
)INI");

	ScriptRuntime rt;
	EXPECT(rt.load(mod.ini));

	FakeBridge bridge;
	rt.run_list("commandlistbody", &bridge, nullptr);
	// ResourceMergedSkeleton undefined -> null -> first draw skipped,
	// ResourceGhost undefined -> null -> second draw issued,
	// $mod_enabled == 1 -> third skipped.
	EXPECT_EQ(bridge.count(FakeBridge::Op::Kind::draw), static_cast<size_t>(1));
	for (const FakeBridge::Op &o : bridge.ops)
		if (o.kind == FakeBridge::Op::Kind::draw)
			EXPECT_EQ(o.a, static_cast<uint64_t>(200));
}

WWMI_TEST(runtime_resource_ref_copy_null)
{
	TempMod mod("resops", R"INI(
[ResourceZero]
type = RWBuffer
format = R32G32B32A32_FLOAT
array = 64

[ResourceAlias]
[ResourceFromGame]
)INI");

	ScriptRuntime rt;
	EXPECT(rt.load(mod.ini));

	IaState ia;
	ia.ib_buffer = 0xABC;
	ia.ib_offset = 16;
	ia.ib_index_size = 2;

	FakeBridge bridge;

	// synthesized: 64 * 16 bytes zero buffer
	ScriptResource *zero = rt.find_resource("zero");
	EXPECT(zero != nullptr);
	uint64_t h = 0;
	// public path: bind_ib to force creation
	TempMod mod2("resops2", R"INI(
[ResourceZero]
type = RWBuffer
format = R32G32B32A32_FLOAT
array = 64
[CommandListMake]
ResourceFromGame = ref ib
ib = ResourceZero
ResourceAlias = ref ResourceFromGame
ResourceAlias = null
)INI");
	ScriptRuntime rt2;
	EXPECT(rt2.load(mod2.ini));
	rt2.run_list("commandlistmake", &bridge, &ia);

	using K = FakeBridge::Op::Kind;
	EXPECT_EQ(bridge.count(K::create), static_cast<size_t>(1)); // zero only
	EXPECT_EQ(bridge.count(K::bind_ib), static_cast<size_t>(1));
	// 'ResourceAlias = null' drops the alias but must NOT destroy the
	// aliased GAME buffer (0xABC) -- game-capture refs are never owned.
	EXPECT_EQ(bridge.count(K::destroy), static_cast<size_t>(0));
	EXPECT(!rt2.resource_alive("ResourceAlias"));
	bool alive_chk = rt2.resource_alive("ResourceFromGame");
	EXPECT(alive_chk); // underlying capture stays live
	for (const FakeBridge::Op &o : bridge.ops)
	{
		if (o.kind == K::create)
			EXPECT_EQ(o.b, static_cast<uint64_t>(64 * 16));
		if (o.kind == K::bind_ib)
			EXPECT_EQ(o.c, static_cast<uint64_t>(4)); // R32G32B32A32 -> index_size 0 -> default 4
	}
	(void)h; (void)zero;
}

WWMI_TEST(runtime_namespaced_noops)
{
	TempMod mod("namespaced", R"INI(
[CommandListRegisterMod]
$\WWMIv1\required_wwmi_version = 0.96
Resource\WWMIv1\ModName = ref ResourceModName
run = CommandList\WWMIv1\RegisterMod
$mod_id = $\WWMIv1\mod_id
)INI");

	ScriptRuntime rt;
	EXPECT(rt.load(mod.ini));
	FakeBridge bridge;
	rt.run_list("commandlistregistermod", &bridge, nullptr);
	// nothing created, nothing drawn, no crash; only the local var landed
	EXPECT_EQ(bridge.ops.size(), static_cast<size_t>(0));
	float v = -1;
	EXPECT(rt.get_var("mod_id", v));
	EXPECT_EQ(v, 0.0f); // undefined framework var reads 0
	EXPECT(!rt.get_var("wwmi1-missing", v));
}

WWMI_TEST(runtime_present_merges_sections)
{
	TempMod mod("present", R"INI(
[Present]
$hb = $attack
$attack = 0
[Present]
$hc = $attack1
$attack1 = 0
)INI");

	ScriptRuntime rt;
	EXPECT(rt.load(mod.ini));
	rt.set_var("attack", 1);
	rt.set_var("attack1", 1);
	rt.run_present(nullptr);
	float v = 0;
	EXPECT(rt.get_var("hb", v) && v == 1.0f);
	EXPECT(rt.get_var("hc", v) && v == 1.0f);
	EXPECT(rt.get_var("attack", v) && v == 0.0f);
	EXPECT(rt.get_var("attack1", v) && v == 0.0f);
}

// M4: 'blend'/'blendalpha'/'blend_factor[N]' merge into the active
// config and wrap every drawindexed with apply/revoke.
WWMI_TEST(runtime_blend_wraps_drawindexed)
{
	TempMod mod("blend", R"INI(
[CommandListTransparency]
blend = ADD INV_BLEND_FACTOR BLEND_FACTOR
blendalpha = ADD SRC_ALPHA INV_SRC_ALPHA
blend_factor[0] = 0.85
drawindexed = 100, 0, 0
)INI");

	ScriptRuntime rt;
	EXPECT(rt.load(mod.ini));

	FakeBridge bridge;
	EXPECT(rt.run_list("CommandListTransparency", &bridge, nullptr));

	// exactly one apply and one revoke around the draw
	EXPECT_EQ(bridge.count(FakeBridge::Op::Kind::apply_blend),
		static_cast<size_t>(1));
	EXPECT_EQ(bridge.count(FakeBridge::Op::Kind::revoke_blend),
		static_cast<size_t>(1));
	EXPECT_EQ(bridge.count(FakeBridge::Op::Kind::draw), static_cast<size_t>(1));

	EXPECT(bridge.blend_applied);
	EXPECT(bridge.last_blend.has_color);
	EXPECT_EQ(bridge.last_blend.op, 0);          // ADD
	EXPECT_EQ(bridge.last_blend.src, 11);        // INV_BLEND_FACTOR
	EXPECT_EQ(bridge.last_blend.dst, 10);        // BLEND_FACTOR
	EXPECT(bridge.last_blend.has_alpha);
	EXPECT_EQ(bridge.last_blend.alpha_src, 6);   // SRC_ALPHA
	EXPECT_EQ(bridge.last_blend.alpha_dst, 7);   // INV_SRC_ALPHA
	EXPECT(bridge.last_blend.has_factors);
	EXPECT_EQ(bridge.last_blend.factor_mask, 1u);
	EXPECT(bridge.last_blend.factors[0] == 0.85f);

	// ordering: apply -> draw -> revoke
	size_t apply_at = SIZE_MAX, draw_at = SIZE_MAX, revoke_at = SIZE_MAX;
	for (size_t i = 0; i < bridge.ops.size(); ++i)
	{
		if (bridge.ops[i].kind == FakeBridge::Op::Kind::apply_blend) apply_at = i;
		else if (bridge.ops[i].kind == FakeBridge::Op::Kind::draw) draw_at = i;
		else if (bridge.ops[i].kind == FakeBridge::Op::Kind::revoke_blend) revoke_at = i;
	}
	EXPECT(apply_at < draw_at && draw_at < revoke_at);

	// revoke carries the token returned by apply
	EXPECT_EQ(bridge.ops[revoke_at].a, bridge.blend_token);
}

// M4: blend state does not leak past one top-level execution.
WWMI_TEST(runtime_blend_scoped_to_one_list)
{
	TempMod mod("blendscope", R"INI(
[CommandListWithBlend]
blend = ADD ONE ONE
drawindexed = 10, 0, 0
[CommandListNoBlend]
drawindexed = 20, 0, 0
)INI");

	ScriptRuntime rt;
	EXPECT(rt.load(mod.ini));

	FakeBridge bridge;
	EXPECT(rt.run_list("CommandListWithBlend", &bridge, nullptr));
	EXPECT(rt.run_list("CommandListNoBlend", &bridge, nullptr));

	EXPECT_EQ(bridge.count(FakeBridge::Op::Kind::apply_blend),
		static_cast<size_t>(1));
	EXPECT_EQ(bridge.count(FakeBridge::Op::Kind::revoke_blend),
		static_cast<size_t>(1));
	EXPECT_EQ(bridge.count(FakeBridge::Op::Kind::draw), static_cast<size_t>(2));
}

// M4: [ShaderOverride*] bodies parse as runnable lists (hash matching is
// the addon's job; the runtime only executes).
WWMI_TEST(runtime_shader_override_body_runs)
{
	TempMod mod("so", R"INI(
[ShaderOverrideAttackLatch]
hash = 525e619cd71fb4b0
$attack = 1
)INI");

	ScriptRuntime rt;
	EXPECT(rt.load(mod.ini));
	// normalize_section_name strips 'shaderoverride'
	float v = 0;
	EXPECT(!rt.get_var("attack", v) || v == 0.0f);

	FakeBridge bridge;
	EXPECT(rt.run_list("ShaderOverrideAttackLatch", &bridge, nullptr));
	EXPECT(rt.get_var("attack", v) && v == 1.0f);
}

// M5 end-to-end: the full body-IB replacement chain against a synthetic
// large index buffer, exactly like the Lynae component split:
//
//   bind -> track (large IB) -> pick_pending prioritizes it
//        -> readback learns the 3DMigoto hash -> find_skip_rule routes
//        the draw by first_index -> run_override rebinds the mod mesh
//        and issues the replacement drawindexed.
WWMI_TEST(e2e_ib_hash_learn_match_and_replace)
{
	// 1. Synthesize a "game body IB" (>= 256 KB so it jumps the pick
	//    queue) and derive the hash a 3DMigoto mod would reference.
	constexpr uint64_t kIbBytes = 300ull * 1024;
	std::vector<uint8_t> ib_data(kIbBytes);
	for (uint64_t i = 0; i + 4 <= kIbBytes; i += 4)
	{
		const uint32_t idx = static_cast<uint32_t>((i * 31 + 7) & 0xffffffff);
		std::memcpy(&ib_data[i], &idx, 4);
	}
	const uint32_t data_hash = calc_buffer_data_hash(ib_data.data(), kIbBytes);
	const uint32_t ib_hash = calc_buffer_hash(data_hash, kIbBytes, BufferRole::index);

	char hash_hex[16];
	std::snprintf(hash_hex, sizeof(hash_hex), "%08x", ib_hash);

	// 2. Mod mirroring the Lynae pattern: one shared rebind list, two
	//    component rules on the SAME IB hash split by first_index.
	const std::string ini = std::string(R"INI(
[ResourceIndexBuffer]
type = Buffer
format = DXGI_FORMAT_R32_UINT
stride = 4
filename = Meshes/Index.buf

[CommandListOverrideSharedResources]
ib = ResourceIndexBuffer

[TextureOverrideComponent0]
hash = )INI" + std::string(hash_hex) + R"INI(
match_first_index = 0
match_index_count = 17970
handling = skip
run = CommandListOverrideSharedResources
drawindexed = 17970, 0, 0

[TextureOverrideComponent1]
hash = )INI" + std::string(hash_hex) + R"INI(
match_first_index = 17970
match_index_count = 45636
handling = skip
run = CommandListOverrideSharedResources
drawindexed = 45636, 17970, 0
)INI");

	const std::string dummy_buf(1024, '\x11');
	TempMod mod("e2e-ib", ini, { { "Meshes/Index.buf", dummy_buf } });

	// 3. Rule side: ModRules -> HashIndex (as addon_main builds it).
	ModRules rules_mod;
	EXPECT(load_mod_rules(mod.ini, rules_mod));
	EXPECT(!rules_mod.overrides.empty());
	HashIndex index;
	index.build(rules_mod.overrides);
	EXPECT(index.has_draw_rules());

	// 4. Script side.
	ScriptRuntime rt;
	EXPECT(rt.load(mod.ini));

	// 5. Bind-time tracking: an older pending VB must NOT outrank the
	//    large IB in the readback budget.
	constexpr uint64_t kIbHandle = 0xA11;
	constexpr uint64_t kVbHandle = 0xB22;
	BufferTracker t;
	t.track(kVbHandle, 4096, BufferRole::vertex, 1);
	t.track(kIbHandle, kIbBytes, BufferRole::index, 2);
	const std::vector<TrackedBuffer> picked = t.pick_pending(1);
	EXPECT_EQ(picked.size(), static_cast<size_t>(1));
	EXPECT_EQ(picked[0].handle, kIbHandle);

	IaState ia;
	ia.bind_index_buffer(kIbHandle, 0, 4);
	ia.bind_vertex_buffers(0, 1, std::vector<uint64_t>{ kVbHandle }.data(),
		nullptr, nullptr);

	DrawCallInfo call;
	call.index_count = 17970;
	call.first_index = 0;
	call.instance_count = 1;

	// 6. Not learned yet: the draw passes through untouched.
	EXPECT(find_skip_rule(index, t, nullptr, ia, call, true) == nullptr);

	// 7. Readback completes: the learned hash engages the rules.
	t.set_hash(kIbHandle, data_hash, ib_hash);
	const TextureOverrideRule *r0 = find_skip_rule(index, t, nullptr, ia, call, true);
	EXPECT(r0 != nullptr);
	EXPECT_EQ(r0->section, std::string("TextureOverrideComponent0"));

	call.index_count = 45636;
	call.first_index = 17970;
	const TextureOverrideRule *r1 = find_skip_rule(index, t, nullptr, ia, call, true);
	EXPECT(r1 != nullptr);
	EXPECT_EQ(r1->section, std::string("TextureOverrideComponent1"));

	// A draw outside every component window runs unmodified.
	call.index_count = 12324;
	call.first_index = 63606;
	EXPECT(find_skip_rule(index, t, nullptr, ia, call, true) == nullptr);

	// 8. The replacement executes: mod IB created + bound, replacement
	//    draw issued with the component-1 window, original skipped.
	FakeBridge bridge;
	call.index_count = 45636;
	call.first_index = 17970;
	const bool skip = rt.run_override(r1->section, &bridge, &ia);
	EXPECT(skip);
	EXPECT_EQ(rt.draws_issued, static_cast<uint64_t>(1));
	EXPECT_EQ(bridge.count(FakeBridge::Op::Kind::create), static_cast<size_t>(1));
	EXPECT_EQ(bridge.count(FakeBridge::Op::Kind::bind_ib), static_cast<size_t>(1));
	const auto it_create = std::find_if(bridge.ops.begin(), bridge.ops.end(),
		[](const FakeBridge::Op &o) { return o.kind == FakeBridge::Op::Kind::create; });
	EXPECT(it_create != bridge.ops.end());
	EXPECT_EQ(it_create->b, static_cast<uint64_t>(dummy_buf.size()));
	const auto it_draw = std::find_if(bridge.ops.begin(), bridge.ops.end(),
		[](const FakeBridge::Op &o) { return o.kind == FakeBridge::Op::Kind::draw; });
	EXPECT(it_draw != bridge.ops.end());
	EXPECT_EQ(it_draw->a, static_cast<uint64_t>(45636));
	EXPECT_EQ(it_draw->b, static_cast<uint64_t>(17970));
}

// M6 end-to-end: pooled index-buffer region hashing. UE DX12 draws the
// body mesh from a view inside a shared 32 MB pool; hashing ONLY the
// rule-window region with desc ByteWidth = span reproduces the DX11
// dedicated-buffer hash the mod references.
WWMI_TEST(e2e_pooled_index_view_region_hash)
{
	// 1. The "original DX11 body IB": two tiling components covering
	//    indices [0, 63606).
	constexpr uint32_t kTotal = 63606;
	std::vector<uint8_t> body(size_t(kTotal) * 4);
	for (uint32_t i = 0; i < kTotal; ++i)
	{
		const uint32_t idx = i * 13 + 5;
		std::memcpy(&body[size_t(i) * 4], &idx, 4);
	}
	const uint32_t dedicated_hash =
		calc_buffer_hash(calc_buffer_data_hash(body.data(), body.size()),
			body.size(), BufferRole::index);

	char hash_hex[16];
	std::snprintf(hash_hex, sizeof(hash_hex), "%08x", dedicated_hash);

	// 2. Mod rules with the same shape (shared hash, window split).
	const std::string ini = std::string(R"INI(
[TextureOverrideComponent0]
hash = )INI" + std::string(hash_hex) + R"INI(
match_first_index = 0
match_index_count = 17970
handling = skip

[TextureOverrideComponent1]
hash = )INI" + std::string(hash_hex) + R"INI(
match_first_index = 17970
match_index_count = 45636
handling = skip
)INI");

	TempMod mod("e2e-pool", ini);

	ModRules rules_mod;
	EXPECT(load_mod_rules(mod.ini, rules_mod));
	HashIndex index;
	index.build(rules_mod.overrides);
	DrawWindowIndex windows;
	windows.build(rules_mod.overrides);
	EXPECT_EQ(windows.groups().size(), static_cast<size_t>(1));

	// 3. The game-side pool: 8 MB filler with the body at offset
	//    0x3000 -- the byte range a bind_index_buffer view exposes.
	constexpr uint64_t kPoolBytes = 8ull * 1024 * 1024;
	constexpr uint64_t kPoolHandle = 0x600000;
	constexpr uint64_t kViewOffset = 0x3000;
	std::vector<uint8_t> pool(kPoolBytes, 0xAB);
	std::memcpy(pool.data() + kViewOffset, body.data(), body.size());

	// 4. Probe arithmetic: the draw signature selects the group, the
	//    group bounds define the region.
	const DrawRuleGroup *grp = windows.find_by_draw(0, 17970);
	EXPECT(grp != nullptr);
	EXPECT_EQ(grp->min_first_index, 0u);
	EXPECT_EQ(grp->max_end_index, kTotal);

	const uint64_t region_offset = kViewOffset + uint64_t(grp->min_first_index) * 4;
	const uint32_t span_bytes = uint32_t(grp->max_end_index - grp->min_first_index) * 4;
	EXPECT_EQ(span_bytes, body.size());

	// 5. The region hash of the pool slice equals the dedicated-buffer
	//    hash -- this is the core of the pooled-IB emulation.
	const uint32_t region_hash = calc_buffer_hash(
		calc_buffer_data_hash(pool.data() + region_offset, span_bytes),
		span_bytes, BufferRole::index);
	EXPECT_EQ(region_hash, dedicated_hash);
	EXPECT_EQ(region_hash, grp->hash);

	// 6. A different region of the pool must NOT verify (garbage bytes).
	const uint32_t wrong_hash = calc_buffer_hash(
		calc_buffer_data_hash(pool.data(), span_bytes), span_bytes, BufferRole::index);
	EXPECT(wrong_hash != dedicated_hash);

	// 7. Verified view routes the draws; a failed view does not.
	IaState ia;
	ia.bind_index_buffer(kPoolHandle, kViewOffset, 4);

	IndexViewTracker views;
	views.track(kPoolHandle, kViewOffset, 4, 1);
	BufferTracker unlearned; // the pool itself is never hash-learned

	DrawCallInfo call;
	call.index_count = 17970;
	call.first_index = 0;
	call.instance_count = 1;
	EXPECT(find_skip_rule(index, unlearned, &views, ia, call, true) == nullptr);

	views.set_verified(kPoolHandle, kViewOffset, region_hash);
	const TextureOverrideRule *r0 =
		find_skip_rule(index, unlearned, &views, ia, call, true);
	EXPECT(r0 != nullptr);
	EXPECT_EQ(r0->section, std::string("TextureOverrideComponent0"));

	call.index_count = 45636;
	call.first_index = 17970;
	const TextureOverrideRule *r1 =
		find_skip_rule(index, unlearned, &views, ia, call, true);
	EXPECT(r1 != nullptr);
	EXPECT_EQ(r1->section, std::string("TextureOverrideComponent1"));
}

// Temporary diagnostic: trace the draws command list execution.
#include "script_runtime.hpp"
#include "buffer_tracker.hpp"
#include "mod_rules.hpp"

#include <cstdio>
#include <filesystem>
#include <fstream>

using namespace wwmi;

struct TraceBridge final : GpuBridge
{
	uint64_t create_buffer(const void *data, uint64_t size, bool, bool) override
	{
		std::printf("create(size=%llu data=%s)\n",
			static_cast<unsigned long long>(size), data ? "yes" : "no");
		return 0x2000 + size;
	}
	void destroy_buffer(uint64_t h) override
	{
		std::printf("destroy(%llx)\n", static_cast<unsigned long long>(h));
	}
	bool buffer_size(uint64_t h, uint64_t *size) override
	{
		std::printf("buffer_size(%llx)\n", static_cast<unsigned long long>(h));
		*size = 4096;
		return true;
	}
	void bind_vb(uint32_t slot, uint64_t res, uint64_t off, uint32_t stride) override
	{
		std::printf("bind_vb(slot=%u res=%llx off=%llu stride=%u)\n",
			slot, static_cast<unsigned long long>(res),
			static_cast<unsigned long long>(off), stride);
	}
	void bind_ib(uint64_t res, uint64_t off, uint32_t idx) override
	{
		std::printf("bind_ib(res=%llx off=%llu idx=%u)\n",
			static_cast<unsigned long long>(res),
			static_cast<unsigned long long>(off), idx);
	}
	void draw_indexed(uint32_t count, uint32_t first, int32_t base, uint32_t inst) override
	{
		std::printf("draw(count=%u first=%u base=%d inst=%u)\n", count, first, base, inst);
	}
	void copy_buffer(uint64_t s, uint64_t so, uint64_t d, uint64_t dof, uint64_t sz) override
	{
		std::printf("copy(%llx,%llu -> %llx,%llu size=%llu)\n",
			static_cast<unsigned long long>(s), static_cast<unsigned long long>(so),
			static_cast<unsigned long long>(d), static_cast<unsigned long long>(dof),
			static_cast<unsigned long long>(sz));
	}
	uint64_t apply_blend(const BlendConfig &cfg) override
	{
		std::printf("apply_blend(color=%d src=%u dst=%u factors=%g,%g,%g,%g)\n",
			cfg.has_color ? 1 : 0, cfg.src, cfg.dst,
			cfg.factors[0], cfg.factors[1], cfg.factors[2], cfg.factors[3]);
		return 0x9001;
	}
	void revoke_blend(uint64_t token) override
	{
		std::printf("revoke_blend(%llx)\n", static_cast<unsigned long long>(token));
	}
};

int main(int argc, char **argv)
{
	// Diagnostic CLI: runtime-probe <mod.ini> [section]
	// Loads a real mod, then executes the inline script of [section]
	// (e.g. TextureOverrideComponent0) against a fake IA state to
	// preview the bridge calls a real interception would record.
	if (argc >= 2)
	{
		// mod_rules side: dump resource table + per-rule binding resolution
		// so 'has no loadable texture' warnings can be traced offline.
		{
			ModRules mod;
			if (load_mod_rules(argv[1], mod))
			{
				std::printf("[mod_rules] resources=%llu overrides=%llu\n",
					static_cast<unsigned long long>(mod.resources.size()),
					static_cast<unsigned long long>(mod.overrides.size()));
				for (const auto &[name, def] : mod.resources)
					std::printf("  res '%s' -> '%s'\n",
						name.c_str(), def.filename.c_str());
				for (const TextureOverrideRule &r : mod.overrides)
				{
					if (!r.has_hash || r.bindings.empty())
						continue;
					if (r.handling != HandlingMode::none)
						continue; // draw-intercept rule: no texture path
					const std::string &ref = r.bindings.front().resource;
					auto it = mod.resources.find(ref);
					if (it == mod.resources.end())
						it = mod.resources.find("resource" + ref);
					std::printf("  rule '%s' hash=%08x binding0='%s' -> %s\n",
						r.section.c_str(), r.hash, ref.c_str(),
						it != mod.resources.end() && !it->second.filename.empty()
							? it->second.filename.c_str()
							: "<UNRESOLVED>");
				}
			}
			else
				std::printf("[mod_rules] load FAILED\n");
		}

		ScriptRuntime rt;
		if (!rt.load(argv[1]))
		{
			std::printf("load FAILED\n");
			return 1;
		}

		std::printf("mod '%s': %llu var(s), %llu key(s)\n",
			rt.mod_name().c_str(),
			static_cast<unsigned long long>(rt.var_count()),
			static_cast<unsigned long long>(rt.keys().size()));

		IaState ia;
		ia.ib_buffer = 0xABC;
		ia.ib_offset = 0;
		ia.ib_index_size = 4;
		ia.vb_valid_mask = 0x1f;
		for (uint32_t i = 0; i < 5; ++i)
		{
			ia.vbs[i].buffer = 0x100 + i;
			ia.vbs[i].offset = 0;
			ia.vbs[i].stride = 16;
		}

		TraceBridge bridge;
		const char *section = argc >= 3 ? argv[2] : "TextureOverrideComponent0";
		const bool skip = rt.run_override(section, &bridge, &ia);
		std::printf("run_override('%s') -> skip=%d draws=%llu lists=%llu\n",
			section, skip ? 1 : 0,
			static_cast<unsigned long long>(rt.draws_issued),
			static_cast<unsigned long long>(rt.lists_executed));

		for (const std::string &w : rt.warnings())
			std::printf("WARN: %s\n", w.c_str());
		return 0;
	}

	const std::filesystem::path dir = std::filesystem::temp_directory_path() / "wwmi-probe-resops";
	std::error_code ec;
	std::filesystem::remove_all(dir, ec);
	std::filesystem::create_directories(dir, ec);
	{
		std::ofstream f(dir / "mod.ini", std::ios::binary);
		f << R"INI([ResourceZero]
type = RWBuffer
format = R32G32B32A32_FLOAT
array = 64
[CommandListMake]
ResourceFromGame = ref ib
ib = ResourceZero
ResourceAlias = ref ResourceFromGame
ResourceAlias = null
)INI";
	}

	ScriptRuntime rt;
	if (!rt.load(dir / "mod.ini"))
	{
		std::printf("load FAILED\n");
		return 1;
	}

	IaState ia;
	ia.ib_buffer = 0xABC;
	ia.ib_offset = 16;
	ia.ib_index_size = 2;

	TraceBridge bridge;
	const bool ran = rt.run_list("commandlistmake", &bridge, &ia);
	std::printf("ran=%d\n", ran ? 1 : 0);
	for (const std::string &w : rt.warnings())
		std::printf("WARN: %s\n", w.c_str());

	for (const char *n : { "resourcefromgame", "fromgame", "resourcealias", "alias", "resourcezero", "zero" })
	{
		const ScriptResource *r = rt.find_resource(n);
		std::printf("res[%s] -> %s handle=%llx capture=%d\n",
			n, r ? "found" : "MISSING",
			r ? static_cast<unsigned long long>(r->handle) : 0ull,
			r ? static_cast<int>(r->game_capture) : -1);
	}
	return 0;
}

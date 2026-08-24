#include "pipeline_tracker.hpp"
#include "xxhash64.hpp"

#include <cstring>
#include <mutex>

namespace wwmi
{
	namespace
	{
		bool is_shader_subobject(pipeline_subobject_type t)
		{
			switch (t)
			{
			case pipeline_subobject_type::vertex_shader:
			case pipeline_subobject_type::hull_shader:
			case pipeline_subobject_type::domain_shader:
			case pipeline_subobject_type::geometry_shader:
			case pipeline_subobject_type::pixel_shader:
			case pipeline_subobject_type::compute_shader:
			case pipeline_subobject_type::amplification_shader:
			case pipeline_subobject_type::mesh_shader:
			case pipeline_subobject_type::raygen_shader:
			case pipeline_subobject_type::any_hit_shader:
			case pipeline_subobject_type::closest_hit_shader:
			case pipeline_subobject_type::miss_shader:
			case pipeline_subobject_type::intersection_shader:
			case pipeline_subobject_type::callable_shader:
				return true;
			default:
				return false;
			}
		}

		// Deep copy for clone purposes. Returns the copied subobject with
		// data pointing at owned storage, or {type, 0, nullptr} when the
		// type is not needed for cloning (unknown types are dropped: a
		// clone missing an exotic subobject fails safely at creation).
		pipeline_subobject copy_subobject(const pipeline_subobject &src)
		{
			using namespace reshade::api;
			pipeline_subobject out;
			out.type = src.type;
			out.count = src.count;

			switch (src.type)
			{
			case pipeline_subobject_type::input_layout:
			{
				const auto *elems = static_cast<const input_element *>(src.data);
				auto *copy = new input_element[src.count];
				for (uint32_t i = 0; i < src.count; ++i)
				{
					copy[i] = elems[i];
					if (elems[i].semantic != nullptr)
					{
						const size_t len = std::strlen(elems[i].semantic) + 1;
						auto *sem = new char[len];
						std::memcpy(sem, elems[i].semantic, len);
						copy[i].semantic = sem;
					}
				}
				out.data = copy;
				return out;
			}
			case pipeline_subobject_type::render_target_formats:
			{
				auto *copy = new format[src.count];
				std::memcpy(copy, src.data, sizeof(format) * src.count);
				out.data = copy;
				return out;
			}
			case pipeline_subobject_type::dynamic_pipeline_states:
			{
				auto *copy = new dynamic_state[src.count];
				std::memcpy(copy, src.data, sizeof(dynamic_state) * src.count);
				out.data = copy;
				return out;
			}
			case pipeline_subobject_type::blend_state:
				out.data = new blend_desc(*static_cast<const blend_desc *>(src.data));
				return out;
			case pipeline_subobject_type::rasterizer_state:
				out.data = new rasterizer_desc(*static_cast<const rasterizer_desc *>(src.data));
				return out;
			case pipeline_subobject_type::depth_stencil_state:
				out.data = new depth_stencil_desc(*static_cast<const depth_stencil_desc *>(src.data));
				return out;
			case pipeline_subobject_type::stream_output_state:
				out.data = new stream_output_desc(*static_cast<const stream_output_desc *>(src.data));
				return out;
			case pipeline_subobject_type::primitive_topology:
				out.data = new primitive_topology(*static_cast<const primitive_topology *>(src.data));
				return out;
			case pipeline_subobject_type::depth_stencil_format:
				out.data = new format(*static_cast<const format *>(src.data));
				return out;
			case pipeline_subobject_type::sample_mask:
			case pipeline_subobject_type::sample_count:
			case pipeline_subobject_type::viewport_count:
			case pipeline_subobject_type::max_vertex_count:
				out.data = new uint32_t(*static_cast<const uint32_t *>(src.data));
				return out;
			default:
				if (is_shader_subobject(src.type))
				{
					// Borrow the bytecode pointer (game shader archive is
					// assumed resident); copy only the descriptor struct.
					out.data = new shader_desc(*static_cast<const shader_desc *>(src.data));
					return out;
				}
				// Unknown type: drop from the clone source.
				out.type = pipeline_subobject_type::unknown;
				out.count = 0;
				out.data = nullptr;
				return out;
			}
		}
	}

	void PipelineTracker::free_subobject_storage(pipeline_subobject *subs, uint32_t count)
	{
		using namespace reshade::api;
		for (uint32_t i = 0; i < count; ++i)
		{
			const pipeline_subobject &s = subs[i];
			switch (s.type)
			{
			case pipeline_subobject_type::input_layout:
			{
				auto *elems = static_cast<input_element *>(s.data);
				for (uint32_t e = 0; e < s.count; ++e)
					delete[] const_cast<char *>(elems[e].semantic); // NOLINT
				delete[] elems;
				break;
			}
			case pipeline_subobject_type::render_target_formats:
				delete[] static_cast<format *>(s.data);
				break;
			case pipeline_subobject_type::dynamic_pipeline_states:
				delete[] static_cast<dynamic_state *>(s.data);
				break;
			case pipeline_subobject_type::blend_state:
				delete static_cast<blend_desc *>(s.data);
				break;
			case pipeline_subobject_type::rasterizer_state:
				delete static_cast<rasterizer_desc *>(s.data);
				break;
			case pipeline_subobject_type::depth_stencil_state:
				delete static_cast<depth_stencil_desc *>(s.data);
				break;
			case pipeline_subobject_type::stream_output_state:
				delete static_cast<stream_output_desc *>(s.data);
				break;
			case pipeline_subobject_type::primitive_topology:
				delete static_cast<primitive_topology *>(s.data);
				break;
			case pipeline_subobject_type::depth_stencil_format:
				delete static_cast<format *>(s.data);
				break;
			case pipeline_subobject_type::sample_mask:
			case pipeline_subobject_type::sample_count:
			case pipeline_subobject_type::viewport_count:
			case pipeline_subobject_type::max_vertex_count:
				delete static_cast<uint32_t *>(s.data);
				break;
			default:
				if (is_shader_subobject(s.type))
					delete static_cast<shader_desc *>(s.data);
				break; // unknown already nullptr
			}
		}
	}

	void PipelineTracker::on_init_pipeline(uint64_t layout, uint32_t count,
		const pipeline_subobject *subobjects, uint64_t pipeline)
	{
		PipelineShaders sh;
		for (uint32_t i = 0; i < count; ++i)
		{
			const pipeline_subobject &s = subobjects[i];
			if (!is_shader_subobject(s.type) || s.data == nullptr)
				continue;
			const auto *desc = static_cast<const reshade::api::shader_desc *>(s.data);
			if (desc->code == nullptr || desc->code_size == 0)
				continue;

			const uint64_t h = xxhash64(desc->code, desc->code_size);
			switch (s.type)
			{
			case pipeline_subobject_type::vertex_shader: sh.vs = h; sh.has_vs = true; break;
			case pipeline_subobject_type::pixel_shader: sh.ps = h; sh.has_ps = true; break;
			case pipeline_subobject_type::geometry_shader: sh.gs = h; sh.has_gs = true; break;
			case pipeline_subobject_type::domain_shader: sh.ds = h; sh.has_ds = true; break;
			case pipeline_subobject_type::hull_shader: sh.hs = h; sh.has_hs = true; break;
			case pipeline_subobject_type::compute_shader: sh.cs = h; sh.has_cs = true; break;
			default: break; // ray/mesh stages: hashed but unused by mods
			}
		}

		std::lock_guard<std::mutex> guard(_lock);

		_hashes[pipeline] = sh;

		if (_max_cached == 0)
			return;

		// Replace any stale entry (pipeline handle reuse by the driver)
		auto old = _by_handle.find(pipeline);
		if (old != _by_handle.end())
		{
			free_subobject_storage(old->second.subobjects.data(),
				static_cast<uint32_t>(old->second.subobjects.size()));
			_by_handle.erase(old);
		}

		Entry entry;
		entry.shaders = sh;
		entry.layout = layout;
		entry.subobjects.reserve(count);
		for (uint32_t i = 0; i < count; ++i)
		{
			pipeline_subobject copy = copy_subobject(subobjects[i]);
			if (copy.type != pipeline_subobject_type::unknown)
				entry.subobjects.push_back(copy);
		}
		_by_handle.emplace(pipeline, std::move(entry));
		touch(pipeline);

		// LRU eviction (leaves _hashes intact: hash lookup stays possible)
		while (_by_handle.size() > _max_cached)
		{
			const uint64_t victim = _lru.back();
			auto it = _by_handle.find(victim);
			if (it != _by_handle.end())
			{
				free_subobject_storage(it->second.subobjects.data(),
					static_cast<uint32_t>(it->second.subobjects.size()));
				_by_handle.erase(it);
			}
			_lru_pos.erase(victim);
			_lru.pop_back();
		}
	}

	void PipelineTracker::on_destroy_pipeline(uint64_t pipeline)
	{
		std::lock_guard<std::mutex> guard(_lock);

		auto it = _by_handle.find(pipeline);
		if (it != _by_handle.end())
		{
			free_subobject_storage(it->second.subobjects.data(),
				static_cast<uint32_t>(it->second.subobjects.size()));
			_by_handle.erase(it);
		}

		auto pos = _lru_pos.find(pipeline);
		if (pos != _lru_pos.end())
		{
			_lru.erase(pos->second);
			_lru_pos.erase(pos);
		}
		_hashes.erase(pipeline);
	}

	const PipelineShaders *PipelineTracker::find(uint64_t pipeline) const
	{
		std::lock_guard<std::mutex> guard(_lock);
		auto it = _hashes.find(pipeline);
		return it == _hashes.end() ? nullptr : &it->second;
	}

	const std::vector<pipeline_subobject> *PipelineTracker::clone_source(uint64_t pipeline)
	{
		std::lock_guard<std::mutex> guard(_lock);
		auto it = _by_handle.find(pipeline);
		if (it == _by_handle.end())
			return nullptr;
		touch(pipeline);
		return &it->second.subobjects;
	}

	uint64_t PipelineTracker::clone_source_layout(uint64_t pipeline) const
	{
		std::lock_guard<std::mutex> guard(_lock);
		auto it = _by_handle.find(pipeline);
		return it == _by_handle.end() ? 0 : it->second.layout;
	}

	void PipelineTracker::touch(uint64_t pipeline)
	{
		auto pos = _lru_pos.find(pipeline);
		if (pos != _lru_pos.end())
		{
			_lru.splice(_lru.begin(), _lru, pos->second);
			pos->second = _lru.begin();
		}
		else
		{
			_lru.push_front(pipeline);
			_lru_pos[pipeline] = _lru.begin();
		}
	}
}

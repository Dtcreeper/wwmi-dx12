// WWMI-DX12: descriptor-slot replacement engine (m1-9). See header.
#include "override_engine.hpp"

#include <algorithm>

namespace wwmi
{

	std::vector<Overwrite> OverrideEngine::activate(uint64_t resource, uint64_t view)
	{
		_replaced[resource] = view;

		std::vector<Overwrite> out;

		const auto it = _resource_slots.find(resource);
		if (it == _resource_slots.end())
			return out;

		for (const SlotId &id : it->second)
		{
			const auto sit = _slots.find(id);
			if (sit == _slots.end() || sit->second.resource != resource)
				continue; // reassigned or dropped since it was recorded

			out.push_back({ sit->second.write.table, sit->second.write.binding, view });
		}
		return out;
	}

	void OverrideEngine::deactivate(uint64_t resource)
	{
		_replaced.erase(resource);
	}

	bool OverrideEngine::is_replaced(uint64_t resource) const
	{
		return _replaced.count(resource) != 0;
	}

	uint64_t OverrideEngine::replacement_view(uint64_t resource) const
	{
		const auto it = _replaced.find(resource);
		return it != _replaced.end() ? it->second : 0;
	}

	void OverrideEngine::record_slot(uint64_t resource, const SlotId &id, const SlotWrite &write)
	{
		const auto it = _slots.find(id);
		if (it != _slots.end())
		{
			// Existing record: keep the list node, move ownership when the
			// slot was rewritten for a different resource.
			if (it->second.resource != resource)
			{
				remove_from_resource(it->second.resource, id);
				it->second.resource = resource;
				_resource_slots[resource].push_back(id);
			}
			it->second.write = write; // coordinates may have been re-derived
			return;
		}

		// Global bound: drop the oldest records to admit the new one.
		while (_slots.size() >= k_max_slots && !_slot_order.empty())
			drop_slot(_slot_order.front());
		if (_slots.size() >= k_max_slots)
			return; // cannot admit (nothing evictable)

		SlotRecord rec;
		rec.resource = resource;
		rec.write = write;
		rec.order = _slot_order.insert(_slot_order.end(), id);
		_slots.emplace(id, rec);

		// Per-resource bound: drop this resource's oldest record.
		std::vector<SlotId> &owned = _resource_slots[resource];
		if (owned.size() >= k_max_slots_per_resource)
		{
			const SlotId drop = owned.front();
			owned.erase(owned.begin());
			const auto dit = _slots.find(drop);
			if (dit != _slots.end() && dit->second.resource == resource)
			{
				_slot_order.erase(dit->second.order);
				_slots.erase(dit);
				++_dropped;
			}
		}
		owned.push_back(id);
	}

	void OverrideEngine::remove_from_resource(uint64_t resource, const SlotId &id)
	{
		const auto it = _resource_slots.find(resource);
		if (it == _resource_slots.end())
			return;

		auto &v = it->second;
		const auto pos = std::find(v.begin(), v.end(), id);
		if (pos != v.end())
			v.erase(pos);
		if (v.empty())
			_resource_slots.erase(it);
	}

	void OverrideEngine::drop_slot(const SlotId &id)
	{
		const auto it = _slots.find(id);
		if (it == _slots.end())
			return;

		_slot_order.erase(it->second.order);
		remove_from_resource(it->second.resource, id);
		_slots.erase(it);
		++_dropped;
	}

	std::vector<Overwrite> OverrideEngine::on_srv_slot(uint64_t resource, const SlotId &id, const SlotWrite &write)
	{
		record_slot(resource, id, write);

		const auto it = _replaced.find(resource);
		if (it == _replaced.end())
			return {};

		return { { write.table, write.binding, it->second } };
	}

	CopyPlan OverrideEngine::on_copy(const CopySegment *segments, size_t count)
	{
		CopyPlan plan;
		if (segments == nullptr || count == 0)
			return plan;

		for (size_t i = 0; i < count; ++i)
		{
			const CopySegment &seg = segments[i];

			for (uint32_t k = 0; k < seg.count; ++k)
			{
				const SlotId src{ seg.source.heap, seg.source.index + k };
				const SlotId dst{ seg.dest.heap, seg.dest.index + k };
				const SlotWrite dst_write{ seg.dest_table, seg.dest_binding + k };

				const auto sit = _slots.find(src);
				if (sit == _slots.end())
				{
					// Unknown content overwriting a known slot: the dest
					// record is no longer attributable to any resource.
					if (_slots.count(dst) != 0)
						drop_slot(dst);
					continue;
				}

				const uint64_t resource = sit->second.resource;
				record_slot(resource, dst, dst_write);

				const auto rit = _replaced.find(resource);
				if (rit != _replaced.end())
				{
					plan.block = true;
					plan.overwrites.push_back({ dst_write.table, dst_write.binding, rit->second });
				}
			}
		}
		return plan;
	}

	void OverrideEngine::on_resource_destroyed(uint64_t resource)
	{
		_replaced.erase(resource);

		const auto it = _resource_slots.find(resource);
		if (it == _resource_slots.end())
			return;

		for (const SlotId &id : it->second)
		{
			const auto sit = _slots.find(id);
			if (sit != _slots.end() && sit->second.resource == resource)
			{
				_slot_order.erase(sit->second.order);
				_slots.erase(sit);
			}
		}
		_resource_slots.erase(it);
	}

	void OverrideEngine::forget_slot(const SlotId &id)
	{
		drop_slot(id);
	}

} // namespace wwmi

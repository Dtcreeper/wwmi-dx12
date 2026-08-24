// WWMI-DX12: descriptor-slot replacement engine (m1-9).
//
// Bookkeeping that turns "resource X is now replaced by view V" into
// concrete descriptor overwrites, fed by two ReShade D3D12 event paths:
//
//   create path: update_descriptor_tables event (fires after every
//                ID3D12Device::CreateShaderResourceView; provides the
//                slot's descriptor_table handle)
//   copy path:   copy_descriptor_tables event (fires before every
//                ID3D12Device::CopyDescriptors; the original call can be
//                blocked and re-issued by the bridge)
//
// Slots are identified canonically by (descriptor heap, descriptor index)
// so a slot written by the create path is recognized again when the game
// later copies it into its shader-visible heap. Each record also keeps the
// (table, binding) write coordinates that device::update_descriptor_tables
// needs to rewrite that slot.
//
// The engine is deliberately free of ReShade dependencies so it stays unit
// testable; the bridge (addon_main.cpp) canonicalizes event coordinates
// via device::get_descriptor_heap_offset and executes the returned actions
// via device::update_descriptor_tables / copy_descriptor_tables.
#pragma once

#include <cstdint>
#include <list>
#include <unordered_map>
#include <vector>

namespace wwmi
{

	// Canonical descriptor slot id: (descriptor heap, descriptor index).
	struct SlotId
	{
		uint64_t heap = 0;  // api::descriptor_heap handle
		uint32_t index = 0; // descriptor index within the heap

		bool operator==(const SlotId &o) const { return heap == o.heap && index == o.index; }
	};

	struct SlotIdHash
	{
		size_t operator()(const SlotId &s) const
		{
			return std::hash<uint64_t>()(s.heap) * 31 + std::hash<uint32_t>()(s.index);
		}
	};

	// Write coordinates of a slot: the (table, binding) pair that
	// device::update_descriptor_tables needs in order to rewrite it.
	struct SlotWrite
	{
		uint64_t table = 0;
		uint32_t binding = 0;
	};

	// One descriptor rewrite the bridge has to execute.
	struct Overwrite
	{
		uint64_t table = 0;
		uint32_t binding = 0;
		uint64_t view = 0; // replacement SRV handle
	};

	// A flattened copy range as delivered by the copy event, translated to
	// canonical ids. 'source'/'dest' identify the FIRST slot of the range;
	// the segment covers 'count' consecutive descriptors on both sides.
	struct CopySegment
	{
		SlotId source;
		SlotId dest;
		uint64_t dest_table = 0; // write coords of the FIRST dest slot
		uint32_t dest_binding = 0;
		uint32_t count = 1;
	};

	// Result of feeding a copy: when 'block' is true the bridge must block
	// the original CopyDescriptors call, re-issue it verbatim via
	// device::copy_descriptor_tables, then apply the overwrites.
	struct CopyPlan
	{
		bool block = false;
		std::vector<Overwrite> overwrites;
	};

	class OverrideEngine
	{
	public:
		// Bounds: bindless engines can produce very many slots; records are
		// only needed for slots of tracked (replacement candidate) textures.
		static constexpr size_t k_max_slots = 65536;
		static constexpr size_t k_max_slots_per_resource = 256;

		// Marks 'resource' as being replaced by 'view'. Returns every known
		// slot overwrite so the bridge can apply them right away (late
		// activation: slots the game wrote before the hash was learned).
		std::vector<Overwrite> activate(uint64_t resource, uint64_t view);

		// Drops replacement state for 'resource' (slot records are kept so
		// a later re-activation still finds them).
		void deactivate(uint64_t resource);

		bool is_replaced(uint64_t resource) const;
		uint64_t replacement_view(uint64_t resource) const;

		// Create path: an SRV for 'resource' was written into a slot whose
		// canonical id and write coordinates are known. Returns an
		// overwrite when the resource is already replaced.
		std::vector<Overwrite> on_srv_slot(uint64_t resource, const SlotId &id, const SlotWrite &write);

		// Copy path: records dest slots for their source resources and
		// emits overwrites for replaced sources.
		CopyPlan on_copy(const CopySegment *segments, size_t count);

		// Resource destroyed: drop replacement state + all its slot records.
		void on_resource_destroyed(uint64_t resource);

		// Drops a single (stale) slot record, e.g. when the bridge's heap
		// validation fails at activation time.
		void forget_slot(const SlotId &id);

		size_t slot_count() const { return _slots.size(); }
		size_t replaced_count() const { return _replaced.size(); }

		// Total dropped records (diagnostics).
		uint64_t dropped_slots() const { return _dropped; }

	private:
		struct SlotRecord
		{
			uint64_t resource = 0;
			SlotWrite write;
			std::list<SlotId>::const_iterator order; // node in _slot_order
		};

		// Records (or re-assigns) a slot. Bounds: silently drops the oldest
		// records when the global / per-resource caps are exceeded.
		void record_slot(uint64_t resource, const SlotId &id, const SlotWrite &write);

		void drop_slot(const SlotId &id);
		void remove_from_resource(uint64_t resource, const SlotId &id);

		std::unordered_map<uint64_t, uint64_t> _replaced; // resource -> view
		std::unordered_map<SlotId, SlotRecord, SlotIdHash> _slots;
		std::unordered_map<uint64_t, std::vector<SlotId>> _resource_slots; // resource -> slots
		std::list<SlotId> _slot_order; // insertion order for the global bound
		uint64_t _dropped = 0;
	};

} // namespace wwmi

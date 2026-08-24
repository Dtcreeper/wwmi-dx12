// WWMI-DX12: unit tests for the descriptor-slot replacement engine (m1-9).
#include "override_engine.hpp"
#include "test_framework.hpp"

#include <algorithm>

using wwmi::CopyPlan;
using wwmi::CopySegment;
using wwmi::Overwrite;
using wwmi::OverrideEngine;
using wwmi::SlotId;
using wwmi::SlotWrite;

namespace
{
	constexpr uint64_t k_heap_a = 0x11110000;
	constexpr uint64_t k_heap_b = 0x22220000;

	// Replacement SRV handles.
	constexpr uint64_t k_view = 0xAAAA0001;
	constexpr uint64_t k_view_1 = 0xAAAA0002;
	constexpr uint64_t k_view_2 = 0xAAAA0003;

	SlotId slot(uint64_t heap, uint32_t index) { return { heap, index }; }
	SlotWrite write(uint64_t table, uint32_t binding) { return { table, binding }; }

	bool has_overwrite(const std::vector<Overwrite> &ows, uint64_t table, uint32_t binding, uint64_t view)
	{
		return std::any_of(ows.begin(), ows.end(), [&](const Overwrite &o) {
			return o.table == table && o.binding == binding && o.view == view;
		});
	}
}

// ---- create path ---------------------------------------------------------

WWMI_TEST(override_srv_slot_records_and_returns_empty_when_unreplaced)
{
	OverrideEngine e;
	auto ows = e.on_srv_slot(0xA1, slot(k_heap_a, 5), write(0xF1, 0));
	EXPECT(ows.empty());
	EXPECT_EQ(e.slot_count(), 1u);
	EXPECT(!e.is_replaced(0xA1));
}

WWMI_TEST(override_srv_slot_returns_overwrite_when_replaced)
{
	OverrideEngine e;
	e.activate(0xA1, k_view);
	// Slot written AFTER activation: overwritten synchronously.
	auto ows = e.on_srv_slot(0xA1, slot(k_heap_a, 7), write(0xF7, 0));
	EXPECT_EQ(ows.size(), 1u);
	EXPECT_EQ(ows[0].table, 0xF7u);
	EXPECT_EQ(ows[0].binding, 0u);
	EXPECT_EQ(ows[0].view, k_view);
}

WWMI_TEST(override_activate_replays_previously_recorded_slots)
{
	OverrideEngine e;
	// Slots recorded BEFORE the replacement is known (hash still pending):
	e.on_srv_slot(0xA1, slot(k_heap_a, 1), write(0x10, 0));
	e.on_srv_slot(0xA1, slot(k_heap_a, 2), write(0x20, 0));
	e.on_srv_slot(0xB2, slot(k_heap_b, 1), write(0x30, 0)); // other texture

	auto ows = e.activate(0xA1, k_view);
	EXPECT_EQ(ows.size(), 2u);
	EXPECT(has_overwrite(ows, 0x10, 0, k_view));
	EXPECT(has_overwrite(ows, 0x20, 0, k_view));
	EXPECT(e.is_replaced(0xA1));
	EXPECT_EQ(e.replacement_view(0xA1), k_view);
}

WWMI_TEST(override_slot_reassignment_moves_ownership)
{
	OverrideEngine e;
	e.on_srv_slot(0xA1, slot(k_heap_a, 1), write(0x10, 0));
	// The game reuses the slot for another texture.
	e.on_srv_slot(0xB2, slot(k_heap_a, 1), write(0x10, 0));

	auto ows = e.activate(0xA1, k_view_1);
	EXPECT(ows.empty()); // slot no longer belongs to A1

	ows = e.activate(0xB2, k_view_2);
	EXPECT_EQ(ows.size(), 1u);
	EXPECT(has_overwrite(ows, 0x10, 0, k_view_2));
}

// ---- copy path -----------------------------------------------------------

WWMI_TEST(override_copy_records_dest_for_late_activation)
{
	OverrideEngine e;
	e.on_srv_slot(0xA1, slot(k_heap_a, 3), write(0x30, 0)); // staging slot

	// Game copies staging slot 3 -> GPU heap slot 42.
	CopySegment seg{ slot(k_heap_a, 3), slot(k_heap_b, 42), 0x3000, 0, 1 };
	CopyPlan plan = e.on_copy(&seg, 1);
	EXPECT(!plan.block); // no replacement yet
	EXPECT(plan.overwrites.empty());

	// Late activation must reach the GPU-visible destination slot (and the
	// staging slot it was copied from -- both belong to the resource).
	auto ows = e.activate(0xA1, k_view);
	EXPECT_EQ(ows.size(), 2u);
	EXPECT(has_overwrite(ows, 0x3000, 0, k_view));
	EXPECT(has_overwrite(ows, 0x30, 0, k_view));
}

WWMI_TEST(override_copy_with_active_replacement_blocks_and_overwrites)
{
	OverrideEngine e;
	e.on_srv_slot(0xA1, slot(k_heap_a, 3), write(0x30, 0));
	e.activate(0xA1, k_view);

	CopySegment seg{ slot(k_heap_a, 3), slot(k_heap_b, 42), 0x3000, 0, 1 };
	CopyPlan plan = e.on_copy(&seg, 1);
	EXPECT(plan.block);
	EXPECT_EQ(plan.overwrites.size(), 1u);
	EXPECT(has_overwrite(plan.overwrites, 0x3000, 0, k_view));
}

WWMI_TEST(override_copy_range_overwrites_matching_offset_only)
{
	OverrideEngine e;
	// Replaced texture sits at slot 2 of a 4-descriptor copy range.
	e.on_srv_slot(0xA1, slot(k_heap_a, 2), write(0x20, 0));
	e.on_srv_slot(0xC3, slot(k_heap_a, 0), write(0x00, 0)); // not replaced
	e.on_srv_slot(0xD4, slot(k_heap_a, 3), write(0x30, 0)); // not replaced
	e.activate(0xA1, k_view);

	CopySegment seg{ slot(k_heap_a, 0), slot(k_heap_b, 100), 0xB000, 0, 4 };
	CopyPlan plan = e.on_copy(&seg, 1);
	EXPECT(plan.block);
	EXPECT_EQ(plan.overwrites.size(), 1u);
	// Dest slot for source offset 2 is dest index 102 -> binding 2.
	EXPECT(has_overwrite(plan.overwrites, 0xB000, 2, k_view));
}

WWMI_TEST(override_copy_unknown_source_drops_known_dest)
{
	OverrideEngine e;
	e.on_srv_slot(0xA1, slot(k_heap_a, 1), write(0x10, 0));
	e.on_srv_slot(0xA1, slot(k_heap_b, 9), write(0x90, 0)); // known dest slot

	// The game copies untracked content over GPU slot 9.
	CopySegment seg{ slot(k_heap_a, 50), slot(k_heap_b, 9), 0x9000, 0, 1 };
	CopyPlan plan = e.on_copy(&seg, 1);
	EXPECT(!plan.block);
	EXPECT(plan.overwrites.empty());

	auto ows = e.activate(0xA1, k_view);
	// Only the staging slot survives; the recycled GPU slot must not be
	// touched (its content no longer belongs to A1).
	EXPECT_EQ(ows.size(), 1u);
	EXPECT(has_overwrite(ows, 0x10, 0, k_view));
}

WWMI_TEST(override_copy_without_any_records_is_noop)
{
	OverrideEngine e;
	CopySegment seg{ slot(k_heap_a, 0), slot(k_heap_b, 0), 0x1000, 0, 16 };
	CopyPlan plan = e.on_copy(&seg, 1);
	EXPECT(!plan.block);
	EXPECT(plan.overwrites.empty());
}

// ---- lifecycle -----------------------------------------------------------

WWMI_TEST(override_resource_destroyed_drops_slots_and_replacement)
{
	OverrideEngine e;
	e.on_srv_slot(0xA1, slot(k_heap_a, 1), write(0x10, 0));
	e.on_srv_slot(0xA1, slot(k_heap_b, 2), write(0x20, 0));
	e.activate(0xA1, k_view);

	e.on_resource_destroyed(0xA1);
	EXPECT(!e.is_replaced(0xA1));
	EXPECT_EQ(e.slot_count(), 0u);

	// Re-activation after the destroy finds nothing.
	auto ows = e.activate(0xA1, k_view);
	EXPECT(ows.empty());
}

WWMI_TEST(override_forget_slot_removes_single_record)
{
	OverrideEngine e;
	e.on_srv_slot(0xA1, slot(k_heap_a, 1), write(0x10, 0));
	e.on_srv_slot(0xA1, slot(k_heap_a, 2), write(0x20, 0));

	e.forget_slot(slot(k_heap_a, 1));
	EXPECT_EQ(e.slot_count(), 1u);

	auto ows = e.activate(0xA1, k_view);
	EXPECT_EQ(ows.size(), 1u);
	EXPECT(has_overwrite(ows, 0x20, 0, k_view));
}

WWMI_TEST(override_deactivate_keeps_slots_for_reactivation)
{
	OverrideEngine e;
	e.on_srv_slot(0xA1, slot(k_heap_a, 1), write(0x10, 0));
	e.activate(0xA1, k_view);
	e.deactivate(0xA1);
	EXPECT(!e.is_replaced(0xA1));
	EXPECT_EQ(e.slot_count(), 1u);

	auto ows = e.activate(0xA1, k_view_2);
	EXPECT_EQ(ows.size(), 1u);
	EXPECT(has_overwrite(ows, 0x10, 0, k_view_2));
}

// ---- bounds --------------------------------------------------------------

WWMI_TEST(override_per_resource_slot_cap_evicts_oldest)
{
	OverrideEngine e;
	for (uint32_t i = 0; i < wwmi::OverrideEngine::k_max_slots_per_resource + 8; ++i)
		e.on_srv_slot(0xA1, slot(k_heap_a, i), write(0x1000 + i, 0));

	EXPECT_EQ(e.slot_count(), wwmi::OverrideEngine::k_max_slots_per_resource);

	auto ows = e.activate(0xA1, k_view);
	// The first few (oldest) records were evicted; the newest survive.
	EXPECT_EQ(ows.size(), wwmi::OverrideEngine::k_max_slots_per_resource);
	EXPECT(!has_overwrite(ows, 0x1000, 0, k_view));      // oldest: evicted
	EXPECT(has_overwrite(ows, 0x1000 + 8, 0, k_view));   // first survivor
}

WWMI_TEST(override_global_slot_cap_evicts_across_resources)
{
	OverrideEngine e;
	const size_t total = wwmi::OverrideEngine::k_max_slots + 10;
	for (size_t i = 0; i < total; ++i)
	{
		const uint64_t res = 0x100 + (i / 4); // several resources, 4 slots each
		e.on_srv_slot(res, slot(k_heap_a, static_cast<uint32_t>(i)), write(0x2000 + i, 0));
	}

	EXPECT_EQ(e.slot_count(), wwmi::OverrideEngine::k_max_slots);
	EXPECT(e.dropped_slots() >= 10);
}

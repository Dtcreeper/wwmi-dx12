// WWMI-DX12 core unit test runner (no external framework).
// Test cases self-register via WWMI_TEST; keep test_*.cpp files listed in
// CMakeLists.txt so their registrars are linked in.
#include "test_framework.hpp"

int main()
{
	return wwmi_test::run_all();
}

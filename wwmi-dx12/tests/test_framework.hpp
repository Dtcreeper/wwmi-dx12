// WWMI-DX12: minimal self-registering unit test framework (no external deps).
#pragma once

#include <cstdio>
#include <vector>

namespace wwmi_test
{
	struct TestCase
	{
		const char *name;
		void (*fn)();
	};

	inline std::vector<TestCase> &registry()
	{
		static std::vector<TestCase> r;
		return r;
	}
	inline int &failures()
	{
		static int f = 0;
		return f;
	}
	inline int &checks()
	{
		static int c = 0;
		return c;
	}

	struct Registrar
	{
		Registrar(const char *name, void (*fn)())
		{
			registry().push_back({ name, fn });
		}
	};

	inline void expect_true(bool cond, const char *expr, const char *file, int line)
	{
		++checks();
		if (!cond)
		{
			++failures();
			std::printf("    FAIL %s:%d: %s\n", file, line, expr);
		}
	}

	inline int run_all()
	{
		std::setvbuf(stdout, nullptr, _IONBF, 0);
		int failed_cases = 0;
		for (const TestCase &tc : registry())
		{
			const int before = failures();
			std::printf("  [TEST] %s\n", tc.name);
			tc.fn();
			if (failures() > before)
				++failed_cases;
		}

		std::printf("\n%d checks, %d failed cases, %d total cases\n",
			checks(), failed_cases, static_cast<int>(registry().size()));
		return failed_cases == 0 ? 0 : 1;
	}
}

#define WWMI_TEST(name)                                                        \
	static void wwmi_test_fn_##name();                                         \
	static wwmi_test::Registrar wwmi_test_reg_##name(#name, &wwmi_test_fn_##name); \
	static void wwmi_test_fn_##name()

#define EXPECT(cond) wwmi_test::expect_true((cond), #cond, __FILE__, __LINE__)
#define EXPECT_EQ(a, b) wwmi_test::expect_true((a) == (b), #a " == " #b, __FILE__, __LINE__)
#define EXPECT_NE(a, b) wwmi_test::expect_true((a) != (b), #a " != " #b, __FILE__, __LINE__)

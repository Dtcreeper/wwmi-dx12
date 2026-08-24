#include "expr.hpp"
#include "test_framework.hpp"

#include <cmath>
#include <map>

using namespace wwmi;

namespace
{
	struct Vars
	{
		std::map<std::string, float> map;
		std::map<std::string, bool> alive;

		expr::EvalContext ctx()
		{
			expr::EvalContext c;
			c.get_var = [this](const std::string &n, float &v)
			{
				const auto it = map.find(n);
				if (it == map.end())
					return false;
				v = it->second;
				return true;
			};
			c.resource_alive = [this](const std::string &n)
			{
				return alive.count(n) > 0 && alive[n];
			};
			return c;
		}
	};

	float run(const std::string &text, Vars &v)
	{
		std::string err;
		expr::NodePtr ast = expr::compile(text, &err);
		EXPECT(ast != nullptr);
		if (ast == nullptr)
			return -12345.0f;
		float out = -12345.0f;
		expr::eval(*ast, v.ctx(), out);
		return out;
	}
}

WWMI_TEST(expr_literals_and_normalization)
{
	EXPECT_EQ(expr::normalize_var("$ModEnabled"), std::string("modenabled"));
	EXPECT_EQ(expr::normalize_var("$\\WWMIv1\\BlendRemapID"), std::string("\\wwmiv1\\blendremapid"));

	Vars v;
	EXPECT_EQ(run("1", v), 1.0f);
	EXPECT_EQ(run("-2.5", v), -2.5f);
	EXPECT_EQ(run("", v), 0.0f);
}

WWMI_TEST(expr_variables_and_default_zero)
{
	Vars v;
	v.map["object_detected"] = 1.0f;
	v.map["leg"] = 2.0f;

	EXPECT_EQ(run("$object_detected", v), 1.0f);
	EXPECT_EQ(run("$leg", v), 2.0f);
	EXPECT_EQ(run("$undefined_var", v), 0.0f); // 3DMigoto: undefined = 0
}

WWMI_TEST(expr_arithmetic_precedence)
{
	Vars v;
	v.map["mesh_vertex_count"] = 141645.0f;

	EXPECT_EQ(run("1 + 2 * 3", v), 7.0f);
	EXPECT_EQ(run("(1 + 2) * 3", v), 9.0f);
	EXPECT_EQ(run("10 / 4", v), 2.5f);
	EXPECT_EQ(run("$mesh_vertex_count - 145", v), 141500.0f);
	EXPECT_EQ(run("-$mesh_vertex_count", v), -141645.0f);
	EXPECT_EQ(run("5 - 2 - 1", v), 2.0f); // left assoc
}

WWMI_TEST(expr_comparisons_and_logic)
{
	Vars v;
	v.map["cloth"] = 0.0f;
	v.map["tm"] = 1.0f;
	v.map["lun"] = 0.0f;
	v.map["shoe"] = 0.0f;
	v.map["mod_id"] = -1000.0f;

	// Lynae-style conditions
	EXPECT_EQ(run("$cloth == 0 && $tm == 1", v), 1.0f);
	EXPECT_EQ(run("$cloth == 0 && $tm == 0", v), 0.0f);
	EXPECT_EQ(run("$lun == 0 && ($shoe == 0 || $shoe == 1)", v), 1.0f);
	EXPECT_EQ(run("$mod_id == -1000", v), 1.0f);
	EXPECT_EQ(run("$mod_id >= 0", v), 0.0f);
	EXPECT_EQ(run("!$cloth", v), 1.0f);
	EXPECT_EQ(run("!$tm", v), 0.0f);
	// '===' and '!==' alias == and !=
	EXPECT_EQ(run("$cloth === 0", v), 1.0f);
	EXPECT_EQ(run("$cloth !== 1", v), 1.0f);
}

WWMI_TEST(expr_resource_null_checks)
{
	Vars v;
	v.alive["resourcemergedskeleton"] = true;

	EXPECT_EQ(run("ResourceMergedSkeleton !== null", v), 1.0f);
	EXPECT_EQ(run("ResourceMergedSkeleton === null", v), 0.0f);
	EXPECT_EQ(run("ResourceUnknown == null", v), 1.0f);
	EXPECT_EQ(run("ResourceUnknown !== null", v), 0.0f);
}

WWMI_TEST(expr_short_circuit_and_division)
{
	Vars v;
	v.map["x"] = 0.0f;

	EXPECT_EQ(run("$x != 0 && 1 / $x > 1", v), 0.0f); // no div-by-zero hit
	EXPECT_EQ(run("1 / 0", v), 0.0f);                  // guarded
	EXPECT_EQ(run("0 || 2", v), 1.0f);
	EXPECT_EQ(run("3 && 0", v), 0.0f);
}

WWMI_TEST(expr_syntax_errors_reported)
{
	std::string err;
	EXPECT(expr::compile("1 +", &err) == nullptr);
	EXPECT(!err.empty());
	err.clear();
	EXPECT(expr::compile("(1", &err) == nullptr);
	EXPECT(!err.empty());
	err.clear();
	EXPECT(expr::compile("a =", &err) == nullptr);
	EXPECT(!err.empty());
	// Valid odd-but-supported forms
	EXPECT(expr::compile("1", &err) != nullptr);
	EXPECT(expr::compile("$a === $b", &err) != nullptr);
}

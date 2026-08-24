// Temporary diagnostic: trace 'ResourceX === null' compilation/eval.
#include "expr.hpp"

#include <cstdio>
#include <string>

using namespace wwmi;

static void dump(const expr::Node &n, int depth)
{
	for (int i = 0; i < depth; ++i) std::printf("  ");
	switch (n.kind)
	{
	case expr::NodeKind::number: std::printf("number %g\n", n.number); break;
	case expr::NodeKind::variable: std::printf("variable '%s'\n", n.name.c_str()); break;
	case expr::NodeKind::resource: std::printf("resource '%s'\n", n.name.c_str()); break;
	case expr::NodeKind::null_lit: std::printf("null\n"); break;
	case expr::NodeKind::unary: std::printf("unary op=%d\n", static_cast<int>(n.op)); if (n.lhs) dump(*n.lhs, depth + 1); break;
	case expr::NodeKind::binary:
		std::printf("binary op=%d\n", static_cast<int>(n.op));
		if (n.lhs) dump(*n.lhs, depth + 1);
		if (n.rhs) dump(*n.rhs, depth + 1);
		break;
	}
}

int main()
{
	const char *cases[] = {
		"ResourceMergedSkeleton !== null",
		"ResourceMergedSkeleton === null",
		"ResourceUnknown == null",
		"$cloth === 0",
		"ResourceBlendBufferOverride === null",
	};
	for (const char *c : cases)
	{
		std::string err;
		expr::NodePtr ast = expr::compile(c, &err);
		std::printf("== %s => %s\n", c, ast ? "OK" : ("FAIL: " + err).c_str());
		if (ast)
		{
			dump(*ast, 1);
			expr::EvalContext ctx{
				[](const std::string &, float &v) { v = 0.0f; return false; },
				[](const std::string &n) { return n == "resourcemergedskeleton"; }
			};
			float out = -777.0f;
			expr::eval(*ast, ctx, out);
			std::printf("   eval = %g\n", out);
		}
	}
	return 0;
}

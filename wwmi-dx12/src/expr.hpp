// WWMI-DX12: 3DMigoto command-list expression language (M3).
//
// Command lists evaluate conditions and assignments through 3DMigoto's
// expression syntax (Parser::EvalValues subset). Everything is float,
// like 3DMigoto's Value. The full Lynae-mod subset:
//
//   operands:   123, -1.5, $var, $\namespace\var, ResourceName, null
//   unary:      !  -
//   binary:     ||  &&  ==  !=  !==  <  >  <=  >=  +  -  *  /
//   grouping:   ( ... )
//
// Semantics mirrored from 3DMigoto:
//  - variables and resources are looked up case-insensitively
//  - '$' is stripped from variable names; '$\NS\var' becomes 'ns\var'
//  - undefined variables evaluate to 0
//  - comparisons and !/&&/|| produce 0.0f or 1.0f
//  - && and || short-circuit
//  - 'ResourceX == null' / '!== null' compare resource liveness (the
//    only resource comparison supported; numeric comparisons against a
//    resource evaluate the resource as 0 with a warning)
//
// Expressions are compiled once at mod-load time into an AST; the
// runtime only walks the tree.
#pragma once

#include <cmath>
#include <memory>
#include <string>
#include <functional>

namespace wwmi::expr
{
	// Variable/liveness lookup the evaluator consults. Both callbacks
	// receive normalized (lowercase) names.
	struct EvalContext
	{
		// Variable lookup. Return false for undefined variables (the
		// evaluator then uses 0.0f, matching 3DMigoto).
		std::function<bool(const std::string &, float &)> get_var;
		// Whether a resource with that (lowercase) name currently has a
		// live GPU instance.
		std::function<bool(const std::string &)> resource_alive;
	};

	struct Node;
	using NodePtr = std::unique_ptr<Node>;

	enum class NodeKind : uint8_t
	{
		number,   // literal
		variable, // $name (normalized, no '$')
		resource, // bare identifier (ResourceX / cs / vb0 ...)
		null_lit, // 'null'
		unary,    // op: '!' or '-'
		binary,   // op: one of the binary ops below
	};

	// Binary operators (unary ops are '!' and '-').
	enum class Op : uint8_t
	{
		logical_or, logical_and,
		equal, not_equal,
		less, less_equal, greater, greater_equal,
		add, sub, mul, div,
	};

	struct Node
	{
		NodeKind kind = NodeKind::number;
		float number = 0.0f;         // number literal
		std::string name;            // variable (normalized) or resource
		Op op = Op::add;             // unary/binary operator
		bool negate = false;         // unary '-' flag (op == sub unused)
		NodePtr lhs, rhs;            // unary uses lhs only
	};

	// Compiles an expression string. Returns nullptr on a syntax error
	// and fills 'err' (when non-null). Empty input is a valid '0'.
	NodePtr compile(const std::string &text, std::string *err);

	// Evaluates a compiled AST. Always fills 'out' (errors degrade to
	// 0.0f like 3DMigoto); returns false when something was undefined
	// so callers can surface warnings.
	bool eval(const Node &node, const EvalContext &ctx, float &out);

	// Normalizes a variable reference: lowercases and strips the '$'
	// prefix; '$\NS\var' -> 'ns\var'.
	std::string normalize_var(std::string_view name);
}

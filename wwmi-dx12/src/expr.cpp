#include "expr.hpp"

#include <cctype>
#include <cstdlib>

namespace wwmi::expr
{
	std::string normalize_var(std::string_view name)
	{
		std::string out;
		out.reserve(name.size());
		for (char c : name)
			out.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
		if (!out.empty() && out[0] == '$')
			out.erase(out.begin());
		return out;
	}

	namespace
	{
		struct Token
		{
			enum class Kind : uint8_t
			{
				end,
				number,    // 123, 1.5, .5
				ident,     // resourcefoo, null, true, false
				variable,  // $foo (text keeps the '$' stripped path)
				lparen, rparen,
				not_,      // '!'
				logical_or, logical_and,
				equal, not_equal,   // '==', '!=' / '!=='
				less, less_equal, greater, greater_equal,
				plus, minus, star, slash,
			};
			Kind kind = Kind::end;
			float number = 0.0f;
			std::string text; // ident / variable
		};

		struct Lexer
		{
			const char *p;
			const char *end;
			std::string err;

			explicit Lexer(const std::string &text)
				: p(text.c_str()), end(p + text.size()) {}

			void skip_ws() { while (p < end && (*p == ' ' || *p == '\t')) ++p; }

			bool fail(const char *msg)
			{
				if (err.empty())
					err = msg;
				return false;
			}

			bool next(Token &t)
			{
				skip_ws();
				if (p >= end)
				{
					t.kind = Token::Kind::end;
					return true;
				}

				const char c = *p;

				if (std::isdigit(static_cast<unsigned char>(c)) ||
					(c == '.' && p + 1 < end && std::isdigit(static_cast<unsigned char>(p[1]))))
				{
					char *stop = nullptr;
					t.number = static_cast<float>(std::strtod(p, &stop));
					t.kind = Token::Kind::number;
					t.text.clear();
					p = stop;
					return true;
				}

				if (c == '$')
				{
					++p;
					const char *start = p;
					while (p < end && *p != ' ' && *p != '\t' && *p != '(' && *p != ')' &&
						*p != '!' && *p != '=' && *p != '<' && *p != '>' &&
						*p != '&' && *p != '|' && *p != '+' && *p != '-' &&
						*p != '*' && *p != '/')
						++p;
					t.kind = Token::Kind::variable;
					t.text = normalize_var(std::string_view(start, p - start));
					return true;
				}

				if (std::isalpha(static_cast<unsigned char>(c)) || c == '_' || c == '\\')
				{
					const char *start = p;
					while (p < end && (std::isalnum(static_cast<unsigned char>(*p)) ||
						*p == '_' || *p == '\\'))
						++p;
					t.kind = Token::Kind::ident;
					t.text = normalize_var(std::string_view(start, p - start));
					return true;
				}

				++p;
				switch (c)
				{
				case '(': t.kind = Token::Kind::lparen; return true;
				case ')': t.kind = Token::Kind::rparen; return true;
				case '!':
					if (p < end && *p == '=')
					{
						++p;
						if (p < end && *p == '=') ++p; // '!=='
						t.kind = Token::Kind::not_equal;
					}
					else t.kind = Token::Kind::not_;
					return true;
				case '=':
					if (p < end && *p == '=')
					{
						++p;
						if (p < end && *p == '=') ++p; // '==='
						t.kind = Token::Kind::equal;
						return true;
					}
					return fail("single '=' is not an expression operator");
				case '|':
					if (p < end && *p == '|') { ++p; t.kind = Token::Kind::logical_or; return true; }
					return fail("single '|' is not an expression operator");
				case '&':
					if (p < end && *p == '&') { ++p; t.kind = Token::Kind::logical_and; return true; }
					return fail("single '&' is not an expression operator");
				case '<':
					if (p < end && *p == '=') { ++p; t.kind = Token::Kind::less_equal; }
					else t.kind = Token::Kind::less;
					return true;
				case '>':
					if (p < end && *p == '=') { ++p; t.kind = Token::Kind::greater_equal; }
					else t.kind = Token::Kind::greater;
					return true;
				case '+': t.kind = Token::Kind::plus; return true;
				case '-': t.kind = Token::Kind::minus; return true;
				case '*': t.kind = Token::Kind::star; return true;
				case '/': t.kind = Token::Kind::slash; return true;
				default:
					return fail("unexpected character in expression");
				}
			}
		};

		// Recursive-descent parser (precedence climbing mirrors 3DMigoto
		// Parser::ParseExpr*: || < && < equality < relational < additive
		// < multiplicative < unary < primary).
		struct Parser
		{
			Lexer &lx;
			Token cur;
			std::string err;

			explicit Parser(Lexer &lx) : lx(lx) { advance(); }

			void advance() { lx.next(cur); }
			bool fail(const char *msg)
			{
				if (err.empty())
					err = msg;
				return false;
			}

			NodePtr parse()
			{
				NodePtr n = parse_or();
				if (n == nullptr)
					return nullptr;
				if (cur.kind != Token::Kind::end)
				{
					fail("trailing tokens after expression");
					return nullptr;
				}
				return n;
			}

			NodePtr parse_or()
			{
				NodePtr lhs = parse_and();
				if (lhs == nullptr) return nullptr;
				while (cur.kind == Token::Kind::logical_or)
				{
					advance();
					NodePtr rhs = parse_and();
					if (rhs == nullptr) return nullptr;
					NodePtr n = std::make_unique<Node>();
					n->kind = NodeKind::binary;
					n->op = Op::logical_or;
					n->lhs = std::move(lhs);
					n->rhs = std::move(rhs);
					lhs = std::move(n);
				}
				return lhs;
			}

			NodePtr parse_and()
			{
				NodePtr lhs = parse_equality();
				if (lhs == nullptr) return nullptr;
				while (cur.kind == Token::Kind::logical_and)
				{
					advance();
					NodePtr rhs = parse_equality();
					if (rhs == nullptr) return nullptr;
					NodePtr n = std::make_unique<Node>();
					n->kind = NodeKind::binary;
					n->op = Op::logical_and;
					n->lhs = std::move(lhs);
					n->rhs = std::move(rhs);
					lhs = std::move(n);
				}
				return lhs;
			}

			NodePtr parse_equality()
			{
				NodePtr lhs = parse_relational();
				if (lhs == nullptr) return nullptr;
				while (cur.kind == Token::Kind::equal || cur.kind == Token::Kind::not_equal)
				{
					const Op op = cur.kind == Token::Kind::equal ? Op::equal : Op::not_equal;
					advance();
					NodePtr rhs = parse_relational();
					if (rhs == nullptr) return nullptr;
					NodePtr n = std::make_unique<Node>();
					n->kind = NodeKind::binary;
					n->op = op;
					n->lhs = std::move(lhs);
					n->rhs = std::move(rhs);
					lhs = std::move(n);
				}
				return lhs;
			}

			NodePtr parse_relational()
			{
				NodePtr lhs = parse_additive();
				if (lhs == nullptr) return nullptr;
				for (;;)
				{
					Op op;
					switch (cur.kind)
					{
					case Token::Kind::less: op = Op::less; break;
					case Token::Kind::less_equal: op = Op::less_equal; break;
					case Token::Kind::greater: op = Op::greater; break;
					case Token::Kind::greater_equal: op = Op::greater_equal; break;
					default: return lhs;
					}
					advance();
					NodePtr rhs = parse_additive();
					if (rhs == nullptr) return nullptr;
					NodePtr n = std::make_unique<Node>();
					n->kind = NodeKind::binary;
					n->op = op;
					n->lhs = std::move(lhs);
					n->rhs = std::move(rhs);
					lhs = std::move(n);
				}
			}

			NodePtr parse_additive()
			{
				NodePtr lhs = parse_multiplicative();
				if (lhs == nullptr) return nullptr;
				for (;;)
				{
					Op op;
					if (cur.kind == Token::Kind::plus) op = Op::add;
					else if (cur.kind == Token::Kind::minus) op = Op::sub;
					else return lhs;
					advance();
					NodePtr rhs = parse_multiplicative();
					if (rhs == nullptr) return nullptr;
					NodePtr n = std::make_unique<Node>();
					n->kind = NodeKind::binary;
					n->op = op;
					n->lhs = std::move(lhs);
					n->rhs = std::move(rhs);
					lhs = std::move(n);
				}
			}

			NodePtr parse_multiplicative()
			{
				NodePtr lhs = parse_unary();
				if (lhs == nullptr) return nullptr;
				for (;;)
				{
					Op op;
					if (cur.kind == Token::Kind::star) op = Op::mul;
					else if (cur.kind == Token::Kind::slash) op = Op::div;
					else return lhs;
					advance();
					NodePtr rhs = parse_unary();
					if (rhs == nullptr) return nullptr;
					NodePtr n = std::make_unique<Node>();
					n->kind = NodeKind::binary;
					n->op = op;
					n->lhs = std::move(lhs);
					n->rhs = std::move(rhs);
					lhs = std::move(n);
				}
			}

			NodePtr parse_unary()
			{
				if (cur.kind == Token::Kind::not_)
				{
					advance();
					NodePtr inner = parse_unary();
					if (inner == nullptr) return nullptr;
					NodePtr n = std::make_unique<Node>();
					n->kind = NodeKind::unary;
					n->op = Op::add; // '!'
					n->lhs = std::move(inner);
					return n;
				}
				if (cur.kind == Token::Kind::minus)
				{
					advance();
					NodePtr inner = parse_unary();
					if (inner == nullptr) return nullptr;
					NodePtr n = std::make_unique<Node>();
					n->kind = NodeKind::unary;
					n->op = Op::sub; // '-'
					n->lhs = std::move(inner);
					return n;
				}
				return parse_primary();
			}

			NodePtr parse_primary()
			{
				switch (cur.kind)
				{
				case Token::Kind::number:
				{
					NodePtr n = std::make_unique<Node>();
					n->kind = NodeKind::number;
					n->number = cur.number;
					advance();
					return n;
				}
				case Token::Kind::variable:
				{
					NodePtr n = std::make_unique<Node>();
					n->kind = NodeKind::variable;
					n->name = cur.text;
					advance();
					return n;
				}
				case Token::Kind::ident:
				{
					NodePtr n = std::make_unique<Node>();
					if (cur.text == "null" || cur.text == "true" || cur.text == "false")
					{
						n->kind = cur.text == "null" ? NodeKind::null_lit : NodeKind::number;
						n->number = cur.text == "true" ? 1.0f : 0.0f;
					}
					else
					{
						n->kind = NodeKind::resource;
						n->name = cur.text;
					}
					advance();
					return n;
				}
				case Token::Kind::lparen:
				{
					advance();
					NodePtr inner = parse_or();
					if (inner == nullptr) return nullptr;
					if (cur.kind != Token::Kind::rparen)
					{
						fail("missing ')'");
						return nullptr;
					}
					advance();
					return inner;
				}
				default:
					fail("expected an operand");
					return nullptr;
				}
			}
		};
	}

	NodePtr compile(const std::string &text, std::string *err)
	{
		Lexer lx(text);
		if (text.find_first_not_of(" \t") == std::string::npos)
		{
			// Empty expression = 0 (3DMigoto allows bare 'if $x' etc.,
			// but an empty condition line evaluates as 0/false).
			NodePtr n = std::make_unique<Node>();
			n->kind = NodeKind::number;
			return n;
		}

		Parser ps(lx);
		NodePtr ast = ps.parse();
		if (ast == nullptr)
		{
			if (err != nullptr)
				*err = ps.err.empty() ? lx.err : ps.err;
			return nullptr;
		}
		return ast;
	}

	namespace
	{
		bool eval_node(const Node &n, const EvalContext &ctx, float &out);

		float operand_value(const Node &n, const EvalContext &ctx, bool &defined)
		{
			switch (n.kind)
			{
			case NodeKind::number:
				defined = true;
				return n.number;
			case NodeKind::variable:
			{
				float v = 0.0f;
				const bool ok = ctx.get_var && ctx.get_var(n.name, v);
				defined = ok; // undefined variable -> 0 (3DMigoto), flagged
				return ok ? v : 0.0f;
			}
			case NodeKind::null_lit:
				defined = true;
				return 0.0f;
			case NodeKind::resource:
				// Bare resource in arithmetic context: 0 (the meaningful
				// form is 'ResourceX == null', handled structurally).
				defined = false;
				return 0.0f;
			default:
			{
				float v = 0.0f;
				defined = eval_node(n, ctx, v);
				return v;
			}
			}
		}

		bool eval_node(const Node &n, const EvalContext &ctx, float &out)
		{
			switch (n.kind)
			{
			case NodeKind::number:
		case NodeKind::variable:
		case NodeKind::null_lit:
		case NodeKind::resource:
		{
			bool defined = false;
			out = operand_value(n, ctx, defined);
			return true;
		}

			case NodeKind::unary:
			{
				float v = 0.0f;
				const bool defined = eval_node(*n.lhs, ctx, v);
				out = (n.op == Op::sub) ? -v : (v == 0.0f ? 1.0f : 0.0f);
				return defined;
			}

			case NodeKind::binary:
			{
				// 'ResourceX ==/!= null' compares liveness structurally.
				const bool null_cmp = n.op == Op::equal || n.op == Op::not_equal;
				if (null_cmp && n.rhs->kind == NodeKind::null_lit &&
					(n.lhs->kind == NodeKind::resource))
				{
					const bool alive = ctx.resource_alive && ctx.resource_alive(n.lhs->name);
				// 'X == null' is true when X is DEAD; '!= null' when alive.
				out = (n.op == Op::equal) ? (alive ? 0.0f : 1.0f) : (alive ? 1.0f : 0.0f);
				return true;
				}

				if (n.op == Op::logical_and)
				{
					float l = 0.0f;
					const bool dl = eval_node(*n.lhs, ctx, l);
					if (l == 0.0f) { out = 0.0f; return true; } // short-circuit
					float r = 0.0f;
					const bool dr = eval_node(*n.rhs, ctx, r);
					out = (r != 0.0f) ? 1.0f : 0.0f;
					return dl && dr;
				}
				if (n.op == Op::logical_or)
				{
					float l = 0.0f;
					const bool dl = eval_node(*n.lhs, ctx, l);
					if (l != 0.0f) { out = 1.0f; return true; }
					float r = 0.0f;
					const bool dr = eval_node(*n.rhs, ctx, r);
					out = (r != 0.0f) ? 1.0f : 0.0f;
					return dl && dr;
				}

				bool dl = true, dr = true;
				const float l = operand_value(*n.lhs, ctx, dl);
				const float r = operand_value(*n.rhs, ctx, dr);
				switch (n.op)
				{
				case Op::equal: out = (l == r) ? 1.0f : 0.0f; break;
				case Op::not_equal: out = (l != r) ? 1.0f : 0.0f; break;
				case Op::less: out = (l < r) ? 1.0f : 0.0f; break;
				case Op::less_equal: out = (l <= r) ? 1.0f : 0.0f; break;
				case Op::greater: out = (l > r) ? 1.0f : 0.0f; break;
				case Op::greater_equal: out = (l >= r) ? 1.0f : 0.0f; break;
				case Op::add: out = l + r; break;
				case Op::sub: out = l - r; break;
				case Op::mul: out = l * r; break;
				case Op::div: out = (r == 0.0f) ? 0.0f : l / r; break;
				default: out = 0.0f; break;
				}
				return dl && dr;
			}
			}
			return false;
		}
	}

	bool eval(const Node &node, const EvalContext &ctx, float &out)
	{
		return eval_node(node, ctx, out);
	}
}

// WWMI-DX12: thin formatted logging wrapper over the ReShade addon log API.
#pragma once

#include "reshade.hpp"
#include <cstdarg>
#include <cstdio>

namespace wwmi::log
{
	constexpr size_t k_max_message = 1024;

	inline void vmessage(reshade::log::level level, const char *fmt, va_list ap)
	{
		char buf[k_max_message];
		const int n = vsnprintf(buf, sizeof(buf), fmt, ap);
		if (n <= 0)
			return;
		buf[sizeof(buf) - 1] = '\0';
		reshade::log::message(level, buf);
	}

	inline void error(const char *fmt, ...)
	{
		va_list ap;
		va_start(ap, fmt);
		vmessage(reshade::log::level::error, fmt, ap);
		va_end(ap);
	}

	inline void warn(const char *fmt, ...)
	{
		va_list ap;
		va_start(ap, fmt);
		vmessage(reshade::log::level::warning, fmt, ap);
		va_end(ap);
	}

	inline void info(const char *fmt, ...)
	{
		va_list ap;
		va_start(ap, fmt);
		vmessage(reshade::log::level::info, fmt, ap);
		va_end(ap);
	}

	// Debug-level logging is only emitted when ReShade's verbose log is enabled.
	inline void debug(const char *fmt, ...)
	{
		va_list ap;
		va_start(ap, fmt);
		vmessage(reshade::log::level::debug, fmt, ap);
		va_end(ap);
	}
}

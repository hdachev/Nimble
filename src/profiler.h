#pragma once

#include "ogl.h"
#include "timer.h"
#include <memory>
#include <string>

#define NIMBLE_SCOPED_SAMPLE(name) nimble::profiler::ScopedProfile __FILE__##__LINE__(name)

namespace nimble
{
namespace profiler
{
	struct ScopedProfile
	{
		ScopedProfile(std::string name);
		~ScopedProfile();

		std::string m_name;
	};

	extern void initialize();
	extern void shutdown();
    extern void begin_sample(std::string name);
    extern void end_sample(std::string name);
	extern void begin_frame();
    extern void end_frame();
    extern void ui();
};
} // namespace nimble
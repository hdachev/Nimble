#pragma once

#include <ogl.h>
#include <unordered_map>
#include <string>

#define NUM_BUFFERED_QUERIES 3

struct ProfileScope
{
	dw::Query queries[NUM_BUFFERED_QUERIES];
	float	  last_result = 0;
	uint32_t  index = 0;
};

class GPUProfiler
{
public:
	static void shutdown();
	static void begin(std::string name);
	static void end(std::string name);
	static float result(std::string name);
private:
	static std::unordered_map<std::string, ProfileScope*> m_scopes;
};
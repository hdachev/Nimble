#include "gpu_profiler.h"
#include "macros.h"

namespace nimble
{
	std::unordered_map<std::string, ProfileScope*> GPUProfiler::m_scopes;

	// -----------------------------------------------------------------------------------------------------------------------------------

	void GPUProfiler::shutdown()
	{
		for (auto pair : m_scopes)
		{
			NIMBLE_SAFE_DELETE(pair.second);
		}
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	void GPUProfiler::begin(std::string name)
	{
		if (m_scopes.find(name) == m_scopes.end())
			m_scopes[name] = new ProfileScope();

		ProfileScope* scope = m_scopes[name];

		uint64_t result = 0;
		scope->queries[scope->index].result_64(&result);
		scope->last_result = result / 1000000.0f;

		scope->queries[scope->index].begin(GL_TIME_ELAPSED);
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	void GPUProfiler::end(std::string name)
	{
		ProfileScope* scope = m_scopes[name];
		scope->queries[scope->index].end(GL_TIME_ELAPSED);

		scope->index++;

		if (scope->index == NUM_BUFFERED_QUERIES)
			scope->index = 0;
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	float GPUProfiler::result(std::string name)
	{
		if (m_scopes.find(name) == m_scopes.end())
			m_scopes[name] = new ProfileScope();

		return m_scopes[name]->last_result;
	}

	// -----------------------------------------------------------------------------------------------------------------------------------
}
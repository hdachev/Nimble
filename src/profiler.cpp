#include "profiler.h"
#include "macros.h"

namespace nimble
{
	std::unordered_map<std::string, std::unique_ptr<ProfileScope>> Profiler::m_scopes;

	// -----------------------------------------------------------------------------------------------------------------------------------

	void Profiler::shutdown()
	{
		for (auto& pair : m_scopes)
			pair.second.reset();

		m_scopes.clear();
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	void Profiler::begin_sample(std::string name)
	{
		if (m_scopes.find(name) == m_scopes.end())
			m_scopes[name] = std::make_unique<ProfileScope>();

		auto& scope = m_scopes[name];

		uint64_t result = 0;
		scope->queries[scope->index].result_64(&result);
		scope->last_result_gpu = result / 1000000.0f;
		scope->last_result_cpu = static_cast<float>(scope->timer.elapsed_time_milisec());

		scope->queries[scope->index].begin(GL_TIME_ELAPSED);
		scope->timer.start();
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	void Profiler::end_sample(std::string name)
	{
		auto& scope = m_scopes[name];
		scope->queries[scope->index].end(GL_TIME_ELAPSED);
		scope->timer.stop();

		scope->index++;

		if (scope->index == NUM_BUFFERED_QUERIES)
			scope->index = 0;
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	void Profiler::result(std::string name, float& cpu_sample, float& gpu_sample)
	{
		if (m_scopes.find(name) == m_scopes.end())
			m_scopes[name] = std::make_unique<ProfileScope>();

		cpu_sample = m_scopes[name]->last_result_cpu;
		gpu_sample = m_scopes[name]->last_result_gpu;
	}

	// -----------------------------------------------------------------------------------------------------------------------------------
}
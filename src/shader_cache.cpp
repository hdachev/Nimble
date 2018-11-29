#include "shader_cache.h"
#include "shader_library.h"

namespace nimble
{
	std::unordered_map<std::string, std::weak_ptr<ShaderLibrary>> ShaderCache::m_library_cache;

	// -----------------------------------------------------------------------------------------------------------------------------------

	std::shared_ptr<ShaderLibrary> ShaderCache::load_library(const std::string& vs, const std::string& fs)
	{
		std::string id = "vs:";
		id += vs;
		id += "-fs:";
		id += fs;

		if (m_library_cache.find(id) != m_library_cache.end() && !m_library_cache[id].expired())
			return m_library_cache[id].lock();
		else
		{
			auto library = std::make_shared<ShaderLibrary>(vs, fs);
			m_library_cache[id] = library;

			return library;
		}
	}

	// -----------------------------------------------------------------------------------------------------------------------------------
}
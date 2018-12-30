#pragma once

#include <unordered_map>
#include <memory>
#include <string>

namespace nimble
{
	class ShaderLibrary;

	class ShaderCache
	{
	public:
		static void shutdown();
		static std::shared_ptr<ShaderLibrary> load_library(const std::string& vs, const std::string& fs);

	private:
		static std::unordered_map<std::string, std::weak_ptr<ShaderLibrary>> m_library_cache;
	};
}
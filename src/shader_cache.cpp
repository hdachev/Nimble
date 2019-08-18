#include "shader_cache.h"
#include "geometry_shader_library.h"

namespace nimble
{
// -----------------------------------------------------------------------------------------------------------------------------------

void ShaderCache::shutdown()
{
    for (auto& pair : m_library_cache)
        pair.second.reset();

    for (auto& pair : m_geometry_library_cache)
        pair.second.reset();
}

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
        auto library                 = std::make_shared<ShaderLibrary>(vs, fs);
        m_library_cache[id] = library;

        return library;
    }
}

// -----------------------------------------------------------------------------------------------------------------------------------

std::shared_ptr<GeometryShaderLibrary> ShaderCache::load_geometry_library(const std::string& vs, const std::string& fs)
{
    std::string id = "vs:";
    id += vs;
    id += "-fs:";
    id += fs;

    if (m_geometry_library_cache.find(id) != m_geometry_library_cache.end() && !m_geometry_library_cache[id].expired())
        return m_geometry_library_cache[id].lock();
    else
    {
        auto library                 = std::make_shared<GeometryShaderLibrary>(vs, fs);
        m_geometry_library_cache[id] = library;

        return library;
    }
}

// -----------------------------------------------------------------------------------------------------------------------------------
} // namespace nimble
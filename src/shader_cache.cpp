#include "shader_cache.h"
#include "generic_shader_library.h"
#include "generated_shader_library.h"

namespace nimble
{
// -----------------------------------------------------------------------------------------------------------------------------------

void ShaderCache::shutdown()
{
    for (auto& pair : m_generic_library_cache)
        pair.second.reset();

    for (auto& pair : m_generated_library_cache)
        pair.second.reset();
}

// -----------------------------------------------------------------------------------------------------------------------------------

std::shared_ptr<GenericShaderLibrary> ShaderCache::load_generic_library(std::vector<std::pair<GLenum, std::string>> shaders)
{
    std::string id = "";

    for (auto& pair : shaders)
    {
        if (pair.first == GL_VERTEX_SHADER)
            id += "-vs:";
        else if (pair.first == GL_FRAGMENT_SHADER)
            id += "-fs:";
        else if (pair.first == GL_COMPUTE_SHADER)
            id += "-cs:";

        id += pair.second;
    }

    if (m_generic_library_cache.find(id) != m_generic_library_cache.end() && !m_generic_library_cache[id].expired())
        return m_generic_library_cache[id].lock();
    else
    {
        auto library                = std::make_shared<GenericShaderLibrary>(shaders);
        m_generic_library_cache[id] = library;

        return library;
    }
}

// -----------------------------------------------------------------------------------------------------------------------------------

std::shared_ptr<GeneratedShaderLibrary> ShaderCache::load_generated_library(const std::string& vs, const std::string& fs)
{
    std::string id = "vs:";
    id += vs;
    id += "-fs:";
    id += fs;

    if (m_generated_library_cache.find(id) != m_generated_library_cache.end() && !m_generated_library_cache[id].expired())
        return m_generated_library_cache[id].lock();
    else
    {
        auto library                  = std::make_shared<GeneratedShaderLibrary>(vs, fs);
        m_generated_library_cache[id] = library;

        return library;
    }
}

// -----------------------------------------------------------------------------------------------------------------------------------
} // namespace nimble
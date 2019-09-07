#pragma once

#include <unordered_map>
#include <memory>
#include <string>
#include "ogl.h"

namespace nimble
{
class GenericShaderLibrary;
class GeneratedShaderLibrary;

class ShaderCache
{
public:
    void                                    shutdown();
    void                                    clear_generated_cache();
    std::shared_ptr<GenericShaderLibrary>   load_generic_library(std::vector<std::pair<GLenum, std::string>> shaders);
    std::shared_ptr<GeneratedShaderLibrary> load_generated_library(const std::string& vs, const std::string& fs);

private:
    std::unordered_map<std::string, std::weak_ptr<GenericShaderLibrary>>   m_generic_library_cache;
    std::unordered_map<std::string, std::weak_ptr<GeneratedShaderLibrary>> m_generated_library_cache;
};
} // namespace nimble
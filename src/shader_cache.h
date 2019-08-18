#pragma once

#include <unordered_map>
#include <memory>
#include <string>
#include "ogl.h"

namespace nimble
{
class ShaderLibrary;
class GeometryShaderLibrary;

class ShaderCache
{
public:
    void                           shutdown();
    std::shared_ptr<ShaderLibrary> load_library(std::vector<std::pair<GLenum, std::string>> shaders);
    std::shared_ptr<GeometryShaderLibrary> load_geometry_library(const std::string& vs, const std::string& fs);

private:
    std::unordered_map<std::string, std::weak_ptr<ShaderLibrary>> m_library_cache;
    std::unordered_map<std::string, std::weak_ptr<GeometryShaderLibrary>> m_geometry_library_cache;
};
} // namespace nimble
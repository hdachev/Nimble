#pragma once

#include "ogl.h"
#include "mesh.h"
#include "static_hash_map.h"
#include "shader_key.h"

namespace nimble
{
#define MAX_SHADER_CACHE_SIZE 10000
#define MAX_PROGRAM_CACHE_SIZE 10000

class ShadowRenderGraph;

class GeometryShaderLibrary
{
public:
    GeometryShaderLibrary(const std::string& vs, const std::string& fs);
    ~GeometryShaderLibrary();

    Program* lookup_program(const ProgramKey& key);
    Program* create_program(const MeshType& type, const uint32_t& flags, const std::shared_ptr<Material>& material, std::shared_ptr<ShadowRenderGraph> directional_light_render_graph, std::shared_ptr<ShadowRenderGraph> spot_light_render_graph, std::shared_ptr<ShadowRenderGraph> point_light_render_graph);

private:
    StaticHashMap<uint64_t, Program*, MAX_PROGRAM_CACHE_SIZE> m_program_cache;
    std::unordered_map<uint64_t, Shader*>                     m_vs_cache;
    std::unordered_map<uint64_t, Shader*>                     m_fs_cache;
    std::string                                               m_vs_template_source;
    std::string                                               m_fs_template_source;
    std::string                                               m_vs_template_includes;
    std::string                                               m_fs_template_includes;
    std::string                                               m_vs_template_defines;
    std::string                                               m_fs_template_defines;
};
} // namespace nimble
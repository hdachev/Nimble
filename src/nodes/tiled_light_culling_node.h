#pragma once

#include "../render_node.h"

namespace nimble
{
#define TILE_SIZE 16
#define MAX_LIGHTS_PER_TILE 1024

struct TileFrustum
{
    glm::vec4 planes[4];
};

class TiledLightCullingNode : public RenderNode
{
public:
    TiledLightCullingNode(RenderGraph* graph);
    ~TiledLightCullingNode();

    void        declare_connections() override;
    bool        initialize(Renderer* renderer, ResourceManager* res_mgr) override;
    void        execute(double delta, Renderer* renderer, Scene* scene, View* view) override;
    void        shutdown() override;
    std::string name() override;
    void        on_window_resized(const uint32_t& w, const uint32_t& h) override;

private:
    void precompute_frustum(Renderer* renderer, View* view);
    void cull_lights(Renderer* renderer, View* view);

private:
    std::shared_ptr<ShaderStorageBuffer> m_precomputed_frustums;
    std::shared_ptr<ShaderStorageBuffer> m_culled_light_indices;
    std::shared_ptr<ShaderStorageBuffer> m_light_grid;
    std::shared_ptr<ShaderStorageBuffer> m_light_counter;
    std::shared_ptr<RenderTarget>        m_depth_rt;

    std::shared_ptr<Shader>  m_tiled_light_cull_cs;
    std::shared_ptr<Program> m_tiled_light_cull_program;

    std::shared_ptr<Shader>  m_reset_counter_cs;
    std::shared_ptr<Program> m_reset_counter_program;

    std::shared_ptr<Shader>  m_frustum_precompute_cs;
    std::shared_ptr<Program> m_frustum_precompute_program;

    bool m_requires_precompute = true;
};

DECLARE_RENDER_NODE_FACTORY(TiledLightCullingNode);
} // namespace nimble

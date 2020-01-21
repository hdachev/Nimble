#pragma once

#include "../render_node.h"

namespace nimble
{
#define TILE_SIZE 16
#define MAX_POINT_LIGHTS_PER_TILE 512
#define MAX_SPOT_LIGHTS_PER_TILE 512

struct LightIndices
{
    uint32_t num_point_lights;
    uint32_t num_spot_lights;
    uint32_t point_light_indices[MAX_POINT_LIGHTS_PER_TILE];
    uint32_t spot_light_indices[MAX_SPOT_LIGHTS_PER_TILE];
};

struct TileFrustum
{
    glm::vec4 planes[6];
    glm::vec4 points[8];
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
    void precompute_frustums(Renderer* renderer, View* view);
    void cull_lights(Renderer* renderer, View* view);

private:
    std::shared_ptr<ShaderStorageBuffer> m_culled_light_indices;
    std::shared_ptr<ShaderStorageBuffer> m_tile_frustums;
    std::shared_ptr<RenderTarget>        m_hiz_depth_rt;

	std::shared_ptr<Shader>  m_tiled_light_cull_cs;
    std::shared_ptr<Program> m_tiled_light_cull_program;
    std::shared_ptr<Shader>  m_frustum_precompute_cs;
    std::shared_ptr<Program> m_frustum_precompute_program;

    bool m_should_precompute = true;
};

DECLARE_RENDER_NODE_FACTORY(TiledLightCullingNode);
} // namespace nimble

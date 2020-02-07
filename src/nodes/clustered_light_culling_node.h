#pragma once

#include "../render_node.h"

namespace nimble
{
#define CLUSTER_GRID_DIM_X 16
#define CLUSTER_GRID_DIM_Y 8
#define CLUSTER_GRID_DIM_Z 24
#define MAX_LIGHTS_PER_CLUSTER 1024

struct ClusterAABB
{
    glm::vec4 aabb_min;
    glm::vec4 aabb_max;
};

class ClusteredLightCullingNode : public RenderNode
{
public:
    ClusteredLightCullingNode(RenderGraph* graph);
    ~ClusteredLightCullingNode();

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
    std::shared_ptr<ShaderStorageBuffer> m_precomputed_clusters;
    std::shared_ptr<ShaderStorageBuffer> m_culled_light_indices;
    std::shared_ptr<ShaderStorageBuffer> m_light_grid;
    std::shared_ptr<ShaderStorageBuffer> m_light_counter;
    std::shared_ptr<RenderTarget>        m_depth_rt;

	std::shared_ptr<Shader>  m_clustered_light_cull_cs;
    std::shared_ptr<Program> m_clustered_light_cull_program;

    std::shared_ptr<Shader>  m_reset_counter_cs;
    std::shared_ptr<Program> m_reset_counter_program;

    std::shared_ptr<Shader>  m_cluster_precompute_cs;
    std::shared_ptr<Program> m_cluster_precompute_program;

    bool m_requires_precompute = true;
};

DECLARE_RENDER_NODE_FACTORY(ClusteredLightCullingNode);
} // namespace nimble

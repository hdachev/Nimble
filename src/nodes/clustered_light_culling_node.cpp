#include "clustered_light_culling_node.h"
#include "../render_graph.h"
#include "../resource_manager.h"
#include "../renderer.h"
#include "../profiler.h"
#include "../renderer.h"

namespace nimble
{
DEFINE_RENDER_NODE_FACTORY(ClusteredLightCullingNode)

// -----------------------------------------------------------------------------------------------------------------------------------

ClusteredLightCullingNode::ClusteredLightCullingNode(RenderGraph* graph) :
    RenderNode(graph)
{
}

// -----------------------------------------------------------------------------------------------------------------------------------

ClusteredLightCullingNode::~ClusteredLightCullingNode()
{
}

// -----------------------------------------------------------------------------------------------------------------------------------

void ClusteredLightCullingNode::declare_connections()
{
    register_input_render_target("Depth");

    on_window_resized(m_graph->window_width(), m_graph->window_height());
}

// -----------------------------------------------------------------------------------------------------------------------------------

bool ClusteredLightCullingNode::initialize(Renderer* renderer, ResourceManager* res_mgr)
{
    m_depth_rt = find_input_render_target("Depth");

    m_clustered_light_cull_cs   = res_mgr->load_shader("shader/clustered/light_culling_cs.glsl", GL_COMPUTE_SHADER);
    m_cluster_precompute_cs = res_mgr->load_shader("shader/clustered/precompute_clusters_cs.glsl", GL_COMPUTE_SHADER);
    m_reset_counter_cs      = res_mgr->load_shader("shader/tiled/reset_counter_cs.glsl", GL_COMPUTE_SHADER);

    if (m_clustered_light_cull_cs && m_cluster_precompute_cs && m_reset_counter_cs)
    {
        m_clustered_light_cull_program = renderer->create_program({ m_clustered_light_cull_cs });
        m_cluster_precompute_program   = renderer->create_program({ m_cluster_precompute_cs });
        m_reset_counter_program      = renderer->create_program({ m_reset_counter_cs });

        return true;
    }

    return false;
}

// -----------------------------------------------------------------------------------------------------------------------------------

void ClusteredLightCullingNode::execute(double delta, Renderer* renderer, Scene* scene, View* view)
{
    if (m_requires_precompute)
        precompute_frustum(renderer, view);

    cull_lights(renderer, view);
}

// -----------------------------------------------------------------------------------------------------------------------------------

void ClusteredLightCullingNode::on_window_resized(const uint32_t& w, const uint32_t& h)
{
    uint32_t tile_count_x = ceil(float(w) / float(CLUSTER_TILE_SIZE));
    uint32_t tile_count_y = ceil(float(h) / float(CLUSTER_TILE_SIZE));

    m_culled_light_indices = std::make_shared<ShaderStorageBuffer>(0, sizeof(glm::uvec4) * tile_count_x * tile_count_y * CLUSTER_Z_SLICES * MAX_LIGHTS_PER_CLUSTER);
    m_light_grid           = std::make_shared<ShaderStorageBuffer>(0, sizeof(glm::uvec4) * tile_count_x * tile_count_y * CLUSTER_Z_SLICES);
    m_light_counter        = std::make_shared<ShaderStorageBuffer>(0, sizeof(glm::uvec4));
    m_precomputed_clusters = std::make_shared<ShaderStorageBuffer>(0, sizeof(Frustum) * tile_count_x * tile_count_y * CLUSTER_Z_SLICES);

    register_output_buffer("LightIndices", m_culled_light_indices);
    register_output_buffer("LightGrid", m_light_grid);

    m_requires_precompute = true;
}

// -----------------------------------------------------------------------------------------------------------------------------------

void ClusteredLightCullingNode::precompute_frustum(Renderer* renderer, View* view)
{
    m_requires_precompute = false;

    m_cluster_precompute_program->use();

    uint32_t tile_count_x = ceil(float(m_graph->window_width()) / float(CLUSTER_TILE_SIZE));
    uint32_t tile_count_y = ceil(float(m_graph->window_height()) / float(CLUSTER_TILE_SIZE));

    m_precomputed_clusters->bind_base(3);

    m_cluster_precompute_program->set_uniform("u_TileCountX", (int32_t)tile_count_x);
    m_cluster_precompute_program->set_uniform("u_TileCountY", (int32_t)tile_count_y);

    uint32_t dispatch_size_x = ceil(float(tile_count_x) / float(CLUSTER_TILE_SIZE));
    uint32_t dispatch_size_y = ceil(float(tile_count_y) / float(CLUSTER_TILE_SIZE));

    dispatch_compute(dispatch_size_x, dispatch_size_y, CLUSTER_Z_SLICES, renderer, view, m_cluster_precompute_program.get(), 0, NODE_USAGE_PER_VIEW_UBO | NODE_USAGE_POINT_LIGHTS | NODE_USAGE_SPOT_LIGHTS);

    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
}

// -----------------------------------------------------------------------------------------------------------------------------------

void ClusteredLightCullingNode::cull_lights(Renderer* renderer, View* view)
{
    // Reset counters

    m_reset_counter_program->use();

    m_light_counter->bind_base(6);

    glDispatchCompute(1, 1, 1);

    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    // Cull lights

    m_clustered_light_cull_program->use();

    uint32_t tile_count_x = ceil(float(m_graph->window_width()) / float(CLUSTER_TILE_SIZE));
    uint32_t tile_count_y = ceil(float(m_graph->window_height()) / float(CLUSTER_TILE_SIZE));

    m_precomputed_clusters->bind_base(3);
    m_culled_light_indices->bind_base(4);
    m_light_grid->bind_base(5);
    m_light_counter->bind_base(6);

    dispatch_compute(tile_count_x, tile_count_y, CLUSTER_Z_SLICES, renderer, view, m_clustered_light_cull_program.get(), 0, NODE_USAGE_PER_VIEW_UBO | NODE_USAGE_POINT_LIGHTS | NODE_USAGE_SPOT_LIGHTS);

    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
}

// -----------------------------------------------------------------------------------------------------------------------------------

void ClusteredLightCullingNode::shutdown()
{
}

// -----------------------------------------------------------------------------------------------------------------------------------

std::string ClusteredLightCullingNode::name()
{
    return "Clustered Light Culling";
}

// -----------------------------------------------------------------------------------------------------------------------------------
} // namespace nimble
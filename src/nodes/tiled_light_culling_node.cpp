#include "tiled_light_culling_node.h"
#include "../render_graph.h"
#include "../resource_manager.h"
#include "../renderer.h"
#include "../profiler.h"
#include "../renderer.h"

namespace nimble
{
DEFINE_RENDER_NODE_FACTORY(TiledLightCullingNode)

// -----------------------------------------------------------------------------------------------------------------------------------

TiledLightCullingNode::TiledLightCullingNode(RenderGraph* graph) :
    RenderNode(graph)
{
}

// -----------------------------------------------------------------------------------------------------------------------------------

TiledLightCullingNode::~TiledLightCullingNode()
{
}

// -----------------------------------------------------------------------------------------------------------------------------------

void TiledLightCullingNode::declare_connections()
{
    register_input_render_target("Depth");

    on_window_resized(m_graph->window_width(), m_graph->window_height());
}

// -----------------------------------------------------------------------------------------------------------------------------------

bool TiledLightCullingNode::initialize(Renderer* renderer, ResourceManager* res_mgr)
{
    m_depth_rt = find_input_render_target("Depth");

    m_tiled_light_cull_cs = res_mgr->load_shader("shader/tiled/light_culling_cs.glsl", GL_COMPUTE_SHADER);
    m_frustum_precompute_cs = res_mgr->load_shader("shader/tiled/precompute_frustums_cs.glsl", GL_COMPUTE_SHADER);

    if (m_tiled_light_cull_cs && m_frustum_precompute_cs)
    {
        m_tiled_light_cull_program = renderer->create_program({ m_tiled_light_cull_cs });
        m_frustum_precompute_program = renderer->create_program({ m_frustum_precompute_cs });

        return true;
    }

    return false;
}

// -----------------------------------------------------------------------------------------------------------------------------------

void TiledLightCullingNode::execute(double delta, Renderer* renderer, Scene* scene, View* view)
{
    if (m_requires_precompute)
        precompute_frustum(renderer, view);

    cull_lights(renderer, view);
}

// -----------------------------------------------------------------------------------------------------------------------------------

void TiledLightCullingNode::on_window_resized(const uint32_t& w, const uint32_t& h)
{
    uint32_t tile_count_x = ceil(float(w) / float(TILE_SIZE));
    uint32_t tile_count_y = ceil(float(h) / float(TILE_SIZE));

    m_culled_light_indices = std::make_shared<ShaderStorageBuffer>(0, sizeof(LightIndices) * tile_count_x * tile_count_y);
    m_precomputed_frustums   = std::make_shared<ShaderStorageBuffer>(0, sizeof(Frustum) * tile_count_x * tile_count_y);

    register_output_buffer("LightIndices", m_culled_light_indices);

    m_requires_precompute = true;
}

// -----------------------------------------------------------------------------------------------------------------------------------

void TiledLightCullingNode::precompute_frustum(Renderer* renderer, View* view)
{
    m_requires_precompute = false;

    m_frustum_precompute_program->use();

    uint32_t tile_count_x = ceil(float(m_graph->window_width()) / float(TILE_SIZE));
    uint32_t tile_count_y = ceil(float(m_graph->window_height()) / float(TILE_SIZE));

    m_precomputed_frustums->bind_base(3);
   
    m_frustum_precompute_program->set_uniform("u_TileCountX", (int32_t)tile_count_x);
    m_frustum_precompute_program->set_uniform("u_TileCountY", (int32_t)tile_count_y);

    uint32_t dispatch_size_x = ceil(float(tile_count_x) / float(TILE_SIZE));
    uint32_t dispatch_size_y = ceil(float(tile_count_y) / float(TILE_SIZE));

    dispatch_compute(dispatch_size_x, dispatch_size_y, 1, renderer, view, m_frustum_precompute_program.get(), 0, NODE_USAGE_PER_VIEW_UBO | NODE_USAGE_POINT_LIGHTS | NODE_USAGE_SPOT_LIGHTS);

    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
}

// -----------------------------------------------------------------------------------------------------------------------------------

void TiledLightCullingNode::cull_lights(Renderer* renderer, View* view)
{
    m_tiled_light_cull_program->use();

    uint32_t tile_count_x = ceil(float(m_graph->window_width()) / float(TILE_SIZE));
    uint32_t tile_count_y = ceil(float(m_graph->window_height()) / float(TILE_SIZE));

    m_precomputed_frustums->bind_base(3);
    m_culled_light_indices->bind_base(4);

    if (m_tiled_light_cull_program->set_uniform("s_Depth", 0) && m_depth_rt)
        m_depth_rt->texture->bind(0);

    dispatch_compute(tile_count_x, tile_count_y, 1, renderer, view, m_tiled_light_cull_program.get(), 0, NODE_USAGE_PER_VIEW_UBO | NODE_USAGE_POINT_LIGHTS | NODE_USAGE_SPOT_LIGHTS);

    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
}

// -----------------------------------------------------------------------------------------------------------------------------------

void TiledLightCullingNode::shutdown()
{
}

// -----------------------------------------------------------------------------------------------------------------------------------

std::string TiledLightCullingNode::name()
{
    return "Tiled Light Culling";
}

// -----------------------------------------------------------------------------------------------------------------------------------
} // namespace nimble
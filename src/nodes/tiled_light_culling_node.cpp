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
    register_input_render_target("HiZDepth");

    uint32_t tile_count_x = ceil(m_graph->window_width() / TILE_SIZE);
    uint32_t tile_count_y = ceil(m_graph->window_height() / TILE_SIZE);
    
	m_culled_light_indices = register_output_buffer("LightIndices", 0, sizeof(LightIndices) * tile_count_x * tile_count_y);
    m_tile_frustums        = std::make_shared<ShaderStorageBuffer>(0, sizeof(TileFrustum) * tile_count_x * tile_count_y);
}

// -----------------------------------------------------------------------------------------------------------------------------------

bool TiledLightCullingNode::initialize(Renderer* renderer, ResourceManager* res_mgr)
{
    m_hiz_depth_rt = find_input_render_target("HiZDepth");

	m_tiled_light_cull_cs = res_mgr->load_shader("shader/post_process/light_culling/tiled_light_culling_cs.glsl", GL_COMPUTE_SHADER);
    m_frustum_precompute_cs = res_mgr->load_shader("shader/post_process/light_culling/frustum_precompute_cs.glsl", GL_COMPUTE_SHADER);

    if (m_tiled_light_cull_cs)
        m_tiled_light_cull_program = renderer->create_program({ m_tiled_light_cull_cs });
    else
        return false;

    if (m_frustum_precompute_cs)
        m_frustum_precompute_program = renderer->create_program({ m_frustum_precompute_cs });
    else
        return false;

    return true;
}

// -----------------------------------------------------------------------------------------------------------------------------------

void TiledLightCullingNode::execute(double delta, Renderer* renderer, Scene* scene, View* view)
{
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
#include "tiled_light_culling_node.h"
#include "../render_graph.h"
#include "../resource_manager.h"
#include "../renderer.h"
#include "../profiler.h"

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
    
	m_culled_light_indices = register_output_buffer("LightIndices", 0, 0);
}

// -----------------------------------------------------------------------------------------------------------------------------------

bool TiledLightCullingNode::initialize(Renderer* renderer, ResourceManager* res_mgr)
{
    m_hiz_depth_rt = find_input_render_target("HiZDepth");

	m_tiled_light_cull_cs = res_mgr->load_shader("shader/post_process/light_culling/tiled_light_culling_cs.glsl", GL_COMPUTE_SHADER);

    if (m_tiled_light_cull_cs)
    {
        m_tiled_light_cull_program = renderer->create_program({ m_tiled_light_cull_cs });
        return true;
    }
    else
        return false;
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
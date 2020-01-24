#include "depth_prepass_node.h"
#include "../render_graph.h"
#include "../renderer.h"

namespace nimble
{
DEFINE_RENDER_NODE_FACTORY(DepthPrepassNode)

// -----------------------------------------------------------------------------------------------------------------------------------

DepthPrepassNode::DepthPrepassNode(RenderGraph* graph) :
    RenderNode(graph)
{
}

// -----------------------------------------------------------------------------------------------------------------------------------

DepthPrepassNode::~DepthPrepassNode()
{
}

// -----------------------------------------------------------------------------------------------------------------------------------

void DepthPrepassNode::declare_connections()
{
    m_depth_rt = register_scaled_output_render_target("Depth", 1.0f, 1.0f, GL_TEXTURE_2D, GL_DEPTH_COMPONENT32F, GL_DEPTH_COMPONENT, GL_FLOAT);
}

// -----------------------------------------------------------------------------------------------------------------------------------

bool DepthPrepassNode::initialize(Renderer* renderer, ResourceManager* res_mgr)
{
    m_library = renderer->shader_cache().load_generated_library("shader/shadows/directional_light/shadow_map/directional_light_depth_vs.glsl", "shader/shadows/directional_light/shadow_map/directional_light_depth_fs.glsl");

    m_depth_rtv = RenderTargetView(0, 0, 0, m_depth_rt->texture);

    return true;
}

// -----------------------------------------------------------------------------------------------------------------------------------

void DepthPrepassNode::execute(double delta, Renderer* renderer, Scene* scene, View* view)
{
    renderer->bind_render_targets(0, nullptr, &m_depth_rtv);

    glViewport(0, 0, m_graph->window_width(), m_graph->window_height());

    glEnable(GL_DEPTH_TEST);

    glCullFace(GL_BACK);

    glClear(GL_DEPTH_BUFFER_BIT);

    render_scene(renderer, scene, view, m_library.get(), NODE_USAGE_SHADOW_MAP);
}

// -----------------------------------------------------------------------------------------------------------------------------------

void DepthPrepassNode::shutdown()
{
}

// -----------------------------------------------------------------------------------------------------------------------------------

std::string DepthPrepassNode::name()
{
    return "Depth Prepass";
}

// -----------------------------------------------------------------------------------------------------------------------------------
} // namespace nimble
#include "g_buffer_node.h"
#include "../renderer.h"
#include "../render_graph.h"

namespace nimble
{
DEFINE_RENDER_NODE_FACTORY(GBufferNode)

// -----------------------------------------------------------------------------------------------------------------------------------

GBufferNode::GBufferNode(RenderGraph* graph) :
    RenderNode(graph)
{
}

// -----------------------------------------------------------------------------------------------------------------------------------

GBufferNode::~GBufferNode()
{
}

// -----------------------------------------------------------------------------------------------------------------------------------

void GBufferNode::declare_connections()
{
}

// -----------------------------------------------------------------------------------------------------------------------------------

bool GBufferNode::initialize(Renderer* renderer, ResourceManager* res_mgr)
{
    m_library = renderer->shader_cache().load_library("shader/g_buffer/g_buffer_vs.glsl", "shader/g_buffer/g_buffer_vs.glsl");

    return true;
}

// -----------------------------------------------------------------------------------------------------------------------------------

void GBufferNode::shutdown()
{
}

// -----------------------------------------------------------------------------------------------------------------------------------

std::string GBufferNode::name()
{
    return "G-Buffer";
}
// -----------------------------------------------------------------------------------------------------------------------------------

void GBufferNode::execute(Renderer* renderer, Scene* scene, View* view)
{
    renderer->bind_render_targets(3, m_gbuffer_rtv, &m_depth_rtv);
    glViewport(0, 0, m_graph->window_width(), m_graph->window_height());

    glEnable(GL_DEPTH_TEST);

    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);

    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    render_scene(renderer, scene, view, m_library.get());
}

// -----------------------------------------------------------------------------------------------------------------------------------
} // namespace nimble
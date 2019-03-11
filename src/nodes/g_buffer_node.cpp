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
    m_gbuffer1_rt = register_scaled_output_render_target("G-Buffer1", 1.0f, 1.0f, GL_TEXTURE_2D, GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE);
    m_gbuffer2_rt = register_scaled_output_render_target("G-Buffer2", 1.0f, 1.0f, GL_TEXTURE_2D, GL_RGBA16F, GL_RGBA, GL_HALF_FLOAT);
    m_gbuffer3_rt = register_scaled_output_render_target("G-Buffer3", 1.0f, 1.0f, GL_TEXTURE_2D, GL_RGBA16F, GL_RGBA, GL_HALF_FLOAT);
    m_gbuffer4_rt = register_scaled_output_render_target("G-Buffer4", 1.0f, 1.0f, GL_TEXTURE_2D, GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE);
    m_depth_rt    = register_scaled_output_render_target("Depth", 1.0f, 1.0f, GL_TEXTURE_2D, GL_DEPTH_COMPONENT32F, GL_DEPTH_COMPONENT, GL_FLOAT);
}

// -----------------------------------------------------------------------------------------------------------------------------------

bool GBufferNode::initialize(Renderer* renderer, ResourceManager* res_mgr)
{
    m_library = renderer->shader_cache().load_library("shader/g_buffer/g_buffer_vs.glsl", "shader/g_buffer/g_buffer_fs.glsl");

    m_gbuffer_rtv[0] = RenderTargetView(0, 0, 0, m_gbuffer1_rt->texture);
    m_gbuffer_rtv[1] = RenderTargetView(0, 0, 0, m_gbuffer2_rt->texture);
    m_gbuffer_rtv[2] = RenderTargetView(0, 0, 0, m_gbuffer3_rt->texture);
    m_gbuffer_rtv[3] = RenderTargetView(0, 0, 0, m_gbuffer4_rt->texture);
    m_depth_rtv      = RenderTargetView(0, 0, 0, m_depth_rt->texture);

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

void GBufferNode::execute(double delta, Renderer* renderer, Scene* scene, View* view)
{
    renderer->bind_render_targets(4, m_gbuffer_rtv, &m_depth_rtv);
    glViewport(0, 0, m_graph->window_width(), m_graph->window_height());

    glEnable(GL_DEPTH_TEST);

    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);

    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    render_scene(renderer, scene, view, m_library.get(), NODE_USAGE_DEFAULT);
}

// -----------------------------------------------------------------------------------------------------------------------------------
} // namespace nimble
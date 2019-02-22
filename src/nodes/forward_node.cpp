#include "forward_node.h"
#include "../render_graph.h"
#include "../renderer.h"

namespace nimble
{
DEFINE_RENDER_NODE_FACTORY(ForwardNode)

// -----------------------------------------------------------------------------------------------------------------------------------

ForwardNode::ForwardNode(RenderGraph* graph) :
    RenderNode(graph)
{
}

// -----------------------------------------------------------------------------------------------------------------------------------

ForwardNode::~ForwardNode()
{
}

// -----------------------------------------------------------------------------------------------------------------------------------

void ForwardNode::declare_connections()
{
    //m_color_rt = register_render_target("Color", 1.0f, 1.0f, GL_TEXTURE_2D, GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE);
    //m_depth_rt = register_render_target("Depth", 1.0f, 1.0f, GL_TEXTURE_2D, GL_DEPTH_COMPONENT32F, GL_DEPTH_COMPONENT, GL_FLOAT);
    //m_velocity_rt = register_render_target("Velocity", 1.0f, 1.0f, GL_TEXTURE_2D, GL_RG16F, GL_RG, GL_HALF_FLOAT);
}

// -----------------------------------------------------------------------------------------------------------------------------------

bool ForwardNode::initialize(Renderer* renderer, ResourceManager* res_mgr)
{
	m_library = renderer->shader_cache().load_library("shader/forward/forward_vs.glsl", "shader/forward/forward_fs.glsl");

    //m_color_rtv = RenderTargetView(0, 0, 0, m_color_rt.get());
    //m_velocity_rtv = RenderTargetView(0, 0, 0, m_velocity_rt.get());
    //m_depth_rtv = RenderTargetView(0, 0, 0, m_depth_rt.get());

    return true;
}

// -----------------------------------------------------------------------------------------------------------------------------------

void ForwardNode::execute(Renderer* renderer, Scene* scene, View* view)
{
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glViewport(0, 0, m_graph->window_width(), m_graph->window_height());

    glEnable(GL_DEPTH_TEST);

    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);

	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	render_scene(renderer, scene, view, m_library.get());
}

// -----------------------------------------------------------------------------------------------------------------------------------

void ForwardNode::shutdown()
{
}

// -----------------------------------------------------------------------------------------------------------------------------------

std::string ForwardNode::name()
{
    return "Forward";
}

// -----------------------------------------------------------------------------------------------------------------------------------
} // namespace nimble
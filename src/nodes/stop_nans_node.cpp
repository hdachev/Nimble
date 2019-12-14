#include "stop_nans_node.h"
#include "../render_graph.h"
#include "../resource_manager.h"
#include "../renderer.h"
#include "../logger.h"

namespace nimble
{
DEFINE_RENDER_NODE_FACTORY(StopNaNsNode)

// -----------------------------------------------------------------------------------------------------------------------------------

StopNaNsNode::StopNaNsNode(RenderGraph* graph) :
    RenderNode(graph)
{
}

// -----------------------------------------------------------------------------------------------------------------------------------

StopNaNsNode::~StopNaNsNode()
{
}

// -----------------------------------------------------------------------------------------------------------------------------------

void StopNaNsNode::declare_connections()
{
    // Declare the inputs to this render node
    register_input_render_target("Color");

	m_output_rt = register_scaled_output_render_target("Color", 1.0f, 1.0f, GL_TEXTURE_2D, GL_RGBA16F, GL_RGBA, GL_HALF_FLOAT);
}

// -----------------------------------------------------------------------------------------------------------------------------------

bool StopNaNsNode::initialize(Renderer* renderer, ResourceManager* res_mgr)
{
    m_texture = find_input_render_target("Color");

	m_output_rtv = RenderTargetView(0, 0, 0, m_output_rt->texture);

    m_vs = res_mgr->load_shader("shader/post_process/fullscreen_triangle_vs.glsl", GL_VERTEX_SHADER);
    m_fs = res_mgr->load_shader("shader/post_process/stop_nans_fs.glsl", GL_FRAGMENT_SHADER);

    if (m_vs && m_fs)
    {
        m_program = renderer->create_program(m_vs, m_fs);

        if (m_program)
            return true;
        else
        {
            NIMBLE_LOG_ERROR("Failed to create Program!");
            return false;
        }
    }
    else
    {
        NIMBLE_LOG_ERROR("Failed to load Shaders!");
        return false;
    }
}

// -----------------------------------------------------------------------------------------------------------------------------------

void StopNaNsNode::execute(double delta, Renderer* renderer, Scene* scene, View* view)
{
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);

    m_program->use();

    renderer->bind_render_targets(1, &m_output_rtv, nullptr);

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
    glViewport(0, 0, m_graph->window_width(), m_graph->window_height());

    if (m_program->set_uniform("s_Color", 0) && m_texture)
        m_texture->texture->bind(0);

    render_fullscreen_triangle(renderer, view);
}

// -----------------------------------------------------------------------------------------------------------------------------------

void StopNaNsNode::shutdown()
{
}

// -----------------------------------------------------------------------------------------------------------------------------------

std::string StopNaNsNode::name()
{
    return "Stop NaNs";
}

// -----------------------------------------------------------------------------------------------------------------------------------
} // namespace nimble
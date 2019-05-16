#include "taa_node.h"
#include "../render_graph.h"
#include "../resource_manager.h"
#include "../renderer.h"
#include "../profiler.h"

namespace nimble
{
DEFINE_RENDER_NODE_FACTORY(TAANode)

// -----------------------------------------------------------------------------------------------------------------------------------

TAANode::TAANode(RenderGraph* graph) :
    RenderNode(graph)
{
}

// -----------------------------------------------------------------------------------------------------------------------------------

TAANode::~TAANode()
{
}

// -----------------------------------------------------------------------------------------------------------------------------------

void TAANode::declare_connections()
{
    register_input_render_target("Color");
    register_input_render_target("Velocity");

    m_taa_rt = register_scaled_output_render_target("TAA", 1.0f, 1.0f, GL_TEXTURE_2D, GL_RGB16F, GL_RGB, GL_HALF_FLOAT);
    m_prev_rt = register_scaled_intermediate_render_target("Previous", 1.0f, 1.0f, GL_TEXTURE_2D, GL_RGB16F, GL_RGB, GL_HALF_FLOAT);
}

// -----------------------------------------------------------------------------------------------------------------------------------

bool TAANode::initialize(Renderer* renderer, ResourceManager* res_mgr)
{
    register_bool_parameter("Enabled", m_enabled);

    m_color_rt    = find_input_render_target("Color");
    m_velocity_rt = find_input_render_target("Velocity");

    m_taa_rtv = RenderTargetView(0, 0, 0, m_taa_rt->texture);

    m_fullscreen_triangle_vs = res_mgr->load_shader("shader/post_process/fullscreen_triangle_vs.glsl", GL_VERTEX_SHADER);
    m_taa_fs                 = res_mgr->load_shader("shader/post_process/taa/taa_fs.glsl", GL_FRAGMENT_SHADER);

    if (m_fullscreen_triangle_vs && m_taa_fs)
    {
        m_taa_program = renderer->create_program(m_fullscreen_triangle_vs, m_taa_fs);
        return true;
    }
    else
        return false;
}

// -----------------------------------------------------------------------------------------------------------------------------------

void TAANode::execute(double delta, Renderer* renderer, Scene* scene, View* view)
{
    if (m_enabled)
    {
        glDisable(GL_DEPTH_TEST);
        glDisable(GL_CULL_FACE);

        m_taa_program->use();

        renderer->bind_render_targets(1, &m_taa_rtv, nullptr);

        glClear(GL_COLOR_BUFFER_BIT);
        glViewport(0, 0, m_graph->window_width(), m_graph->window_height());

        if (m_taa_program->set_uniform("s_Color", 0) && m_color_rt)
            m_color_rt->texture->bind(0);

		if (m_taa_program->set_uniform("s_Prev", 1) && m_prev_rt)
            m_prev_rt->texture->bind(1);

        if (m_taa_program->set_uniform("s_Velocity", 2) && m_velocity_rt)
            m_velocity_rt->texture->bind(2);

        render_fullscreen_triangle(renderer, view);

		// Copy Current Target to Previous
        blit_render_target(renderer, m_taa_rt, m_prev_rt);
    }
    else
        blit_render_target(renderer, m_color_rt, m_taa_rt);
}

// -----------------------------------------------------------------------------------------------------------------------------------

void TAANode::shutdown()
{
}

// -----------------------------------------------------------------------------------------------------------------------------------

std::string TAANode::name()
{
    return "TAA";
}

// -----------------------------------------------------------------------------------------------------------------------------------
} // namespace nimble
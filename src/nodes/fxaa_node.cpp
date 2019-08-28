#include "fxaa_node.h"
#include "../render_graph.h"
#include "../resource_manager.h"
#include "../renderer.h"
#include "../profiler.h"

namespace nimble
{
DEFINE_RENDER_NODE_FACTORY(FXAANode)

// -----------------------------------------------------------------------------------------------------------------------------------

FXAANode::FXAANode(RenderGraph* graph) :
    RenderNode(graph)
{
}

// -----------------------------------------------------------------------------------------------------------------------------------

FXAANode::~FXAANode()
{
}

// -----------------------------------------------------------------------------------------------------------------------------------

void FXAANode::declare_connections()
{
    register_input_render_target("Color");

    m_fxaa_rt = register_scaled_output_render_target("FXAA", 1.0f, 1.0f, GL_TEXTURE_2D, GL_RGB16F, GL_RGB, GL_HALF_FLOAT);
}

// -----------------------------------------------------------------------------------------------------------------------------------

bool FXAANode::initialize_private(Renderer* renderer, ResourceManager* res_mgr)
{
    register_bool_parameter("Enabled", m_enabled);
    register_float_parameter("Quality Edge Threshold", m_quality_edge_threshold);
    register_float_parameter("Quality Edge Threshold Min", m_quality_edge_threshold_min);

    m_color_rt = find_input_render_target("Color");

    m_fxaa_rtv = RenderTargetView(0, 0, 0, m_fxaa_rt->texture);

    m_fullscreen_triangle_vs = res_mgr->load_shader("shader/post_process/fullscreen_triangle_vs.glsl", GL_VERTEX_SHADER);
    m_fxaa_fs                = res_mgr->load_shader("shader/post_process/fxaa/fxaa_fs.glsl", GL_FRAGMENT_SHADER, { "FXAA_PC 1", "FXAA_GLSL_130 1", "FXAA_QUALITY__PRESET 39", "FXAA_GREEN_AS_LUMA 1" });

    if (m_fullscreen_triangle_vs && m_fxaa_fs)
    {
        m_fxaa_program = renderer->create_program(m_fullscreen_triangle_vs, m_fxaa_fs);
        return true;
    }
    else
        return false;
}

// -----------------------------------------------------------------------------------------------------------------------------------

void FXAANode::execute(double delta, Renderer* renderer, Scene* scene, View* view)
{
    if (m_enabled)
    {
        glDisable(GL_DEPTH_TEST);
        glDisable(GL_CULL_FACE);

        m_fxaa_program->use();

        renderer->bind_render_targets(1, &m_fxaa_rtv, nullptr);

        glClear(GL_COLOR_BUFFER_BIT);
        glViewport(0, 0, m_graph->window_width(), m_graph->window_height());

        if (m_fxaa_program->set_uniform("s_Texture", 0) && m_color_rt)
            m_color_rt->texture->bind(0);

        m_fxaa_program->set_uniform("u_QualityEdgeThreshold", m_quality_edge_threshold);
        m_fxaa_program->set_uniform("u_QualityEdgeThresholdMin", m_quality_edge_threshold_min);
        m_fxaa_program->set_uniform("u_QualityRcpFrame", glm::vec2(1.0f / m_graph->window_width(), 1.0f / m_graph->window_height()));

        render_fullscreen_triangle(renderer, view);
    }
    else
        blit_render_target(renderer, m_color_rt, m_fxaa_rt);
}

// -----------------------------------------------------------------------------------------------------------------------------------

void FXAANode::shutdown()
{
}

// -----------------------------------------------------------------------------------------------------------------------------------

std::string FXAANode::name()
{
    return "FXAA";
}

// -----------------------------------------------------------------------------------------------------------------------------------
} // namespace nimble
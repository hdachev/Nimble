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

    m_taa_rt  = register_scaled_output_render_target("TAA", 1.0f, 1.0f, GL_TEXTURE_2D, GL_RGB16F, GL_RGB, GL_HALF_FLOAT);
    m_prev_rt = register_scaled_intermediate_render_target("Previous", 1.0f, 1.0f, GL_TEXTURE_2D, GL_RGB16F, GL_RGB, GL_HALF_FLOAT);
}

// -----------------------------------------------------------------------------------------------------------------------------------

bool TAANode::initialize(Renderer* renderer, ResourceManager* res_mgr)
{
    register_bool_parameter("Enabled", m_enabled);
    register_enum_parameter("Neighborhood", &m_neighborhood, { { MIN_MAX_3X3, "Min Max 3x3" }, { MIN_MAX_3X3_ROUNDED, "Min Max 3x3 Rounded" }, { MIN_MAX_4_TAP_VARYING, "Min Max 4 Tap Varying" } });
    register_bool_parameter("Unjitter Color Samples", m_unjitter_color_samples);
    register_bool_parameter("Unjitter Neighborhood", m_unjitter_neighborhood);
    register_bool_parameter("Unjitter Reprojection", m_unjitter_reprojection);
    register_bool_parameter("Use YCoCg", m_use_ycocg);
    register_bool_parameter("Use Clipping", m_use_clipping);
    register_bool_parameter("Use Dilation", m_use_dilation);
    register_bool_parameter("Use Motion Blur", m_use_motion_blur);
    register_bool_parameter("Use Optimizations", m_use_optimizations);
    register_bool_parameter("Use Dilation", m_use_dilation);
    register_float_parameter("Feedback Min", m_feedback_min, 0.0f, 1.0f);
    register_float_parameter("Feedback Max", m_feedback_max, 0.0, 1.0f);

    m_color_rt    = find_input_render_target("Color");
    m_velocity_rt = find_input_render_target("Velocity");

    m_taa_rtv = RenderTargetView(0, 0, 0, m_taa_rt->texture);

    m_fullscreen_triangle_vs = res_mgr->load_shader("shader/post_process/fullscreen_triangle_vs.glsl", GL_VERTEX_SHADER);

	std::vector<std::string> defines;

	if (m_neighborhood == MIN_MAX_3X3)
		defines.push_back("MINMAX_3X3");
	if (m_neighborhood == MIN_MAX_3X3_ROUNDED)
	    defines.push_back("MINMAX_3X3_ROUNDED");
	if (m_neighborhood == MIN_MAX_4_TAP_VARYING)
	    defines.push_back("MINMAX_4TAP_VARYING");
	if (m_unjitter_color_samples)
		defines.push_back("UNJITTER_COLORSAMPLES");
	if (m_unjitter_neighborhood)
	    defines.push_back("UNJITTER_NEIGHBORHOOD");
	if (m_unjitter_reprojection)
	    defines.push_back("UNJITTER_REPROJECTION");
	if (m_use_ycocg)
	    defines.push_back("USE_YCOCG");
	if (m_use_clipping)
	    defines.push_back("USE_CLIPPING");
	if (m_use_dilation)
	    defines.push_back("USE_DILATION");
	if (m_use_motion_blur)
	    defines.push_back("USE_MOTION_BLUR");
	if (m_use_optimizations)
	    defines.push_back("USE_OPTIMIZATIONS");

    m_taa_fs = res_mgr->load_shader("shader/post_process/taa/taa_fs.glsl", GL_FRAGMENT_SHADER, defines);

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

        m_taa_program->set_uniform("u_TexelSize", glm::vec4(1.0f / m_graph->window_width(), 1.0f / m_graph->window_height(), m_graph->window_width(), m_graph->window_height()));
        m_taa_program->set_uniform("u_FeedbackMin", m_feedback_min);
        m_taa_program->set_uniform("u_FeedbackMax", m_feedback_max);
        m_taa_program->set_uniform("u_MotionScale", m_motion_blur_strength);

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
#include "motion_blur_node.h"
#include "../render_graph.h"
#include "../resource_manager.h"
#include "../renderer.h"
#include "../logger.h"

namespace nimble
{
DEFINE_RENDER_NODE_FACTORY(MotionBlurNode)

// -----------------------------------------------------------------------------------------------------------------------------------

MotionBlurNode::MotionBlurNode(RenderGraph* graph) :
    RenderNode(graph)
{
}

// -----------------------------------------------------------------------------------------------------------------------------------

MotionBlurNode::~MotionBlurNode()
{
}

// -----------------------------------------------------------------------------------------------------------------------------------

void MotionBlurNode::declare_connections()
{
    register_input_render_target("Color");
    register_input_render_target("Velocity");

    m_motion_blur_rt = register_scaled_output_render_target("MotionBlur", 1.0f, 1.0f, GL_TEXTURE_2D, GL_RGB16F, GL_RGB, GL_HALF_FLOAT, 1, 1);
}

// -----------------------------------------------------------------------------------------------------------------------------------

bool MotionBlurNode::initialize(Renderer* renderer, ResourceManager* res_mgr)
{
    register_bool_parameter("Enabled", m_enabled);
    register_int_parameter("Num Samples", m_num_samples, 0, 32);

    m_color_rt    = find_input_render_target("Color");
    m_velocity_rt = find_input_render_target("Velocity");

    m_motion_blur_rtv = RenderTargetView(0, 0, 0, m_motion_blur_rt->texture);

    m_vs = res_mgr->load_shader("shader/post_process/fullscreen_triangle_vs.glsl", GL_VERTEX_SHADER);
    m_fs = res_mgr->load_shader("shader/post_process/motion_blur/motion_blur_fs.glsl", GL_FRAGMENT_SHADER);

    if (m_vs && m_fs)
    {
        m_program = renderer->create_program(m_vs, m_fs);
        return true;
    }
    else
        return false;
}

// -----------------------------------------------------------------------------------------------------------------------------------

void MotionBlurNode::execute(double delta, Renderer* renderer, Scene* scene, View* view)
{
    if (m_enabled)
    {
        glDisable(GL_DEPTH_TEST);
        glDisable(GL_CULL_FACE);

        m_program->use();

        renderer->bind_render_targets(1, &m_motion_blur_rtv, nullptr);
        glViewport(0, 0, m_graph->window_width(), m_graph->window_height());

        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        if (m_program->set_uniform("s_Color", 0))
            m_color_rt->texture->bind(0);

        if (m_program->set_uniform("s_Velocity", 1))
            m_velocity_rt->texture->bind(1);

        int current_fps = int((1.0f / (static_cast<float>(delta)) * 1000.0f));
        int target_fps  = 60;

        m_program->set_uniform("u_Scale", static_cast<float>(current_fps) / static_cast<float>(target_fps));
        m_program->set_uniform("u_NumSamples", m_num_samples);

        render_fullscreen_triangle(renderer, nullptr);
    }
    else
        blit_render_target(renderer, m_color_rt, m_motion_blur_rt);
}

// -----------------------------------------------------------------------------------------------------------------------------------

void MotionBlurNode::shutdown()
{
}

// -----------------------------------------------------------------------------------------------------------------------------------

std::string MotionBlurNode::name()
{
    return "Motion Blur";
}

// -----------------------------------------------------------------------------------------------------------------------------------
} // namespace nimble
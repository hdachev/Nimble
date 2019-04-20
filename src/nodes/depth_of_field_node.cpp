#include "depth_of_field_node.h"
#include "../render_graph.h"
#include "../resource_manager.h"
#include "../renderer.h"
#include "../profiler.h"

namespace nimble
{
DEFINE_RENDER_NODE_FACTORY(DepthOfFieldNode)

// -----------------------------------------------------------------------------------------------------------------------------------

DepthOfFieldNode::DepthOfFieldNode(RenderGraph* graph) :
    RenderNode(graph)
{
}

// -----------------------------------------------------------------------------------------------------------------------------------

DepthOfFieldNode::~DepthOfFieldNode()
{
}

// -----------------------------------------------------------------------------------------------------------------------------------

void DepthOfFieldNode::declare_connections()
{
    register_input_render_target("Color");
    register_input_render_target("Velocity");

    m_taa_rt = register_scaled_output_render_target("TAA", 1.0f, 1.0f, GL_TEXTURE_2D, GL_RGB16F, GL_RGB, GL_HALF_FLOAT);
}

// -----------------------------------------------------------------------------------------------------------------------------------

bool DepthOfFieldNode::initialize(Renderer* renderer, ResourceManager* res_mgr)
{
    register_bool_parameter("Enabled", m_enabled);

    m_color_rt    = find_input_render_target("Color");
    m_depth_rt = find_input_render_target("Velocity");

    m_taa_rtv = RenderTargetView(0, 0, 0, m_taa_rt->texture);

    m_triangle_vs         = res_mgr->load_shader("shader/post_process/fullscreen_triangle_vs.glsl", GL_VERTEX_SHADER);
    m_coc_fs              = res_mgr->load_shader("shader/post_process/depth_of_field/coc_fs.glsl", GL_FRAGMENT_SHADER);
    m_downsample_fs       = res_mgr->load_shader("shader/post_process/depth_of_field/downsample_fs.glsl", GL_FRAGMENT_SHADER);
    m_near_coc_max_x4_fs  = res_mgr->load_shader("shader/post_process/filters_fs.glsl", GL_FRAGMENT_SHADER, { "HORIZONTAL", "MAX13", "CHANNELS_COUNT_1" });
    m_near_coc_max4_fs    = res_mgr->load_shader("shader/post_process/filters_fs.glsl", GL_FRAGMENT_SHADER, { "VERTICAL", "MAX13", "CHANNELS_COUNT_1" });
    m_near_coc_blur_x4_fs = res_mgr->load_shader("shader/post_process/filters_fs.glsl", GL_FRAGMENT_SHADER, { "HORIZONTAL", "BLUR13", "CHANNELS_COUNT_1" });
    m_near_coc_blur4_fs   = res_mgr->load_shader("shader/post_process/filters_fs.glsl", GL_FRAGMENT_SHADER, { "VERTICAL", "BLUR13", "CHANNELS_COUNT_1" });
    m_computation_fs      = res_mgr->load_shader("shader/post_process/depth_of_field/computation_fs.glsl", GL_FRAGMENT_SHADER);
    m_fill_fs             = res_mgr->load_shader("shader/post_process/depth_of_field/fill_fs.glsl", GL_FRAGMENT_SHADER);
    m_composite_fs        = res_mgr->load_shader("shader/post_process/depth_of_field/composite_fs.glsl", GL_FRAGMENT_SHADER);

    if (m_triangle_vs)
    {
        if (m_coc_fs)
            m_coc_program = renderer->create_program(m_triangle_vs, m_coc_fs);
        else
            return false;

        if (m_downsample_fs)
            m_downsample_program = renderer->create_program(m_triangle_vs, m_downsample_fs);
        else
            return false;

        if (m_near_coc_max_x4_fs)
            m_near_coc_max_x4_program = renderer->create_program(m_triangle_vs, m_near_coc_max_x4_fs);
        else
            return false;

        if (m_near_coc_max4_fs)
            m_near_coc_max4_program = renderer->create_program(m_triangle_vs, m_near_coc_max4_fs);
        else
            return false;

        if (m_near_coc_blur_x4_fs)
            m_near_coc_blur_x4_program = renderer->create_program(m_triangle_vs, m_near_coc_blur_x4_fs);
        else
            return false;

        if (m_near_coc_blur4_fs)
            m_near_coc_blur4_program = renderer->create_program(m_triangle_vs, m_near_coc_blur4_fs);
        else
            return false;

        if (m_computation_fs)
            m_computation_program = renderer->create_program(m_triangle_vs, m_computation_fs);
        else
            return false;

        if (m_fill_fs)
            m_fill_program = renderer->create_program(m_triangle_vs, m_fill_fs);
        else
            return false;

        if (m_composite_fs)
            m_composite_program = renderer->create_program(m_triangle_vs, m_composite_fs);
        else
            return false;

        return true;
    }
    else
        return false;
}

// -----------------------------------------------------------------------------------------------------------------------------------

void DepthOfFieldNode::execute(double delta, Renderer* renderer, Scene* scene, View* view)
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

        if (m_taa_program->set_uniform("s_Velocity", 1) && m_velocity_rt)
            m_velocity_rt->texture->bind(1);

        render_fullscreen_triangle(renderer, view);
    }
    else
        blit_render_target(renderer, m_color_rt, m_taa_rt);
}

// -----------------------------------------------------------------------------------------------------------------------------------

void DepthOfFieldNode::shutdown()
{
}

// -----------------------------------------------------------------------------------------------------------------------------------

std::string DepthOfFieldNode::name()
{
    return "Depth Of Field";
}

// -----------------------------------------------------------------------------------------------------------------------------------
} // namespace nimble
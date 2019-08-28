#include "adaptive_exposure_node.h"
#include "../render_graph.h"
#include "../resource_manager.h"
#include "../renderer.h"
#include "../logger.h"

namespace nimble
{
DEFINE_RENDER_NODE_FACTORY(AdaptiveExposureNode)

#define LUM_THREADS 8
#define AVG_LUM_THREADS 8
#define LUM_SIZE 1024

// -----------------------------------------------------------------------------------------------------------------------------------

AdaptiveExposureNode::AdaptiveExposureNode(RenderGraph* graph) :
    RenderNode(graph)
{
}

// -----------------------------------------------------------------------------------------------------------------------------------

AdaptiveExposureNode::~AdaptiveExposureNode()
{
}

// -----------------------------------------------------------------------------------------------------------------------------------

void AdaptiveExposureNode::declare_connections()
{
    register_input_render_target("Color");

    m_luma_rt         = register_intermediate_render_target("InitialLuma", LUM_SIZE, LUM_SIZE, GL_TEXTURE_2D, GL_R32F, GL_RED, GL_FLOAT);
    m_compute_luma_rt = register_intermediate_render_target("ComputeLuma", LUM_SIZE / 2, LUM_SIZE / 2, GL_TEXTURE_2D, GL_R32F, GL_RED, GL_FLOAT);
    m_avg_luma_rt     = register_output_render_target("Luminance", 1, 1, GL_TEXTURE_2D, GL_R32F, GL_RED, GL_FLOAT);
}

// -----------------------------------------------------------------------------------------------------------------------------------

bool AdaptiveExposureNode::initialize_private(Renderer* renderer, ResourceManager* res_mgr)
{
    register_float_parameter("Middle Grey", m_middle_grey);
    register_float_parameter("Rate", m_rate);

    m_color_rt = find_input_render_target("Color");

    m_luma_rtv = RenderTargetView(0, 0, 0, m_luma_rt->texture);

    m_vs = res_mgr->load_shader("shader/post_process/fullscreen_triangle_vs.glsl", GL_VERTEX_SHADER);

    if (m_vs)
    {
        m_lum_fs = res_mgr->load_shader("shader/post_process/adaptive_exposure/initial_luminance_fs.glsl", GL_FRAGMENT_SHADER);

        if (m_lum_fs)
            m_lum_program = renderer->create_program(m_vs, m_lum_fs);
        else
            return false;

        m_compute_lum_fs = res_mgr->load_shader("shader/post_process/adaptive_exposure/compute_luminance_cs.glsl", GL_COMPUTE_SHADER);

        if (m_compute_lum_fs)
            m_compute_lum_program = renderer->create_program({ m_compute_lum_fs });
        else
            return false;

        m_average_lum_fs = res_mgr->load_shader("shader/post_process/adaptive_exposure/average_luminance_cs.glsl", GL_COMPUTE_SHADER);

        if (m_average_lum_fs)
            m_average_lum_program = renderer->create_program({ m_average_lum_fs });
        else
            return false;

        return true;
    }
    else
        return false;
}

// -----------------------------------------------------------------------------------------------------------------------------------

void AdaptiveExposureNode::execute(double delta, Renderer* renderer, Scene* scene, View* view)
{
    initial_luminance(renderer, scene, view);
    compute_luminance(renderer, scene, view);
    average_luminance(delta, renderer, scene, view);
}

// -----------------------------------------------------------------------------------------------------------------------------------

void AdaptiveExposureNode::shutdown()
{
}

// -----------------------------------------------------------------------------------------------------------------------------------

std::string AdaptiveExposureNode::name()
{
    return "Adaptive Exposure";
}

// -----------------------------------------------------------------------------------------------------------------------------------

void AdaptiveExposureNode::initial_luminance(Renderer* renderer, Scene* scene, View* view)
{
    glPushDebugGroup(GL_DEBUG_SOURCE_APPLICATION, 0, -1, "Initial Luminance");

    glDisable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);

    m_lum_program->use();

    if (m_lum_program->set_uniform("s_Texture", 0))
        m_color_rt->texture->bind(0);

    renderer->bind_render_targets(1, &m_luma_rtv, nullptr);
    glViewport(0, 0, LUM_SIZE, LUM_SIZE);
    glClear(GL_COLOR_BUFFER_BIT);

    render_fullscreen_triangle(renderer, nullptr);

    glPopDebugGroup();
}

// -----------------------------------------------------------------------------------------------------------------------------------

void AdaptiveExposureNode::compute_luminance(Renderer* renderer, Scene* scene, View* view)
{
    glPushDebugGroup(GL_DEBUG_SOURCE_APPLICATION, 0, -1, "Compute Luminance");

    m_compute_lum_program->use();

    m_compute_luma_rt->texture->bind_image(0, 0, 0, GL_READ_WRITE, GL_R32F);

    if (m_compute_lum_program->set_uniform("u_InitialLuma", 0))
        m_luma_rt->texture->bind(0);

    glDispatchCompute(512 / LUM_THREADS, 512 / LUM_THREADS, 1);

    glPopDebugGroup();
}

// -----------------------------------------------------------------------------------------------------------------------------------

void AdaptiveExposureNode::average_luminance(double delta, Renderer* renderer, Scene* scene, View* view)
{
    glPushDebugGroup(GL_DEBUG_SOURCE_APPLICATION, 0, -1, "Average Luminance");

    m_average_lum_program->use();

    m_compute_luma_rt->texture->bind_image(0, 0, 0, GL_READ_ONLY, GL_R32F);

    m_avg_luma_rt->texture->bind_image(1, 0, 0, GL_READ_WRITE, GL_R32F);

    m_average_lum_program->set_uniform("u_MiddleGrey", m_middle_grey);
    m_average_lum_program->set_uniform("u_Rate", m_rate);
    m_average_lum_program->set_uniform("u_Delta", static_cast<float>(delta) / 1000.0f);
    m_average_lum_program->set_uniform("u_First", m_first ? 1 : 0);

    glDispatchCompute(1, 1, 1);

    glPopDebugGroup();

    if (m_first)
        m_first = false;
}

// -----------------------------------------------------------------------------------------------------------------------------------
} // namespace nimble
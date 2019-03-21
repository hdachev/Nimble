#include "adaptive_exposure_node.h"
#include "../render_graph.h"
#include "../resource_manager.h"
#include "../renderer.h"
#include "../logger.h"

namespace nimble
{
DEFINE_RENDER_NODE_FACTORY(AdaptiveExposureNode)

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

    m_luminance_rt            = register_intermediate_render_target("InitialLuminance", 1024, 1024, GL_TEXTURE_2D, GL_R32F, GL_RED, GL_FLOAT);
    m_adapted_luminance_rt[0] = register_intermediate_render_target("AdaptedLuminance1", 1024, 1024, GL_TEXTURE_2D, GL_R32F, GL_RED, GL_FLOAT, 1, 1, -1);
    m_adapted_luminance_rt[1] = register_intermediate_render_target("AdaptedLuminance2", 1024, 1024, GL_TEXTURE_2D, GL_R32F, GL_RED, GL_FLOAT, 1, 1, -1);
    m_final_luminance_rt      = register_output_render_target("Luminance", 1, 1, GL_TEXTURE_2D, GL_R32F, GL_RED, GL_FLOAT);
}

// -----------------------------------------------------------------------------------------------------------------------------------

bool AdaptiveExposureNode::initialize(Renderer* renderer, ResourceManager* res_mgr)
{
    register_float_parameter("Tau", m_tau);

    m_color_rt = find_input_render_target("Color");

    m_luminance_rtv            = RenderTargetView(0, 0, 0, m_luminance_rt->texture);
    m_adapted_luminance_rtv[0] = RenderTargetView(0, 0, 0, m_adapted_luminance_rt[0]->texture);
    m_adapted_luminance_rtv[1] = RenderTargetView(0, 0, 0, m_adapted_luminance_rt[1]->texture);
    m_final_luminance_rtv      = RenderTargetView(0, 0, 0, m_final_luminance_rt->texture);

    m_vs = res_mgr->load_shader("shader/post_process/fullscreen_triangle_vs.glsl", GL_VERTEX_SHADER);

    if (m_vs)
    {
        m_lum_fs = res_mgr->load_shader("shader/post_process/adaptive_exposure/initial_luminance_fs.glsl", GL_FRAGMENT_SHADER);

        if (m_lum_fs)
            m_lum_program = renderer->create_program(m_vs, m_lum_fs);
        else
            return false;

        m_adapted_lum_fs = res_mgr->load_shader("shader/post_process/adaptive_exposure/adapted_luminance_fs.glsl", GL_FRAGMENT_SHADER);

        if (m_adapted_lum_fs)
            m_adapted_lum_program = renderer->create_program(m_vs, m_adapted_lum_fs);
        else
            return false;

        m_copy_lum_fs = res_mgr->load_shader("shader/post_process/adaptive_exposure/copy_luminance_fs.glsl", GL_FRAGMENT_SHADER);

        if (m_copy_lum_fs)
            m_copy_lum_program = renderer->create_program(m_vs, m_copy_lum_fs);
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
    adapted_luminance(delta, renderer, scene, view);
    copy_luminance(renderer, scene, view);

    m_current_rt = !m_current_rt;
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

    renderer->bind_render_targets(1, &m_luminance_rtv, nullptr);
    glViewport(0, 0, 1024, 1024);
    glClear(GL_COLOR_BUFFER_BIT);

    render_fullscreen_triangle(renderer, nullptr);

    glPopDebugGroup();
}

// -----------------------------------------------------------------------------------------------------------------------------------

void AdaptiveExposureNode::adapted_luminance(double delta, Renderer* renderer, Scene* scene, View* view)
{
    glPushDebugGroup(GL_DEBUG_SOURCE_APPLICATION, 0, -1, "Adapted Luminance");

    glDisable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);

    if (!m_initialized)
    {
        glClearColor(0.0f, 0.0f, 0.0f, 0.0f);

        renderer->bind_render_targets(1, &m_adapted_luminance_rtv[0], nullptr);
        glViewport(0, 0, 1024, 1024);
        glClear(GL_COLOR_BUFFER_BIT);

        renderer->bind_render_targets(1, &m_adapted_luminance_rtv[1], nullptr);
        glViewport(0, 0, 1024, 1024);
        glClear(GL_COLOR_BUFFER_BIT);

        m_initialized = true;
    }

    m_adapted_lum_program->use();

    if (m_adapted_lum_program->set_uniform("s_PreviousLuminance", 0))
        m_adapted_luminance_rt[!m_current_rt]->texture->bind(0);

    if (m_adapted_lum_program->set_uniform("s_CurrentLuminance", 1))
        m_luminance_rt->texture->bind(1);

    m_adapted_lum_program->set_uniform("u_Tau", m_tau);
    m_adapted_lum_program->set_uniform("u_Delta", static_cast<float>(delta) / 1000.0f);

    renderer->bind_render_targets(1, &m_adapted_luminance_rtv[m_current_rt], nullptr);
    glViewport(0, 0, 1024, 1024);
    glClear(GL_COLOR_BUFFER_BIT);

    render_fullscreen_triangle(renderer, nullptr);

    m_adapted_luminance_rt[m_current_rt]->texture->generate_mipmaps();

    glPopDebugGroup();
}

// -----------------------------------------------------------------------------------------------------------------------------------

void AdaptiveExposureNode::copy_luminance(Renderer* renderer, Scene* scene, View* view)
{
    glPushDebugGroup(GL_DEBUG_SOURCE_APPLICATION, 0, -1, "Copy Luminance");

    glDisable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);

    m_copy_lum_program->use();

    if (m_copy_lum_program->set_uniform("s_Texture", 0))
        m_adapted_luminance_rt[m_current_rt]->texture->bind(0);

    int mip = m_adapted_luminance_rt[m_current_rt]->texture->mip_levels() - 1;
    m_copy_lum_program->set_uniform("u_MipLevel", mip);

    renderer->bind_render_targets(1, &m_final_luminance_rtv, nullptr);
    glViewport(0, 0, 1, 1);
    glClear(GL_COLOR_BUFFER_BIT);

    render_fullscreen_triangle(renderer, nullptr);

    glPopDebugGroup();
}

// -----------------------------------------------------------------------------------------------------------------------------------
} // namespace nimble
#include "reflection_node.h"
#include "../render_graph.h"
#include "../resource_manager.h"
#include "../renderer.h"
#include "../profiler.h"

namespace nimble
{
DEFINE_RENDER_NODE_FACTORY(ReflectionNode)

// -----------------------------------------------------------------------------------------------------------------------------------

ReflectionNode::ReflectionNode(RenderGraph* graph) :
    RenderNode(graph)
{
}

// -----------------------------------------------------------------------------------------------------------------------------------

ReflectionNode::~ReflectionNode()
{
}

// -----------------------------------------------------------------------------------------------------------------------------------

void ReflectionNode::declare_connections()
{
    register_input_render_target("Color");
    register_input_render_target("SSR");

    m_reflection_rt = register_scaled_output_render_target("Reflection", 1.0f, 1.0f, GL_TEXTURE_2D, GL_RGB16F, GL_RGB, GL_HALF_FLOAT);
}

// -----------------------------------------------------------------------------------------------------------------------------------

bool ReflectionNode::initialize(Renderer* renderer, ResourceManager* res_mgr)
{
    register_bool_parameter("Screen Space Reflections", m_ssr);

    m_color_rt = find_input_render_target("Color");
    m_ssr_rt   = find_input_render_target("SSR");

    m_reflection_rtv = RenderTargetView(0, 0, 0, m_reflection_rt->texture);

    m_fullscreen_triangle_vs = res_mgr->load_shader("shader/post_process/fullscreen_triangle_vs.glsl", GL_VERTEX_SHADER);
    m_reflection_fs          = res_mgr->load_shader("shader/post_process/ssr/reflection_fs.glsl", GL_FRAGMENT_SHADER);

    if (m_fullscreen_triangle_vs && m_reflection_fs)
    {
        m_reflection_program = renderer->create_program(m_fullscreen_triangle_vs, m_reflection_fs);
        return true;
    }
    else
        return false;
}

// -----------------------------------------------------------------------------------------------------------------------------------

void ReflectionNode::execute(double delta, Renderer* renderer, Scene* scene, View* view)
{
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);

    m_reflection_program->use();

    renderer->bind_render_targets(1, &m_reflection_rtv, nullptr);

    glClear(GL_COLOR_BUFFER_BIT);
    glViewport(0, 0, m_graph->window_width(), m_graph->window_height());

    if (m_ssr_rt)
    {
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

        if (m_reflection_program->set_uniform("s_SSR", 0))
            m_ssr_rt->texture->bind(0);

        if (m_reflection_program->set_uniform("s_Color", 1) && m_color_rt)
            m_color_rt->texture->bind(1);

        m_reflection_program->set_uniform("u_SSR", m_ssr_rt ? (float)m_ssr : 0.0f);

        render_fullscreen_triangle(renderer, view);
    }
    else
        blit_render_target(renderer, m_color_rt, m_reflection_rt);
}

// -----------------------------------------------------------------------------------------------------------------------------------

void ReflectionNode::shutdown()
{
}

// -----------------------------------------------------------------------------------------------------------------------------------

std::string ReflectionNode::name()
{
    return "Reflections";
}

// -----------------------------------------------------------------------------------------------------------------------------------
} // namespace nimble
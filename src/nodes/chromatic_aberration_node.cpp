#include "chromatic_aberration_node.h"
#include "../resource_manager.h"
#include "../renderer.h"
#include "../logger.h"

namespace nimble
{
DEFINE_RENDER_NODE_FACTORY(ChromaticAberrationNode)

// -----------------------------------------------------------------------------------------------------------------------------------

ChromaticAberrationNode::ChromaticAberrationNode(RenderGraph* graph) :
    RenderNode(graph)
{
}

// -----------------------------------------------------------------------------------------------------------------------------------

ChromaticAberrationNode::~ChromaticAberrationNode()
{
}

// -----------------------------------------------------------------------------------------------------------------------------------

void ChromaticAberrationNode::declare_connections()
{
    register_input_render_target("Color");

    m_output_rt = register_scaled_output_render_target("ChromaticAberration", 1.0f, 1.0f, GL_TEXTURE_2D, GL_RGB8, GL_RGB, GL_UNSIGNED_BYTE);
}

// -----------------------------------------------------------------------------------------------------------------------------------

bool ChromaticAberrationNode::initialize(Renderer* renderer, ResourceManager* res_mgr)
{
    register_float_parameter("Strength", m_strength, 0.0f, 1.0f);

    m_texture = find_input_render_target("Color");

    m_output_rtv = RenderTargetView(0, 0, 0, m_output_rt->texture);

    m_vs = res_mgr->load_shader("shader/post_process/fullscreen_triangle_vs.glsl", GL_VERTEX_SHADER);
    m_fs = res_mgr->load_shader("shader/post_process/chromatic_aberration/chromatic_aberration_fs.glsl", GL_FRAGMENT_SHADER);

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

void ChromaticAberrationNode::execute(double delta, Renderer* renderer, Scene* scene, View* view)
{
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);

    m_program->use();

    renderer->bind_render_targets(1, &m_output_rtv, nullptr);

    glClear(GL_COLOR_BUFFER_BIT);
    glViewport(0, 0, m_graph->window_width(), m_graph->window_height());

    m_program->set_uniform("u_Strength", m_strength);

    if (m_program->set_uniform("s_Texture", 0) && m_texture)
        m_texture->texture->bind(0);

    render_fullscreen_triangle(renderer, view);
}

// -----------------------------------------------------------------------------------------------------------------------------------

void ChromaticAberrationNode::shutdown()
{
}

// -----------------------------------------------------------------------------------------------------------------------------------

std::string ChromaticAberrationNode::name()
{
    return "Chromatic Aberration";
}

// -----------------------------------------------------------------------------------------------------------------------------------
} // namespace nimble
#include "tone_map_node.h"
#include "../render_graph.h"
#include "../resource_manager.h"
#include "../renderer.h"
#include "../logger.h"

namespace nimble
{
DEFINE_RENDER_NODE_FACTORY(ToneMapNode)

// -----------------------------------------------------------------------------------------------------------------------------------

ToneMapNode::ToneMapNode(RenderGraph* graph) :
    RenderNode(graph)
{
}

// -----------------------------------------------------------------------------------------------------------------------------------

ToneMapNode::~ToneMapNode()
{
}

// -----------------------------------------------------------------------------------------------------------------------------------

void ToneMapNode::declare_connections()
{
    // Declare the inputs to this render node
    register_input_render_target("Color");
    register_input_render_target("Luminance");

    m_tonemap_rt = register_scaled_output_render_target("ToneMap", 1.0f, 1.0f, GL_TEXTURE_2D, GL_RGB8, GL_RGB, GL_UNSIGNED_BYTE);
}

// -----------------------------------------------------------------------------------------------------------------------------------

bool ToneMapNode::initialize_private(Renderer* renderer, ResourceManager* res_mgr)
{
    register_int_parameter("Operator", m_tone_map_operator, 0, 4);

    m_texture  = find_input_render_target("Color");
    m_avg_luma = find_input_render_target("Luminance");

    m_tonemap_rtv = RenderTargetView(0, 0, 0, m_tonemap_rt->texture);

    m_vs = res_mgr->load_shader("shader/post_process/fullscreen_triangle_vs.glsl", GL_VERTEX_SHADER);
    m_fs = res_mgr->load_shader("shader/post_process/tone_map_fs.glsl", GL_FRAGMENT_SHADER);

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

void ToneMapNode::execute(double delta, Renderer* renderer, Scene* scene, View* view)
{
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);

    m_program->use();

    renderer->bind_render_targets(1, &m_tonemap_rtv, nullptr);

    glClear(GL_COLOR_BUFFER_BIT);
    glViewport(0, 0, m_graph->window_width(), m_graph->window_height());

    m_program->set_uniform("u_ToneMapOperator", m_tone_map_operator);

    if (m_program->set_uniform("s_Texture", 0) && m_texture)
        m_texture->texture->bind(0);

    if (m_program->set_uniform("s_AvgLuma", 1) && m_avg_luma)
        m_avg_luma->texture->bind(1);

    render_fullscreen_triangle(renderer, view);
}

// -----------------------------------------------------------------------------------------------------------------------------------

void ToneMapNode::shutdown()
{
}

// -----------------------------------------------------------------------------------------------------------------------------------

std::string ToneMapNode::name()
{
    return "Tone Map";
}

// -----------------------------------------------------------------------------------------------------------------------------------
} // namespace nimble
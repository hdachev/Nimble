#include "present_node.h"
#include "../render_graph.h"
#include "../resource_manager.h"
#include "../renderer.h"
#include "../logger.h"

namespace nimble
{
DEFINE_RENDER_NODE_FACTORY(PresentNode)

// -----------------------------------------------------------------------------------------------------------------------------------

PresentNode::PresentNode(RenderGraph* graph) :
    RenderNode(graph)
{
}

// -----------------------------------------------------------------------------------------------------------------------------------

PresentNode::~PresentNode()
{
}

// -----------------------------------------------------------------------------------------------------------------------------------

void PresentNode::declare_connections()
{
    // Declare the inputs to this render node
    register_input_render_target("Color");

	m_texture = find_input_render_target("Color");
}

// -----------------------------------------------------------------------------------------------------------------------------------

bool PresentNode::initialize(Renderer* renderer, ResourceManager* res_mgr)
{
    m_vs = res_mgr->load_shader("shader/present/present_vs.glsl", GL_VERTEX_SHADER);
    m_fs = res_mgr->load_shader("shader/present/present_fs.glsl", GL_FRAGMENT_SHADER);

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

void PresentNode::execute(Renderer* renderer, Scene* scene, View* view)
{
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);
    glDisable(GL_CULL_FACE);

    m_program->use();

	glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glViewport(0, 0, m_graph->window_width(), m_graph->window_height());

    if (m_program->set_uniform("s_Texture", 0) && m_texture)
		m_texture->texture->bind(0);

    render_fullscreen_quad(renderer, view);

    glDepthFunc(GL_LESS);
}

// -----------------------------------------------------------------------------------------------------------------------------------

void PresentNode::shutdown()
{
}

// -----------------------------------------------------------------------------------------------------------------------------------

std::string PresentNode::name()
{
    return "Present";
}

// -----------------------------------------------------------------------------------------------------------------------------------
} // namespace nimble
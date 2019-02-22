#include "cubemap_skybox_node.h"
#include "../render_graph.h"
#include "../resource_manager.h"
#include "../renderer.h"
#include "../logger.h"

namespace nimble
{
DEFINE_RENDER_NODE_FACTORY(CubemapSkyboxNode)

// -----------------------------------------------------------------------------------------------------------------------------------

CubemapSkyboxNode::CubemapSkyboxNode(RenderGraph* graph) :
    RenderNode(graph)
{
}

// -----------------------------------------------------------------------------------------------------------------------------------

CubemapSkyboxNode::~CubemapSkyboxNode()
{
}

// -----------------------------------------------------------------------------------------------------------------------------------

void CubemapSkyboxNode::declare_connections()
{
    // Declare the inputs to this render node
    register_input_render_target("Color");
    register_input_render_target("Depth");

    // Since we're rendering to the render targets provided as input, we'll simply forward the input
    // render targets as outputs.
    m_color_rt = register_forwarded_output_render_target("Color");
    m_depth_rt = register_forwarded_output_render_target("Depth");
}

// -----------------------------------------------------------------------------------------------------------------------------------

bool CubemapSkyboxNode::initialize(Renderer* renderer, ResourceManager* res_mgr)
{
    m_scene_rtv = RenderTargetView(0, 0, 0, m_color_rt->texture);
    m_depth_rtv = RenderTargetView(0, 0, 0, m_depth_rt->texture);

    m_vs = res_mgr->load_shader("shader/skybox/cubemap_skybox_vs.glsl", GL_VERTEX_SHADER);
    m_fs = res_mgr->load_shader("shader/skybox/cubemap_skybox_fs.glsl", GL_FRAGMENT_SHADER);

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

void CubemapSkyboxNode::execute(Renderer* renderer, Scene* scene, View* view)
{
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);
    glDisable(GL_CULL_FACE);

    m_program->use();

    renderer->bind_render_targets(1, &m_scene_rtv, &m_depth_rtv);
    glViewport(0, 0, m_graph->window_width(), m_graph->window_height());

    if (m_program->set_uniform("s_Skybox", 0) && scene->env_map())
        scene->env_map()->bind(0);

    render_fullscreen_quad(renderer, view);

    glDepthFunc(GL_LESS);
}

// -----------------------------------------------------------------------------------------------------------------------------------

void CubemapSkyboxNode::shutdown()
{
}

// -----------------------------------------------------------------------------------------------------------------------------------

std::string CubemapSkyboxNode::name()
{
    return "Cubemap Skybox";
}

// -----------------------------------------------------------------------------------------------------------------------------------
} // namespace nimble
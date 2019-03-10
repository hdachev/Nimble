#include "average_luminance_node.h"
#include "../render_graph.h"
#include "../resource_manager.h"
#include "../renderer.h"
#include "../logger.h"

namespace nimble
{
DEFINE_RENDER_NODE_FACTORY(AverageLuminanceNode)

// -----------------------------------------------------------------------------------------------------------------------------------

AverageLuminanceNode::AverageLuminanceNode(RenderGraph* graph) :
    RenderNode(graph)
{
}

// -----------------------------------------------------------------------------------------------------------------------------------

AverageLuminanceNode::~AverageLuminanceNode()
{
}

// -----------------------------------------------------------------------------------------------------------------------------------

void AverageLuminanceNode::declare_connections()
{
	register_input_render_target("Color");
	
	m_luminance_rt = register_scaled_output_render_target("Luminance", 1.0f, 1.0f, GL_TEXTURE_2D, GL_R16F, GL_RED, GL_HALF_FLOAT, 1, 1, -1);
}

// -----------------------------------------------------------------------------------------------------------------------------------

bool AverageLuminanceNode::initialize(Renderer* renderer, ResourceManager* res_mgr)
{
    m_color_rt = find_input_render_target("Color");

	m_luminance_rtv = RenderTargetView(0, 0, 0, m_luminance_rt->texture);
	
	m_vs = res_mgr->load_shader("shader/post_process/fullscreen_triangle_vs.glsl", GL_VERTEX_SHADER);
	m_fs = res_mgr->load_shader("shader/post_process/luminance/luminance_fs.glsl", GL_FRAGMENT_SHADER);

	if (m_vs && m_fs)
	{
		m_program = renderer->create_program(m_vs, m_fs);
	    return true;
	}
	else
	    return false;
}

// -----------------------------------------------------------------------------------------------------------------------------------

void AverageLuminanceNode::execute(Renderer* renderer, Scene* scene, View* view)
{
	glDisable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);

    m_program->use();

    if (m_program->set_uniform("s_Texture", 0))
        m_color_rt->texture->bind(0);

    renderer->bind_render_targets(1, &m_luminance_rtv, nullptr);
    glViewport(0, 0, m_graph->window_width(), m_graph->window_height());
    glClear(GL_COLOR_BUFFER_BIT);

    render_fullscreen_triangle(renderer, nullptr);

	m_luminance_rt->texture->generate_mipmaps();
}

// -----------------------------------------------------------------------------------------------------------------------------------

void AverageLuminanceNode::shutdown()
{
}

// -----------------------------------------------------------------------------------------------------------------------------------

std::string AverageLuminanceNode::name()
{
    return "Average Luminance";
}

// -----------------------------------------------------------------------------------------------------------------------------------
} // namespace nimble
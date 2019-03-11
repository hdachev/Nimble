#include "hiz_node.h"
#include "../render_graph.h"
#include "../resource_manager.h"
#include "../renderer.h"
#include "../logger.h"

namespace nimble
{
DEFINE_RENDER_NODE_FACTORY(HiZNode)

// -----------------------------------------------------------------------------------------------------------------------------------

HiZNode::HiZNode(RenderGraph* graph) :
    RenderNode(graph)
{
}

// -----------------------------------------------------------------------------------------------------------------------------------

HiZNode::~HiZNode()
{
}

// -----------------------------------------------------------------------------------------------------------------------------------

void HiZNode::declare_connections()
{
	register_input_render_target("Depth");
	
	m_hiz_rt = register_scaled_output_render_target("HiZDepth", 1.0f, 1.0f, GL_TEXTURE_2D, GL_RG32F, GL_RG, GL_FLOAT, 1, 1, -1);
}

// -----------------------------------------------------------------------------------------------------------------------------------

bool HiZNode::initialize(Renderer* renderer, ResourceManager* res_mgr)
{
    m_depth_rt = find_input_render_target("Depth");

	create_rtvs();
	
	m_triangle_vs = res_mgr->load_shader("shader/post_process/fullscreen_triangle_vs.glsl", GL_VERTEX_SHADER);
	m_hiz_fs = res_mgr->load_shader("shader/post_process/hiz/hiz_fs.glsl", GL_FRAGMENT_SHADER);
	m_copy_fs  = res_mgr->load_shader("shader/post_process/hiz/hiz_copy_fs.glsl", GL_FRAGMENT_SHADER);
	
	if (m_triangle_vs)
	{
	    if (m_hiz_fs)
	        m_hiz_program = renderer->create_program(m_triangle_vs, m_hiz_fs);
	    else
	        return false;
	
	    if (m_copy_fs)
	        m_copy_program = renderer->create_program(m_triangle_vs, m_copy_fs);
	    else
	        return false;
	
	    return true;
	}
	else
	    return false;
}

// -----------------------------------------------------------------------------------------------------------------------------------

void HiZNode::execute(double delta, Renderer* renderer, Scene* scene, View* view)
{
    // Copy G-Buffer Depth into Mip 0 of HiZ
    copy_depth(renderer, scene, view);

    // Generate HiZ Chain
    downsample(renderer, scene, view);
}

// -----------------------------------------------------------------------------------------------------------------------------------

void HiZNode::copy_depth(Renderer* renderer, Scene* scene, View* view)
{
	glDisable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);

    m_copy_program->use();

	renderer->bind_render_targets(1, &m_hiz_rtv[0], nullptr);
    glViewport(0, 0, m_graph->window_width(), m_graph->window_height());

	glClearColor(1.0f, 0.0f, 0.0f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	if (m_copy_program->set_uniform("s_Texture", 0))
        m_depth_rt->texture->bind(0);

	render_fullscreen_triangle(renderer, nullptr);
}

// -----------------------------------------------------------------------------------------------------------------------------------

void HiZNode::downsample(Renderer* renderer, Scene* scene, View* view)
{
	glDisable(GL_DEPTH_TEST);
	glDisable(GL_CULL_FACE);

    m_hiz_program->use();

    for (uint32_t i = 1; i < m_num_rtv; i++)
    {
        float scale = pow(2, i);

		renderer->bind_render_targets(1, &m_hiz_rtv[i], nullptr);
        glViewport(0, 0, m_graph->window_width()/scale, m_graph->window_height()/scale);

		glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		 if (m_hiz_program->set_uniform("s_Texture", 0))
            m_hiz_rt->texture->bind(0);

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, i - 1);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, i - 1);

        render_fullscreen_triangle(renderer, nullptr);
    }

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, 0);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 1000);
}

// -----------------------------------------------------------------------------------------------------------------------------------

void HiZNode::create_rtvs()
{
	m_hiz_rt->texture->generate_mipmaps();

	m_num_rtv = m_hiz_rt->texture->mip_levels();
	
	for (uint32_t i = 0; i < m_num_rtv; i++)
		m_hiz_rtv[i] = RenderTargetView(0, 0, i, m_hiz_rt->texture);
}

// -----------------------------------------------------------------------------------------------------------------------------------

void HiZNode::shutdown()
{
}

// -----------------------------------------------------------------------------------------------------------------------------------

std::string HiZNode::name()
{
    return "HiZ Depth";
}

// -----------------------------------------------------------------------------------------------------------------------------------

void HiZNode::on_window_resized(const uint32_t& w, const uint32_t& h)
{
	create_rtvs();
}

// -----------------------------------------------------------------------------------------------------------------------------------
} // namespace nimble
#include "cubemap_skybox_node.h"
#include "../render_graph.h"
#include "../resource_manager.h"
#include "../renderer.h"
#include "../logger.h"

namespace nimble
{
DEFINE_RENDER_NODE_FACTORY(CubemapSkyboxNode)

	// -----------------------------------------------------------------------------------------------------------------------------------

	CubemapSkyboxNode::CubemapSkyboxNode(RenderGraph* graph) : FullscreenRenderNode(graph)
	{

	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	CubemapSkyboxNode::~CubemapSkyboxNode()
	{

	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	bool CubemapSkyboxNode::register_resources()
	{
		register_input_render_target("Scene");
		register_input_render_target("Depth");

		return true;
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	bool CubemapSkyboxNode::initialize()
	{
		InputRenderTarget* scene_rt = find_input_render_target_slot("Scene");
		InputRenderTarget* depth_rt = find_input_render_target_slot("Depth");

		if (scene_rt && depth_rt)
		{
			m_scene_rtv = RenderTargetView(0, 0, 0, scene_rt->prev_render_target->texture);
			m_depth_rtv = RenderTargetView(0, 0, 0, depth_rt->prev_render_target->texture);

			attach_sub_pass("Skybox Render", std::bind(&CubemapSkyboxNode::render_skybox, this, std::placeholders::_1));

			m_vs = m_graph->renderer()->resource_manager()->load_shader("shader/skybox/cubemap_skybox_vs.glsl", GL_VERTEX_SHADER);
			m_fs = m_graph->renderer()->resource_manager()->load_shader("shader/skybox/cubemap_skybox_fs.glsl", GL_FRAGMENT_SHADER);

			if (m_vs && m_fs)
			{
				m_program = m_graph->renderer()->create_program(m_vs, m_fs);

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
		else
			return false;
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

	void CubemapSkyboxNode::render_skybox(const View* view)
	{
		glEnable(GL_DEPTH_TEST);
		glDepthFunc(GL_LEQUAL);
		glDisable(GL_CULL_FACE);

		m_program->use();

		m_graph->renderer()->per_view_ssbo()->bind_range(0, sizeof(PerViewUniforms) * view->uniform_idx, sizeof(PerViewUniforms));

		m_graph->renderer()->bind_render_targets(1, &m_scene_rtv, &m_depth_rtv);
		glViewport(0, 0, m_graph->window_width(), m_graph->window_height());

		if (m_program->set_uniform("s_Skybox", 0) && view->scene->env_map())
			view->scene->env_map()->bind(0);

		glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

		glDepthFunc(GL_LESS);
	}

	// -----------------------------------------------------------------------------------------------------------------------------------
} // namespace nimble
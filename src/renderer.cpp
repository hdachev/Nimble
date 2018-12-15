#include "renderer.h"
#include "camera.h"
#include "material.h"
#include "mesh.h"
#include "logger.h"
#include "utility.h"
#include "imgui.h"
#include "entity.h"
#include "global_graphics_resources.h"
#include "constants.h"
#include "profiler.h"
#include "render_graph.h"

#include <gtc/matrix_transform.hpp>
#include <fstream>

namespace nimble
{
	// -----------------------------------------------------------------------------------------------------------------------------------

	Renderer::Renderer() {}

	// -----------------------------------------------------------------------------------------------------------------------------------

	Renderer::~Renderer() {}

	// -----------------------------------------------------------------------------------------------------------------------------------

	void Renderer::render()
	{
		queue_default_views();

		cull_scene();

		render_all_views();

		m_num_active_views = 0;
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	void Renderer::set_scene(std::shared_ptr<Scene> scene)
	{
		m_scene = scene;
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	void Renderer::queue_view(View view)
	{
		if (m_num_active_views == MAX_VIEWS)
			NIMBLE_LOG_ERROR("Maximum number of Views reached (64)");
		else
			m_active_views[m_num_active_views++] = view;
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	void Renderer::push_directional_light_views(View& dependent_view)
	{

	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	void Renderer::push_spot_light_views()
	{

	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	void Renderer::push_point_light_views()
	{

	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	void Renderer::clear_all_views()
	{

	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	void Renderer::cull_scene()
	{

	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	void Renderer::queue_default_views()
	{
		if (m_scene)
		{
			// Allocate view for scene camera
			Camera* camera = m_scene->camera();
			View scene_view;

			scene_view.m_enabled = true;
			scene_view.m_culling = true;
			scene_view.m_direction = camera->m_forward;
			scene_view.m_position = camera->m_position;
			scene_view.m_view_mat = camera->m_view;
			scene_view.m_projection_mat = camera->m_projection;
			scene_view.m_vp_mat = camera->m_view_projection;
			scene_view.m_prev_vp_mat = camera->m_prev_view_projection;
			scene_view.m_inv_view_mat = glm::inverse(camera->m_view);
			scene_view.m_inv_projection_mat = glm::inverse(camera->m_projection);
			scene_view.m_inv_vp_mat = glm::inverse(camera->m_view_projection);
			scene_view.m_jitter = glm::vec4(camera->m_prev_jitter, camera->m_current_jitter);
			scene_view.m_render_target_array_slice = 0;
			scene_view.m_render_target_cubemap_slice = 0;
			scene_view.m_render_target = nullptr;

			// @TODO: Create shadow views for scene views
		}
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	void Renderer::render_all_views()
	{
		for (uint32_t i = 0; i < m_num_active_views; i++)
		{
			View& view = m_active_views[i];
			view.m_id = i;

			if (view.m_graph)
				view.m_graph->execute(view);
			else
				NIMBLE_LOG_ERROR("Render Graph not assigned for View!");
		}
	}
	
	// -----------------------------------------------------------------------------------------------------------------------------------
}
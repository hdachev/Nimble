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
#include "geometry.h"

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

		update_uniforms();

		cull_scene();

		render_all_views();

		clear_all_views();
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	void Renderer::set_scene(std::shared_ptr<Scene> scene)
	{
		m_scene = scene;
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	void Renderer::set_scene_render_graph(RenderGraph* graph)
	{
		if (graph)
		{
			m_scene_render_graph = graph;

			if (!m_scene_render_graph->initialize())
				NIMBLE_LOG_ERROR("Failed to initialize Scene Render Graph!");
		}
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	void Renderer::queue_view(View view)
	{
		if (m_num_active_views == MAX_VIEWS)
			NIMBLE_LOG_ERROR("Maximum number of Views reached (64)");
		else
		{
			uint32_t idx = m_num_active_views++;

			Frustum frustum;
			frustum_from_matrix(frustum, view.m_vp_mat);

			m_active_views[idx] = view;
			m_active_frustums[idx] = frustum;
		}
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
		m_num_active_views = 0;
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	void Renderer::on_window_resized(const uint32_t& w, const uint32_t& h)
	{
		GlobalGraphicsResources::initialize_render_targets(w, h);

		if (m_scene_render_graph)
			m_scene_render_graph->on_window_resized(w, h);
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	void Renderer::update_uniforms()
	{
		if (m_scene)
		{
			// Update per entity uniforms
			Entity* entities = m_scene->entities();

			for (uint32_t i = 0; i < m_scene->entity_count(); i++)
			{
				Entity& entity = entities[i];

				m_per_entity_uniforms[i].modalMat = entity.m_transform;
				m_per_entity_uniforms[i].lastModelMat = entity.m_prev_transform;
				m_per_entity_uniforms[i].worldPos = glm::vec4(entity.m_position, 0.0f);
			}

			void* ptr = GlobalGraphicsResources::per_entity_ubo()->map(GL_WRITE_ONLY);
			memcpy(ptr, &m_per_entity_uniforms[0], sizeof(PerEntityUniforms) * m_scene->entity_count());
			GlobalGraphicsResources::per_entity_ubo()->unmap();

			// Update per view uniforms
			for (uint32_t i = 0; i < m_num_active_views; i++)
			{
				View& view = m_active_views[i];

				m_per_view_uniforms[i].viewMat = view.m_view_mat;
				m_per_view_uniforms[i].projMat = view.m_projection_mat;
				m_per_view_uniforms[i].viewProj= view.m_vp_mat;
				m_per_view_uniforms[i].lastViewProj = view.m_prev_vp_mat;
				m_per_view_uniforms[i].invProj = view.m_inv_projection_mat;
				m_per_view_uniforms[i].invView = view.m_inv_view_mat;
				m_per_view_uniforms[i].invViewProj = view.m_inv_vp_mat;
				m_per_view_uniforms[i].viewPos = glm::vec4(view.m_position, 0.0f);
			}

			ptr = GlobalGraphicsResources::per_view_ubo()->map(GL_WRITE_ONLY);
			memcpy(ptr, &m_per_view_uniforms[0], sizeof(PerViewUniforms) * m_num_active_views);
			GlobalGraphicsResources::per_view_ubo()->unmap();
		}
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	void Renderer::cull_scene()
	{
		if (m_scene)
		{
			Entity* entities = m_scene->entities();

			for (uint32_t i = 0; i < m_scene->entity_count(); i++)
			{
				Entity& entity = entities[i];

				entity.m_obb.position = entity.m_position;
				entity.m_obb.orientation = glm::mat3(entity.m_transform);
				
				for (uint32_t j = 0; j < m_num_active_views; j++)
				{
					if (m_active_views[j].m_culling)
					{
						if (intersects(m_active_frustums[j], entity.m_obb))
						{
							entity.set_visible(j);

#ifdef ENABLE_SUBMESH_CULLING
							for (uint32_t k = 0; k < entity.m_mesh->submesh_count(); k++)
							{
								SubMesh& submesh = entity.m_mesh->submesh(k);
								glm::vec3 center = (submesh.min_extents + submesh.max_extents) / 2.0f;

								entity.m_submesh_spheres[k].position = center + entity.m_position;

								if (intersects(m_active_frustums[j], entity.m_submesh_spheres[k]))
									entity.set_submesh_visible(k, j);
								else
									entity.set_submesh_invisible(k, j);
							}
#endif
						}
						else
							entity.set_invisible(j);
					}
					else
						entity.set_visible(j);
				}
			}
		}
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
			scene_view.m_dest_render_target_view = nullptr;
			scene_view.m_graph = m_scene_render_graph;
			scene_view.m_scene = m_scene.get();

			// @TODO: Create shadow views for scene views

			// Finally queue the scene view
			queue_view(scene_view);
		}
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	void Renderer::render_all_views()
	{
		for (uint32_t i = 0; i < m_num_active_views; i++)
		{
			View& view = m_active_views[i];

			if (view.m_enabled)
			{
				view.m_id = i;

				if (view.m_graph)
					view.m_graph->execute(view);
				else
					NIMBLE_LOG_ERROR("Render Graph not assigned for View!");
			}
		}
	}
	
	// -----------------------------------------------------------------------------------------------------------------------------------
}
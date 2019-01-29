#include "renderer.h"
#include "camera.h"
#include "material.h"
#include "mesh.h"
#include "logger.h"
#include "utility.h"
#include "imgui.h"
#include "entity.h"
#include "constants.h"
#include "profiler.h"
#include "render_graph.h"
#include "geometry.h"
#include "profiler.h"

#include <gtc/matrix_transform.hpp>
#include <fstream>

namespace nimble
{
	static const uint32_t kDirectionalLightShadowMapSizes[] =
	{
		512,
		1024,
		2048,
		4096
	};

	static const uint32_t kSpotLightShadowMapSizes[] =
	{
		512,
		1024,
		2048,
		4096
	};

	static const uint32_t kPointShadowMapSizes[] =
	{
		256,
		512,
		1024,
		2048
	};

	struct RenderTargetKey
	{
		uint32_t face = UINT32_MAX;
		uint32_t layer = UINT32_MAX;
		uint32_t mip_level = UINT32_MAX;
		uint32_t gl_id = UINT32_MAX;
	};

	struct FramebufferKey
	{
		RenderTargetKey rt_keys[8];
		RenderTargetKey depth_key;
	};

	static glm::vec3 s_cube_view_params[6][2] = 
	{
		{ glm::vec3(1.0, 0.0, 0.0), glm::vec3(0.0, -1.0, 0.0) },
		{ glm::vec3(-1.0, 0.0, 0.0), glm::vec3(0.0, -1.0, 0.0) },
		{ glm::vec3(0.0, 1.0, 0.0), glm::vec3(0.0, 0.0, 1.0) },
		{ glm::vec3(0.0, -1.0, 0.0), glm::vec3(0.0, 0.0, -1.0) },
		{ glm::vec3(0.0, 0.0, 1.0), glm::vec3(0.0, -1.0, 0.0) },
		{ glm::vec3(0.0, 0.0, -1.0), glm::vec3(0.0, -1.0, 0.0) }
	};

	// -----------------------------------------------------------------------------------------------------------------------------------

	Renderer::Renderer(Settings settings) : m_settings(settings) { }

	// -----------------------------------------------------------------------------------------------------------------------------------

	Renderer::~Renderer() {}

	// -----------------------------------------------------------------------------------------------------------------------------------

	bool Renderer::initialize(const uint32_t& w, const uint32_t& h)
	{
		m_window_width = w;
		m_window_height = h;

		m_directional_light_shadow_maps.reset();
		m_spot_light_shadow_maps.reset();
		m_point_light_shadow_maps.reset();

		// Create shadow maps
		m_directional_light_shadow_maps = std::make_shared<Texture2D>(kDirectionalLightShadowMapSizes[m_settings.shadow_map_quality], kDirectionalLightShadowMapSizes[m_settings.shadow_map_quality], m_settings.cascade_count * MAX_SHADOW_CASTING_DIRECTIONAL_LIGHTS, 1, 1, GL_DEPTH_COMPONENT32F, GL_DEPTH_COMPONENT, GL_FLOAT, false);
		m_spot_light_shadow_maps = std::make_shared<Texture2D>(kSpotLightShadowMapSizes[m_settings.shadow_map_quality], kSpotLightShadowMapSizes[m_settings.shadow_map_quality], MAX_SHADOW_CASTING_SPOT_LIGHTS, 1, 1, GL_DEPTH_COMPONENT32F, GL_DEPTH_COMPONENT, GL_FLOAT, false);
		m_point_light_shadow_maps = std::make_shared<TextureCube>(kPointShadowMapSizes[m_settings.shadow_map_quality], kPointShadowMapSizes[m_settings.shadow_map_quality], MAX_SHADOW_CASTING_POINT_LIGHTS, 1, GL_DEPTH_COMPONENT32F, GL_DEPTH_COMPONENT, GL_FLOAT, false);

		m_directional_light_shadow_maps->set_min_filter(GL_LINEAR);
		m_spot_light_shadow_maps->set_min_filter(GL_LINEAR);
		m_point_light_shadow_maps->set_min_filter(GL_LINEAR);

		// Create shadow map Render Target Views
		for (uint32_t i = 0; i < MAX_SHADOW_CASTING_DIRECTIONAL_LIGHTS; i++)
		{
			for (uint32_t j = 0; j < m_settings.cascade_count; j++)
				m_directionl_light_rt_views.push_back({ 0, i * m_settings.cascade_count + j, 0, m_directional_light_shadow_maps });
		}

		for (uint32_t i = 0; i < MAX_SHADOW_CASTING_SPOT_LIGHTS; i++)
			m_spot_light_rt_views.push_back({ 0, i, 0, m_spot_light_shadow_maps });

		for (uint32_t i = 0; i < MAX_SHADOW_CASTING_POINT_LIGHTS; i++)
		{
			for (uint32_t j = 0; j < 6; j++)
				m_point_light_rt_views.push_back({ j, i, 0, m_point_light_shadow_maps });
		}

		// Common resources
		m_per_view = std::make_unique<UniformBuffer>(GL_DYNAMIC_DRAW, MAX_VIEWS * sizeof(PerViewUniforms));
		m_per_entity = std::make_unique<UniformBuffer>(GL_DYNAMIC_DRAW, MAX_ENTITIES * sizeof(PerEntityUniforms));
		m_per_scene = std::make_unique<ShaderStorageBuffer>(GL_DYNAMIC_DRAW, sizeof(PerSceneUniforms));

		create_cube();

		bake_render_graphs();

		for (auto& current_graph : m_registered_render_graphs)
		{
			current_graph->on_window_resized(m_window_width, m_window_height);

			if (!current_graph->initialize())
				return false;
		}

		return true;
	}

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

	void Renderer::shutdown()
	{
		// Delete common geometry VBO's and VAO's.
		m_cube_vao.reset();
		m_cube_vbo.reset();

		// Delete framebuffer
		for (int i = 0; i < m_fbo_cache.size(); i++)
		{
			NIMBLE_SAFE_DELETE(m_fbo_cache.m_value[i]);
			m_fbo_cache.remove(m_fbo_cache.m_key[i]);
		}

		// Clean up Shader Cache
		m_shader_cache.shutdown();

		// Delete programs.
		for (auto itr : m_program_cache)
			itr.second.reset();

		m_per_view.reset();
		m_per_entity.reset();
		m_per_scene.reset();

		m_directional_light_shadow_maps.reset();
		m_spot_light_shadow_maps.reset();
		m_point_light_shadow_maps.reset();
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	void Renderer::set_settings(Settings settings)
	{
		m_settings = settings;
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	void Renderer::set_scene(std::shared_ptr<Scene> scene)
	{
		m_scene = scene;
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	void Renderer::register_render_graph(std::shared_ptr<RenderGraph> graph)
	{
		for (auto& current_graph : m_registered_render_graphs)
		{
			if (current_graph->name() == graph->name())
			{
				NIMBLE_LOG_WARNING("Attempting to register the same Render Graph twice: " + graph->name());
				return;
			}
		}

		if (graph->build())
			m_registered_render_graphs.push_back(graph);
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	void Renderer::set_scene_render_graph(std::shared_ptr<RenderGraph> graph)
	{
		if (graph)
			m_scene_render_graph = graph;	
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	void Renderer::set_directional_light_render_graph(std::shared_ptr<ShadowRenderGraph> graph)
	{
		if (graph)
			m_directional_light_render_graph = graph;
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	void Renderer::set_spot_light_render_graph(std::shared_ptr<ShadowRenderGraph> graph)
	{
		if (graph)
			m_spot_light_render_graph = graph;
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	void Renderer::set_point_light_render_graph(std::shared_ptr<ShadowRenderGraph> graph)
	{
		if (graph)
			m_point_light_render_graph = graph;
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
			frustum_from_matrix(frustum, view.vp_mat);

			m_active_views[idx] = view;
			m_active_frustums[idx] = frustum;
		}
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	void Renderer::push_directional_light_views(View& dependent_view)
	{
		if (!m_scene.expired())
		{
			auto scene = m_scene.lock();

			uint32_t shadow_casting_light_idx = 0;
			DirectionalLight* lights = scene->directional_lights();

			for (uint32_t light_idx = 0; light_idx < scene->directional_light_count(); light_idx++)
			{
				DirectionalLight& light = lights[light_idx];

				if (light.casts_shadow)
				{
					for (uint32_t cascade_idx = 0; cascade_idx < m_settings.cascade_count; cascade_idx++)
					{
						View light_view;

						light_view.enabled = true;
						light_view.culling = true;
						light_view.direction = light.transform.forward();
						light_view.position = light.transform.position;
						light_view.view_mat = glm::mat4(1.0f); // @TODO
						light_view.projection_mat = glm::mat4(1.0f); // @TODO
						light_view.vp_mat = glm::mat4(1.0f); // @TODO
						light_view.prev_vp_mat = glm::mat4(1.0f); // @TODO
						light_view.inv_view_mat = glm::mat4(1.0f); // @TODO
						light_view.inv_projection_mat = glm::mat4(1.0f); // @TODO
						light_view.inv_vp_mat = glm::mat4(1.0f); // @TODO
						light_view.jitter = glm::vec4(0.0);
						light_view.dest_render_target_view = &m_directionl_light_rt_views[shadow_casting_light_idx * m_settings.cascade_count + cascade_idx];
						light_view.graph = m_directional_light_render_graph;
						light_view.scene = scene.get();
						light_view.type = VIEW_DIRECTIONAL_LIGHT;
						light_view.light_index = light_idx;

						queue_view(light_view);
					}

					shadow_casting_light_idx++;
				}	

				// Stop adding views if max number of shadow casting lights is already queued.
				if (shadow_casting_light_idx == (MAX_SHADOW_CASTING_DIRECTIONAL_LIGHTS - 1))
					break;
			}
		}
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	void Renderer::push_spot_light_views()
	{
		if (!m_scene.expired())
		{
			auto scene = m_scene.lock();

			uint32_t shadow_casting_light_idx = 0;
			SpotLight* lights = scene->spot_lights();

			for (uint32_t light_idx = 0; light_idx < scene->spot_light_count(); light_idx++)
			{
				SpotLight& light = lights[light_idx];

				if (light.casts_shadow)
				{
					View light_view;

					light_view.enabled = true;
					light_view.culling = true;
					light_view.direction = light.transform.forward();
					light_view.position = light.transform.position;
					light_view.view_mat = glm::mat4(1.0f); // @TODO
					light_view.projection_mat = glm::mat4(1.0f); // @TODO
					light_view.vp_mat = glm::mat4(1.0f); // @TODO
					light_view.prev_vp_mat = glm::mat4(1.0f); // @TODO
					light_view.inv_view_mat = glm::mat4(1.0f); // @TODO
					light_view.inv_projection_mat = glm::mat4(1.0f); // @TODO
					light_view.inv_vp_mat = glm::mat4(1.0f); // @TODO
					light_view.jitter = glm::vec4(0.0);
					light_view.dest_render_target_view = &m_spot_light_rt_views[shadow_casting_light_idx];
					light_view.graph = m_spot_light_render_graph;
					light_view.scene = scene.get();
					light_view.type = VIEW_SPOT_LIGHT;
					light_view.light_index = light_idx;

					queue_view(light_view);

					shadow_casting_light_idx++;
				}

				// Stop adding views if max number of shadow casting lights is already queued.
				if (shadow_casting_light_idx == (MAX_SHADOW_CASTING_SPOT_LIGHTS - 1))
					break;
			}
		}
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	void Renderer::push_point_light_views()
	{
		if (!m_scene.expired())
		{
			auto scene = m_scene.lock();

			uint32_t shadow_casting_light_idx = 0;
			PointLight* lights = scene->point_lights();

			for (uint32_t light_idx = 0; light_idx < scene->point_light_count(); light_idx++)
			{
				PointLight& light = lights[light_idx];

				if (light.casts_shadow)
				{
					for (uint32_t face_idx = 0; face_idx < 6; face_idx++)
					{
						View light_view;

						light_view.enabled = true;
						light_view.culling = true;
						light_view.direction = light.transform.forward();
						light_view.position = light.transform.position;
						light_view.view_mat = glm::lookAt(light.transform.position, light.transform.position + s_cube_view_params[face_idx][0], s_cube_view_params[face_idx][1]);
						light_view.projection_mat = glm::perspective(glm::radians(90.0f), 1.0f, 1.0f, light.range);
						light_view.vp_mat = light_view.projection_mat * light_view.view_mat;
						light_view.prev_vp_mat = glm::mat4(1.0f);
						light_view.inv_view_mat = glm::inverse(light_view.view_mat);
						light_view.inv_projection_mat = glm::inverse(light_view.projection_mat);
						light_view.inv_vp_mat = glm::inverse(light_view.vp_mat);
						light_view.jitter = glm::vec4(0.0);
						light_view.dest_render_target_view = &m_point_light_rt_views[shadow_casting_light_idx * 6 + face_idx];
						light_view.graph = m_point_light_render_graph;
						light_view.scene = scene.get();
						light_view.type = VIEW_POINT_LIGHT;
						light_view.light_index = light_idx;

						queue_view(light_view);
					}

					shadow_casting_light_idx++;
				}

				// Stop adding views if max number of shadow casting lights is already queued.
				if (shadow_casting_light_idx == (MAX_SHADOW_CASTING_POINT_LIGHTS - 1))
					break;
			}
		}
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	void Renderer::clear_all_views()
	{
		m_num_active_views = 0;
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	void Renderer::on_window_resized(const uint32_t& w, const uint32_t& h)
	{
		m_window_width = w;
		m_window_height = h;

		for (auto& desc : m_rt_cache)
		{
			if (desc.rt->is_scaled() && desc.rt->target == GL_TEXTURE_2D)
			{
				uint32_t width = float(w) * desc.rt->scale_w;
				uint32_t height = float(h) * desc.rt->scale_h;

				Texture2D* texture = (Texture2D*)desc.rt->texture.get();
				texture->resize(width, height);
			}
		}

		if (m_scene_render_graph)
			m_scene_render_graph->on_window_resized(w, h);
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	std::shared_ptr<Program> Renderer::create_program(const std::shared_ptr<Shader>& vs, const std::shared_ptr<Shader>& fs)
	{
		std::string id = std::to_string(vs->id()) + "-";
		id += std::to_string(fs->id());

		if (m_program_cache.find(id) != m_program_cache.end() && m_program_cache[id].lock())
			return m_program_cache[id].lock();
		else
		{
			Shader* shaders[] = { vs.get(), fs.get() };

			std::shared_ptr<Program> program = std::make_shared<Program>(2, shaders);

			if (!program)
			{
				NIMBLE_LOG_ERROR("Program failed to link!");
				return nullptr;
			}

			m_program_cache[id] = program;

			program->uniform_block_binding("u_PerView", 0);
			program->uniform_block_binding("u_PerScene", 1);
			program->uniform_block_binding("u_PerEntity", 2);

			return program;
		}
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	std::shared_ptr<Program> Renderer::create_program(const std::vector<std::shared_ptr<Shader>>& shaders)
	{
		std::vector<Shader*> shaders_raw;
		std::string id = "";

		for (const auto& shader : shaders)
		{
			shaders_raw.push_back(shader.get());
			id += std::to_string(shader->id());
			id += "-";
		}

		if (m_program_cache.find(id) != m_program_cache.end() && m_program_cache[id].lock())
			return m_program_cache[id].lock();
		else
		{
			std::shared_ptr<Program> program = std::make_shared<Program>(shaders_raw.size(), shaders_raw.data());

			if (!program)
			{
				NIMBLE_LOG_ERROR("Program failed to link!");
				return nullptr;
			}

			m_program_cache[id] = program;

			program->uniform_block_binding("u_PerView", 0);
			program->uniform_block_binding("u_PerScene", 1);
			program->uniform_block_binding("u_PerEntity", 2);

			return program;
		}
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	Framebuffer* Renderer::framebuffer_for_render_targets(const uint32_t& num_render_targets, const RenderTargetView* rt_views, const RenderTargetView* depth_view)
	{
		FramebufferKey key;

		if (rt_views)
		{
			for (int i = 0; i < num_render_targets; i++)
			{
				key.rt_keys[i].face = rt_views[i].face;
				key.rt_keys[i].layer = rt_views[i].layer;
				key.rt_keys[i].mip_level = rt_views[i].mip_level;
				key.rt_keys[i].gl_id = rt_views[i].texture->id();
			}
		}

		if (depth_view)
		{
			key.depth_key.face = depth_view->face;
			key.depth_key.layer = depth_view->layer;
			key.depth_key.mip_level = depth_view->mip_level;
			key.depth_key.gl_id = depth_view->texture->id();;
		}

		uint64_t hash = murmur_hash_64(&key, sizeof(FramebufferKey), 5234);

		Framebuffer* fbo = nullptr;

		if (!m_fbo_cache.get(hash, fbo))
		{
			fbo = new Framebuffer();

			if (rt_views)
			{
				if (num_render_targets == 0)
				{
					if (rt_views[0].texture->target() == GL_TEXTURE_2D || rt_views[0].texture->target() == GL_TEXTURE_2D_ARRAY)
						fbo->attach_render_target(0, rt_views[0].texture.get(), rt_views[0].layer, rt_views[0].mip_level);
					else if (rt_views[0].texture->target() == GL_TEXTURE_CUBE_MAP || rt_views[0].texture->target() == GL_TEXTURE_CUBE_MAP_ARRAY)
						fbo->attach_render_target(0, static_cast<TextureCube*>(rt_views[0].texture.get()), rt_views[0].face, rt_views[0].layer, rt_views[0].mip_level);
				}
				else
				{
					Texture* textures[8];

					for (int i = 0; i < num_render_targets; i++)
						textures[i] = rt_views[i].texture.get();

					fbo->attach_multiple_render_targets(num_render_targets, textures);
				}

			}

			if (depth_view)
			{
				if (depth_view->texture->target() == GL_TEXTURE_2D || depth_view->texture->target() == GL_TEXTURE_2D_ARRAY)
					fbo->attach_depth_stencil_target(depth_view->texture.get(), depth_view->layer, depth_view->mip_level);
				else if (depth_view->texture->target() == GL_TEXTURE_CUBE_MAP || depth_view->texture->target() == GL_TEXTURE_CUBE_MAP_ARRAY)
					fbo->attach_depth_stencil_target(static_cast<TextureCube*>(depth_view->texture.get()), depth_view->face, depth_view->layer, depth_view->mip_level);
			}

			m_fbo_cache.set(hash, fbo);
		}

		return fbo;
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	void Renderer::bind_render_targets(const uint32_t& num_render_targets, const RenderTargetView* rt_views, const RenderTargetView* depth_view)
	{
		Framebuffer* fbo = framebuffer_for_render_targets(num_render_targets, rt_views, depth_view);

		if (fbo)
			fbo->bind();
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	void Renderer::create_cube()
	{
		float cube_vertices[] =
		{
			// back face
			-1.0f, -1.0f, -1.0f,  0.0f,  0.0f, -1.0f, 0.0f, 0.0f, // bottom-left
			1.0f,  1.0f, -1.0f,  0.0f,  0.0f, -1.0f, 1.0f, 1.0f, // top-right
			1.0f, -1.0f, -1.0f,  0.0f,  0.0f, -1.0f, 1.0f, 0.0f, // bottom-right         
			1.0f,  1.0f, -1.0f,  0.0f,  0.0f, -1.0f, 1.0f, 1.0f, // top-right
			-1.0f, -1.0f, -1.0f,  0.0f,  0.0f, -1.0f, 0.0f, 0.0f, // bottom-left
			-1.0f,  1.0f, -1.0f,  0.0f,  0.0f, -1.0f, 0.0f, 1.0f, // top-left
			// front face
			-1.0f, -1.0f,  1.0f,  0.0f,  0.0f,  1.0f, 0.0f, 0.0f, // bottom-left
			1.0f, -1.0f,  1.0f,  0.0f,  0.0f,  1.0f, 1.0f, 0.0f, // bottom-right
			1.0f,  1.0f,  1.0f,  0.0f,  0.0f,  1.0f, 1.0f, 1.0f, // top-right
			1.0f,  1.0f,  1.0f,  0.0f,  0.0f,  1.0f, 1.0f, 1.0f, // top-right
			-1.0f,  1.0f,  1.0f,  0.0f,  0.0f,  1.0f, 0.0f, 1.0f, // top-left
			-1.0f, -1.0f,  1.0f,  0.0f,  0.0f,  1.0f, 0.0f, 0.0f, // bottom-left
			// left face
			-1.0f,  1.0f,  1.0f, -1.0f,  0.0f,  0.0f, 1.0f, 0.0f, // top-right
			-1.0f,  1.0f, -1.0f, -1.0f,  0.0f,  0.0f, 1.0f, 1.0f, // top-left
			-1.0f, -1.0f, -1.0f, -1.0f,  0.0f,  0.0f, 0.0f, 1.0f, // bottom-left
			-1.0f, -1.0f, -1.0f, -1.0f,  0.0f,  0.0f, 0.0f, 1.0f, // bottom-left
			-1.0f, -1.0f,  1.0f, -1.0f,  0.0f,  0.0f, 0.0f, 0.0f, // bottom-right
			-1.0f,  1.0f,  1.0f, -1.0f,  0.0f,  0.0f, 1.0f, 0.0f, // top-right
			// right face
			1.0f,  1.0f,  1.0f,  1.0f,  0.0f,  0.0f, 1.0f, 0.0f, // top-left
			1.0f, -1.0f, -1.0f,  1.0f,  0.0f,  0.0f, 0.0f, 1.0f, // bottom-right
			1.0f,  1.0f, -1.0f,  1.0f,  0.0f,  0.0f, 1.0f, 1.0f, // top-right         
			1.0f, -1.0f, -1.0f,  1.0f,  0.0f,  0.0f, 0.0f, 1.0f, // bottom-right
			1.0f,  1.0f,  1.0f,  1.0f,  0.0f,  0.0f, 1.0f, 0.0f, // top-left
			1.0f, -1.0f,  1.0f,  1.0f,  0.0f,  0.0f, 0.0f, 0.0f, // bottom-left     
			// bottom face
			-1.0f, -1.0f, -1.0f,  0.0f, -1.0f,  0.0f, 0.0f, 1.0f, // top-right
			1.0f, -1.0f, -1.0f,  0.0f, -1.0f,  0.0f, 1.0f, 1.0f, // top-left
			1.0f, -1.0f,  1.0f,  0.0f, -1.0f,  0.0f, 1.0f, 0.0f, // bottom-left
			1.0f, -1.0f,  1.0f,  0.0f, -1.0f,  0.0f, 1.0f, 0.0f, // bottom-left
			-1.0f, -1.0f,  1.0f,  0.0f, -1.0f,  0.0f, 0.0f, 0.0f, // bottom-right
			-1.0f, -1.0f, -1.0f,  0.0f, -1.0f,  0.0f, 0.0f, 1.0f, // top-right
			// top face
			-1.0f,  1.0f, -1.0f,  0.0f,  1.0f,  0.0f, 0.0f, 1.0f, // top-left
			1.0f,  1.0f , 1.0f,  0.0f,  1.0f,  0.0f, 1.0f, 0.0f, // bottom-right
			1.0f,  1.0f, -1.0f,  0.0f,  1.0f,  0.0f, 1.0f, 1.0f, // top-right     
			1.0f,  1.0f,  1.0f,  0.0f,  1.0f,  0.0f, 1.0f, 0.0f, // bottom-right
			-1.0f,  1.0f, -1.0f,  0.0f,  1.0f,  0.0f, 0.0f, 1.0f, // top-left
			-1.0f,  1.0f,  1.0f,  0.0f,  1.0f,  0.0f, 0.0f, 0.0f  // bottom-left        
		};

		m_cube_vbo = std::make_shared<VertexBuffer>(GL_STATIC_DRAW, sizeof(cube_vertices), (void*)cube_vertices);

		VertexAttrib attribs[] =
		{
			{ 3,GL_FLOAT, false, 0, },
		{ 3,GL_FLOAT, false, sizeof(float) * 3 },
		{ 2,GL_FLOAT, false, sizeof(float) * 6 }
		};

		m_cube_vao = std::make_shared<VertexArray>(m_cube_vbo.get(), nullptr, sizeof(float) * 8, 3, attribs);

		if (!m_cube_vbo || !m_cube_vao)
		{
			NIMBLE_LOG_FATAL("Failed to create Vertex Buffers/Arrays");
		}
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	int32_t Renderer::find_render_target_last_usage(std::shared_ptr<RenderTarget> rt)
	{
		int32_t node_gid = 0;
		int32_t last_node_id = -1;

		for (uint32_t graph_idx = 0; graph_idx < m_registered_render_graphs.size(); graph_idx++)
		{
			std::shared_ptr<RenderGraph> graph = m_registered_render_graphs[graph_idx];

			for (uint32_t node_idx = 0; node_idx < graph->node_count(); node_idx++)
			{
				std::shared_ptr<RenderNode> node = graph->node(node_idx);

				for (uint32_t rt_idx = 0; rt_idx < node->input_render_target_count(); rt_idx++)
				{
					std::shared_ptr<RenderTarget> input_rt = node->input_render_target(rt_idx);

					if (rt->id == input_rt->id)
						last_node_id = node_gid;
				}

				node_gid++;
			}
		}

		return last_node_id;
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	bool Renderer::is_aliasing_candidate(std::shared_ptr<RenderTarget> rt, uint32_t write_node, uint32_t read_node, const RenderTargetDesc& rt_desc)
	{
		bool format_match = rt->internal_format == rt_desc.rt->texture->internal_format() && 
							rt->target == rt_desc.rt->texture->target() &&
							rt->scale_h == rt_desc.rt->scale_h &&
							rt->scale_w == rt_desc.rt->scale_w &&
							rt->w == rt_desc.rt->w &&
							rt->h == rt_desc.rt->h;

		if (!format_match)
			return false;

		for (auto& pair : rt_desc.lifetimes)
		{
			// Is this an intermediate texture?
			if (write_node == read_node)
			{
				if (write_node == pair.first || write_node == pair.second)
					return false;
			}
			else
			{
				if (write_node == pair.first || write_node == pair.second || read_node == pair.first || read_node == pair.second || (write_node > pair.first && read_node < pair.second))
					return false;
			}
		}

		return true;
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	void Renderer::create_texture_for_render_target(std::shared_ptr<RenderTarget> rt, uint32_t write_node, uint32_t read_node)
	{
		// Create new texture
		std::shared_ptr<Texture> tex;

		if (rt->is_scaled())
		{
			rt->w = uint32_t(rt->scale_w * float(m_window_width));
			rt->h = uint32_t(rt->scale_h * float(m_window_height));
		}

		if (rt->target == GL_TEXTURE_2D)
			tex = std::make_shared<Texture2D>(rt->w, rt->h, rt->array_size, rt->mip_levels, rt->num_samples, rt->internal_format, rt->format, rt->type);
		else if (rt->target == GL_TEXTURE_CUBE_MAP)
			tex = std::make_shared<TextureCube>(rt->w, rt->h, rt->array_size, rt->mip_levels, rt->internal_format, rt->format, rt->type);

		// Assign it to the current output Render Target
		rt->texture = tex;

		// Push it into the list of total Render Targets
		m_rt_cache.push_back({ rt, {{ write_node, read_node }} });
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	void Renderer::bake_render_graphs()
	{
		uint32_t node_gid = 0;

		for (uint32_t graph_idx = 0; graph_idx < m_registered_render_graphs.size(); graph_idx++)
		{
			std::shared_ptr<RenderGraph> graph = m_registered_render_graphs[graph_idx];

			for (uint32_t node_idx = 0; node_idx < graph->node_count(); node_idx++)
			{
				std::shared_ptr<RenderNode> node = graph->node(node_idx);

				for (uint32_t rt_idx = 0; rt_idx < node->output_render_target_count(); rt_idx++)
				{
					std::shared_ptr<RenderTarget> rt = node->output_render_target(rt_idx);

					// Find last usage of output
					int32_t current_node_id = node_gid;
					int32_t last_node_id = find_render_target_last_usage(rt);

					bool found_texture = false;

					// Try to find an already created texture that does not have an overlapping lifetime
					for (auto& desc : m_rt_cache)
					{
						// Check if current Texture is suitable to be aliased
						if (is_aliasing_candidate(rt, current_node_id, last_node_id, desc))
						{
							found_texture = true;
							// Add the new lifetime to the existing texture
							desc.lifetimes.push_back({ current_node_id, last_node_id });
							rt->texture = desc.rt->texture;
						}
					}

					if (!found_texture)
						create_texture_for_render_target(rt, current_node_id, last_node_id);
				}

				for (uint32_t rt_idx = 0; rt_idx < node->intermediate_render_target_count(); rt_idx++)
				{
					std::shared_ptr<RenderTarget> rt = node->intermediate_render_target(rt_idx);

					bool found_texture = false;

					// Try to find an already created texture that does not have an overlapping lifetime
					for (auto& desc : m_rt_cache)
					{
						// Check if current Texture is suitable to be aliased
						if (is_aliasing_candidate(rt, node_gid, node_gid, desc))
						{
							found_texture = true;
							// Add the new lifetime to the existing texture
							desc.lifetimes.push_back({ node_gid, node_gid });
							rt->texture = desc.rt->texture;
						}
					}

					if (!found_texture)
						create_texture_for_render_target(rt, node_gid, node_gid);
				}

				node_gid++;
			}
		}
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	void Renderer::update_uniforms()
	{
		if (!m_scene.expired())
		{
			auto scene = m_scene.lock();

			// Update per entity uniforms
			Entity* entities = scene->entities();

			for (uint32_t i = 0; i < scene->entity_count(); i++)
			{
				Entity& entity = entities[i];

				m_per_entity_uniforms[i].modal_mat = entity.transform.model;
				m_per_entity_uniforms[i].last_model_mat = entity.transform.prev_model;
				m_per_entity_uniforms[i].world_pos = glm::vec4(entity.transform.position, 0.0f);
			}

			void* ptr = m_per_entity->map(GL_WRITE_ONLY);
			memcpy(ptr, &m_per_entity_uniforms[0], sizeof(PerEntityUniforms) * scene->entity_count());
			m_per_entity->unmap();

			// Update per view uniforms
			for (uint32_t i = 0; i < m_num_active_views; i++)
			{
				View& view = m_active_views[i];

				m_per_view_uniforms[i].view_mat = view.view_mat;
				m_per_view_uniforms[i].proj_mat = view.projection_mat;
				m_per_view_uniforms[i].view_proj= view.vp_mat;
				m_per_view_uniforms[i].last_view_proj = view.prev_vp_mat;
				m_per_view_uniforms[i].inv_proj = view.inv_projection_mat;
				m_per_view_uniforms[i].inv_view = view.inv_view_mat;
				m_per_view_uniforms[i].inv_view_proj = view.inv_vp_mat;
				m_per_view_uniforms[i].view_pos = glm::vec4(view.position, 0.0f);
			}

			ptr = m_per_view->map(GL_WRITE_ONLY);
			memcpy(ptr, &m_per_view_uniforms[0], sizeof(PerViewUniforms) * m_num_active_views);
			m_per_view->unmap();

			// Update per scene uniforms
			DirectionalLight* dir_lights = scene->directional_lights();

			m_per_scene_uniforms.directional_light_count = scene->directional_light_count();

			for (uint32_t light_idx = 0; light_idx < m_per_scene_uniforms.directional_light_count; light_idx++)
			{
				DirectionalLight& light = dir_lights[light_idx];

				m_per_scene_uniforms.directional_lights[light_idx].direction = glm::vec4(light.transform.forward(), 0.0f);
				m_per_scene_uniforms.directional_lights[light_idx].color_intensity = glm::vec4(light.color, light.intensity);
				m_per_scene_uniforms.directional_lights[light_idx].casts_shadow = light.casts_shadow ? 1 : 0;
			}

			SpotLight* spot_lights = scene->spot_lights();

			m_per_scene_uniforms.spot_light_count = scene->spot_light_count();

			for (uint32_t light_idx = 0; light_idx < m_per_scene_uniforms.spot_light_count; light_idx++)
			{
				SpotLight& light = spot_lights[light_idx];

				m_per_scene_uniforms.spot_lights[light_idx].direction_range = glm::vec4(light.transform.forward(), light.range);
				m_per_scene_uniforms.spot_lights[light_idx].color_intensity = glm::vec4(light.color, light.intensity);
				m_per_scene_uniforms.spot_lights[light_idx].position_cone_angle = glm::vec4(light.transform.position, cosf(glm::radians(light.cone_angle)));
				m_per_scene_uniforms.spot_lights[light_idx].casts_shadow = light.casts_shadow ? 1 : 0;
			}

			PointLight* point_lights = scene->point_lights();

			m_per_scene_uniforms.point_light_count = scene->point_light_count();

			for (uint32_t light_idx = 0; light_idx < m_per_scene_uniforms.point_light_count; light_idx++)
			{
				PointLight& light = point_lights[light_idx];

				m_per_scene_uniforms.point_lights[light_idx].position_range = glm::vec4(light.transform.position, light.range);
				m_per_scene_uniforms.point_lights[light_idx].color_intensity = glm::vec4(light.color, light.intensity);
				m_per_scene_uniforms.point_lights[light_idx].casts_shadow = light.casts_shadow ? 1 : 0;
			}

			ptr = m_per_scene->map(GL_WRITE_ONLY);
			memcpy(ptr, &m_per_scene_uniforms, sizeof(PerSceneUniforms));
			m_per_scene->unmap();
		}
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	void Renderer::cull_scene()
	{
		Profiler::begin_cpu_sample(PROFILER_FRUSTUM_CULLING);
		 
		if (!m_scene.expired())
		{
			auto scene = m_scene.lock();

			Entity* entities = scene->entities();

			for (uint32_t i = 0; i < scene->entity_count(); i++)
			{
				Entity& entity = entities[i];

				entity.obb.position = entity.transform.position;
				entity.obb.orientation = glm::mat3(entity.transform.model);
				
				for (uint32_t j = 0; j < m_num_active_views; j++)
				{
					if (m_active_views[j].culling)
					{
						if (intersects(m_active_frustums[j], entity.obb))
						{
							entity.set_visible(j);

#ifdef ENABLE_SUBMESH_CULLING
							for (uint32_t k = 0; k < entity.mesh->submesh_count(); k++)
							{
								SubMesh& submesh = entity.mesh->submesh(k);
								glm::vec3 center = (submesh.min_extents + submesh.max_extents) / 2.0f;

								entity.submesh_spheres[k].position = center + entity.transform.position;

								if (intersects(m_active_frustums[j], entity.submesh_spheres[k]))
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

		Profiler::end_cpu_sample(PROFILER_FRUSTUM_CULLING);
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	void Renderer::queue_default_views()
	{
		if (!m_scene.expired())
		{
			auto scene = m_scene.lock();

			// Allocate view for scene camera
			auto camera = scene->camera();
			View scene_view;

			scene_view.enabled = true;
			scene_view.culling = true;
			scene_view.direction = camera->m_forward;
			scene_view.position = camera->m_position;
			scene_view.view_mat = camera->m_view;
			scene_view.projection_mat = camera->m_projection;
			scene_view.vp_mat = camera->m_view_projection;
			scene_view.prev_vp_mat = camera->m_prev_view_projection;
			scene_view.inv_view_mat = glm::inverse(camera->m_view);
			scene_view.inv_projection_mat = glm::inverse(camera->m_projection);
			scene_view.inv_vp_mat = glm::inverse(camera->m_view_projection);
			scene_view.jitter = glm::vec4(camera->m_prev_jitter, camera->m_current_jitter);
			scene_view.dest_render_target_view = nullptr;
			scene_view.graph = m_scene_render_graph;
			scene_view.scene = scene.get();
			scene_view.type = VIEW_STANDARD;

			// Queue shadow views
			push_point_light_views();

			// Finally queue the scene view
			queue_view(scene_view);
		}
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	void Renderer::render_all_views()
	{
		if (m_num_active_views > 0)
		{
			for (uint32_t i = 0; i < m_num_active_views; i++)
			{
				View& view = m_active_views[i];

				if (view.enabled)
				{
					view.id = i;

					if (view.graph)
						view.graph->execute(view);
					else
						NIMBLE_LOG_ERROR("Render Graph not assigned for View!");
				}
			}
		}
		else
			glClear(GL_COLOR_BUFFER_BIT);
	}
	
	// -----------------------------------------------------------------------------------------------------------------------------------
}
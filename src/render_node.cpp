#include "render_node.h"
#include "render_graph.h"
#include "profiler.h"
#include "view.h"
#include "scene.h"
#include "shader_library.h"
#include "logger.h"
#include "renderer.h"

namespace nimble
{
	// -----------------------------------------------------------------------------------------------------------------------------------

	RenderNode::RenderNode(RenderNodeType type, RenderGraph* graph) : m_enabled(true), m_render_node_type(type), m_graph(graph), m_total_time_cpu(0.0f), m_total_time_gpu(0.0f)
	{

	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	RenderNode::~RenderNode()
	{

	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	std::shared_ptr<RenderTarget> RenderNode::find_output_render_target(const std::string& name)
	{
		for (auto& pair : m_output_rts)
		{
			if (pair.first == name)
				return pair.second;
		}

		return nullptr;
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	std::shared_ptr<RenderTarget> RenderNode::find_intermediate_render_target(const std::string& name)
	{
		for (auto& pair : m_intermediate_rts)
		{
			if (pair.first == name)
				return pair.second;
		}

		return nullptr;
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	std::shared_ptr<RenderTarget> RenderNode::find_input_render_target(const std::string& name)
	{
		for (auto& pair : m_input_rts)
		{
			if (pair.first == name)
				return pair.second;
		}

		return nullptr;
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	std::shared_ptr<Buffer> RenderNode::find_input_buffer(const std::string& name)
	{
		for (auto& pair : m_input_buffers)
		{
			if (pair.first == name)
				return pair.second;
		}

		return nullptr;
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	void RenderNode::set_input(const std::string& name, std::shared_ptr<RenderTarget> rt)
	{
		for (auto& pair : m_input_rts)
		{
			if (pair.first == name)
			{
				pair.second = rt;
				return;
			}
		}

		NIMBLE_LOG_ERROR("No input render target slot named: " + name);
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	void RenderNode::set_input(const std::string& name, std::shared_ptr<Buffer> buffer)
	{
		for (auto& pair : m_input_buffers)
		{
			if (pair.first == name)
			{
				pair.second = buffer;
				return;
			}
		}

		NIMBLE_LOG_ERROR("No input buffer slot named: " + name);
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	void RenderNode::timing_total(float& cpu_time, float& gpu_time)
	{
		cpu_time = m_total_time_cpu;
		gpu_time = m_total_time_gpu;
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	bool RenderNode::initialize_internal()
	{
		m_passthrough_name = name();
		m_passthrough_name += "_Passthrough";

		return true;
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	bool RenderNode::register_resources()
	{
		return true;
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	void RenderNode::passthrough()
	{

	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	uint32_t RenderNode::flags()
	{
		return NODE_USAGE_DEFAULT;
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	void RenderNode::on_window_resized(const uint32_t& w, const uint32_t& h)
	{

	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	void RenderNode::register_input_render_target(const std::string& name)
	{
		for (auto& pair : m_input_rts)
		{
			if (pair.first == name)
			{
				NIMBLE_LOG_ERROR("Input render target slot already registered: " + name);
				return;
			}
		}

		m_input_rts.push_back({ name, nullptr });
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	void RenderNode::register_input_buffer(const std::string& name)
	{
		for (auto& pair : m_input_buffers)
		{
			if (pair.first == name)
			{
				NIMBLE_LOG_ERROR("Input buffer slot already registered: " + name);
				return;
			}
		}

		m_input_buffers.push_back({ name, nullptr });
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	std::shared_ptr<RenderTarget> RenderNode::register_output_render_target(const std::string& name, const uint32_t& w, const uint32_t& h, GLenum target, GLenum internal_format, GLenum format, GLenum type, uint32_t num_samples, uint32_t array_size, uint32_t mip_levels)
	{
		for (auto& pair : m_output_rts)
		{
			if (pair.first == name)
			{
				NIMBLE_LOG_ERROR("Output render target already registered: " + name);
				return nullptr;
			}
		}

		std::shared_ptr<RenderTarget> rt = std::make_shared<RenderTarget>();
		
		rt->w = w;
		rt->h = h;
		rt->scale_w = 0.0f;
		rt->scale_h = 0.0f;
		rt->target = target;
		rt->internal_format = internal_format;
		rt->format = format;
		rt->type = type;
		rt->num_samples = num_samples;
		rt->array_size = array_size;
		rt->mip_levels = mip_levels;

		m_output_rts.push_back({ name, rt });

		return rt;
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	std::shared_ptr<RenderTarget> RenderNode::register_scaled_output_render_target(const std::string& name, const float& w, const float& h, GLenum target, GLenum internal_format, GLenum format, GLenum type, uint32_t num_samples, uint32_t array_size, uint32_t mip_levels)
	{
		for (auto& pair : m_output_rts)
		{
			if (pair.first == name)
			{
				NIMBLE_LOG_ERROR("Output render target already registered: " + name);
				return nullptr;
			}
		}

		std::shared_ptr<RenderTarget> rt = std::make_shared<RenderTarget>();

		rt->w = 0;
		rt->h = 0;
		rt->scale_w = w;
		rt->scale_h = h;
		rt->target = target;
		rt->internal_format = internal_format;
		rt->format = format;
		rt->type = type;
		rt->num_samples = num_samples;
		rt->array_size = array_size;
		rt->mip_levels = mip_levels;

		m_output_rts.push_back({ name, rt });

		return rt;
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	std::shared_ptr<RenderTarget> RenderNode::register_intermediate_render_target(const std::string& name, const uint32_t& w, const uint32_t& h, GLenum target, GLenum internal_format, GLenum format, GLenum type, uint32_t num_samples, uint32_t array_size, uint32_t mip_levels)
	{
		for (auto& pair : m_intermediate_rts)
		{
			if (pair.first == name)
			{
				NIMBLE_LOG_ERROR("Intermediate render target already registered: " + name);
				return nullptr;
			}
		}

		std::shared_ptr<RenderTarget> rt = std::make_shared<RenderTarget>();

		rt->w = w;
		rt->h = h;
		rt->scale_w = 0.0f;
		rt->scale_h = 0.0f;
		rt->target = target;
		rt->internal_format = internal_format;
		rt->format = format;
		rt->type = type;
		rt->num_samples = num_samples;
		rt->array_size = array_size;
		rt->mip_levels = mip_levels;

		m_intermediate_rts.push_back({ name, rt });

		return rt;
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	std::shared_ptr<RenderTarget> RenderNode::register_scaled_intermediate_render_target(const std::string& name, const float& w, const float& h, GLenum target, GLenum internal_format, GLenum format, GLenum type, uint32_t num_samples, uint32_t array_size, uint32_t mip_levels)
	{
		for (auto& pair : m_intermediate_rts)
		{
			if (pair.first == name)
			{
				NIMBLE_LOG_ERROR("Intermediate render target already registered: " + name);
				return nullptr;
			}
		}

		std::shared_ptr<RenderTarget> rt = std::make_shared<RenderTarget>();

		rt->w = 0;
		rt->h = 0;
		rt->scale_w = w;
		rt->scale_h = h;
		rt->target = target;
		rt->internal_format = internal_format;
		rt->format = format;
		rt->type = type;
		rt->num_samples = num_samples;
		rt->array_size = array_size;
		rt->mip_levels = mip_levels;

		m_intermediate_rts.push_back({ name, rt });

		return rt;
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	SceneRenderNode::Params::Params()
	{
		view = nullptr;
		num_rt_views = 1;
		rt_views = nullptr;
		depth_views = nullptr;
		x = 0;
		y = 0;
		w = 0;
		h = 0;
		clear_flags = GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT;
		num_clear_colors = 0;
		clear_colors[0][0] = 0.0f;
		clear_colors[0][1] = 0.0f;
		clear_colors[0][2] = 0.0f;
		clear_colors[0][3] = 0.0f;
		clear_depth = 1;
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	SceneRenderNode::SceneRenderNode(RenderGraph* graph) : RenderNode(RENDER_NODE_SCENE, graph)
	{
		
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	SceneRenderNode::~SceneRenderNode()
	{

	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	bool SceneRenderNode::initialize_internal()
	{
		bool status = RenderNode::initialize_internal();
		m_library = m_graph->renderer()->shader_cache().load_library(vs_template_path(), fs_template_path());

		return status && m_library != nullptr;
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	void SceneRenderNode::execute(const View& view)
	{
		float cpu_time, gpu_time;

		Profiler::result(name(), cpu_time, gpu_time);

		m_total_time_cpu = cpu_time;
		m_total_time_gpu = gpu_time;

		Profiler::begin_sample(name());

		execute_internal(view);

		Profiler::end_sample(name());
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	void SceneRenderNode::set_shader_uniforms(const View* view, Program* program, int32_t& tex_unit)
	{

	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	void SceneRenderNode::render_scene(const Params& params)
	{
		if (params.rt_views || params.depth_views)
			m_graph->renderer()->bind_render_targets(params.num_rt_views, params.rt_views, params.depth_views);
		else
			glBindFramebuffer(GL_FRAMEBUFFER, 0);

		glViewport(params.x, params.y, params.w, params.h);

		if (params.enable_depth)
			glEnable(GL_DEPTH_TEST);
		else
			glDisable(GL_DEPTH_TEST);

		if (params.cull_face == GL_NONE)
			glDisable(GL_CULL_FACE);
		else
		{
			glEnable(GL_CULL_FACE);
			glCullFace(params.cull_face);
		}

		if (params.clear_flags != 0)
		{
			if (params.rt_views || params.depth_views)
			{
				if (params.num_clear_colors == 1)
					glClearColor(params.clear_colors[0][0], params.clear_colors[0][1], params.clear_colors[0][2], params.clear_colors[0][3]);
				else
				{
					for (uint32_t i = 0; i < params.num_clear_colors; i++)
						glClearBufferfv(GL_COLOR, i, &params.clear_colors[i][0]);
				}
			}
			else
			{
				if (params.num_clear_colors == 1)
					glClearColor(params.clear_colors[0][0], params.clear_colors[0][1], params.clear_colors[0][2], params.clear_colors[0][3]);
			}

			glClearDepth(params.clear_depth);

			glClear(params.clear_flags);
		}

		if (params.view->scene)
		{
			Scene* scene = params.view->scene;
			Entity* entities = scene->entities();

			// Bind buffers
			if (HAS_BIT_FLAG(flags(), NODE_USAGE_PER_VIEW_UBO))
				m_graph->renderer()->per_view_ubo()->bind_range(0, sizeof(PerViewUniforms) * params.view->id, sizeof(PerViewUniforms));

			if (HAS_BIT_FLAG(flags(), NODE_USAGE_POINT_LIGHTS) || HAS_BIT_FLAG(flags(), NODE_USAGE_SPOT_LIGHTS) || HAS_BIT_FLAG(flags(), NODE_USAGE_DIRECTIONAL_LIGHTS))
				m_graph->renderer()->per_scene_ssbo()->bind_base(2);

			for (uint32_t i = 0; i < scene->entity_count(); i++)
			{
				Entity& e = entities[i];

				if (!params.view->culling || (params.view->culling && e.visibility(params.view->id)))
				{
					// Bind mesh VAO
					e.mesh->bind();

					for (uint32_t j = 0; j < e.mesh->submesh_count(); j++)
					{
						SubMesh& s = e.mesh->submesh(j);

						int32_t tex_unit = 0;

#ifdef ENABLE_SUBMESH_CULLING
						if (!params.view->culling || (params.view->culling && e.submesh_visibility(j, params.view->id)))
						{
#endif
							ProgramKey& key = s.material->program_key();

							key.set_mesh_type(e.mesh->type());

							// Lookup shader program from library
							Program* program = m_library->lookup_program(key);

							if (!program)
							{
								program = m_library->create_program(e.mesh->type(), 
																	flags(), 
																	s.material, 
																	m_graph->type() == RENDER_GRAPH_STANDARD ? m_graph->renderer()->directional_light_render_graph() : nullptr,
																	m_graph->type() == RENDER_GRAPH_STANDARD ? m_graph->renderer()->spot_light_render_graph() : nullptr,
																	m_graph->type() == RENDER_GRAPH_STANDARD ? m_graph->renderer()->point_light_render_graph() : nullptr);
							}

							program->use();

							// Bind material
							if (HAS_BIT_FLAG(flags(), NODE_USAGE_MATERIAL_ALBEDO) && !s.material->surface_texture(TEXTURE_TYPE_ALBEDO))
								program->set_uniform("u_Albedo", s.material->uniform_albedo());

							if (HAS_BIT_FLAG(flags(), NODE_USAGE_MATERIAL_EMISSIVE) && !s.material->surface_texture(TEXTURE_TYPE_EMISSIVE))
								program->set_uniform("u_Emissive", s.material->uniform_emissive());

							if ((HAS_BIT_FLAG(flags(), NODE_USAGE_MATERIAL_ROUGH_SMOOTH) && !s.material->surface_texture(TEXTURE_TYPE_ROUGH_SMOOTH)) || (HAS_BIT_FLAG(flags(), NODE_USAGE_MATERIAL_METAL_SPEC) && !s.material->surface_texture(TEXTURE_TYPE_METAL_SPEC)))
								program->set_uniform("u_MetalRough", glm::vec4(s.material->uniform_metallic(), s.material->uniform_roughness(), 0.0f, 0.0f));

							s.material->bind(program, tex_unit);

							if ((HAS_BIT_FLAG(flags(), NODE_USAGE_SHADOW_MAPPING)) && program->set_uniform("s_DirectionalLightShadowMaps", tex_unit))
								m_graph->renderer()->directional_light_shadow_maps()->bind(tex_unit++);

							if ((HAS_BIT_FLAG(flags(), NODE_USAGE_SHADOW_MAPPING)) && program->set_uniform("s_SpotLightShadowMaps", tex_unit))
								m_graph->renderer()->spot_light_shadow_maps()->bind(tex_unit++);

							if ((HAS_BIT_FLAG(flags(), NODE_USAGE_SHADOW_MAPPING)) && program->set_uniform("s_PointLightShadowMaps", tex_unit))
								m_graph->renderer()->point_light_shadow_maps()->bind(tex_unit++);

							if (HAS_BIT_FLAG(flags(), NODE_USAGE_PER_OBJECT_UBO))
								m_graph->renderer()->per_entity_ubo()->bind_range(1, sizeof(PerEntityUniforms) * i, sizeof(PerEntityUniforms));

							set_shader_uniforms(params.view, program, tex_unit);

							glDrawElementsBaseVertex(GL_TRIANGLES, s.index_count, GL_UNSIGNED_INT, (void*)(sizeof(unsigned int) * s.base_index), s.base_vertex);

#ifdef ENABLE_SUBMESH_CULLING
						}
#endif
					}
				}
			}
		}
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	MultiPassRenderNode::MultiPassRenderNode(RenderNodeType type, RenderGraph* graph) : RenderNode(type, graph)
	{

	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	MultiPassRenderNode::~MultiPassRenderNode()
	{

	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	void MultiPassRenderNode::execute(const View& view)
	{
		float cpu_time, gpu_time;

		m_total_time_cpu = 0;
		m_total_time_gpu = 0;

		if (is_enabled())
		{

			for (uint32_t i = 0; i < m_sub_passes.size(); i++)
			{
				const auto& pass = m_sub_passes[i];

				Profiler::result(pass.first, cpu_time, gpu_time);

				m_total_time_cpu += cpu_time;
				m_total_time_gpu += gpu_time;

				m_sub_pass_timings[i].first = cpu_time;
				m_sub_pass_timings[i].second = gpu_time;

				Profiler::begin_sample(pass.first);

				// Execute subpass.
				pass.second();

				Profiler::end_sample(pass.first);
			}
		}
		else
		{
			Profiler::result(m_passthrough_name, cpu_time, gpu_time);

			m_total_time_cpu += cpu_time;
			m_total_time_gpu += gpu_time;

			Profiler::begin_sample(m_passthrough_name);

			// Execute passthrough.
			passthrough();

			Profiler::end_sample(m_passthrough_name);
		}
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	void MultiPassRenderNode::attach_sub_pass(const std::string& node_name, std::function<void(void)> function)
	{
		std::string formatted_name = name();
		formatted_name += "_";
		formatted_name += node_name;

		m_sub_passes.push_back({ formatted_name, function });
		m_sub_pass_timings.push_back({ 0.0f, 0.0f });
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	void MultiPassRenderNode::timing_sub_pass(const uint32_t& index, std::string& name, float& cpu_time, float& gpu_time)
	{
		if (index < sub_pass_count())
		{
			name = m_sub_passes[index].first;
			cpu_time = m_sub_pass_timings[index].first;
			gpu_time = m_sub_pass_timings[index].second;
		}
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	FullscreenRenderNode::Params::Params()
	{
		scene = nullptr;
		view = nullptr;
		num_rt_views = 0;
		rt_views = nullptr;
		x = 0;
		y = 0;
		w = 0;
		h = 0;
		clear_flags = GL_COLOR_BUFFER_BIT;
		num_clear_colors = 0;
		clear_colors[0][0] = 0.0f;
		clear_colors[0][1] = 0.0f;
		clear_colors[0][2] = 0.0f;
		clear_colors[0][3] = 0.0f;
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	FullscreenRenderNode::FullscreenRenderNode(RenderGraph* graph) : MultiPassRenderNode(RENDER_NODE_FULLSCREEN, graph)
	{

	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	FullscreenRenderNode::~FullscreenRenderNode()
	{

	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	void FullscreenRenderNode::render_triangle(const Params& params)
	{
		if (params.rt_views)
			m_graph->renderer()->bind_render_targets(params.num_rt_views, params.rt_views, nullptr);
		else
			glBindFramebuffer(GL_FRAMEBUFFER, 0);

		glViewport(params.x, params.y, params.h, params.w);

		if (params.clear_flags != 0)
		{
			if (params.rt_views)
			{
				if (params.num_clear_colors == 1)
					glClearColor(params.clear_colors[0][0], params.clear_colors[0][1], params.clear_colors[0][2], params.clear_colors[0][3]);
				else
				{
					for (uint32_t i = 0; i < params.num_clear_colors; i++)
						glClearBufferfv(GL_COLOR, i, &params.clear_colors[i][0]);
				}
			}
			else
			{
				if (params.num_clear_colors == 1)
					glClearColor(params.clear_colors[0][0], params.clear_colors[0][1], params.clear_colors[0][2], params.clear_colors[0][3]);
			}

			glClear(params.clear_flags);
		}

		m_graph->renderer()->per_view_ubo()->bind_range(0, sizeof(PerViewUniforms) * params.view->id, sizeof(PerViewUniforms));

		// Render fullscreen triangle
		glDrawArrays(GL_TRIANGLES, 0, 3);
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	ComputeRenderNode::ComputeRenderNode(RenderGraph* graph) : MultiPassRenderNode(RENDER_NODE_COMPUTE, graph)
	{

	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	ComputeRenderNode::~ComputeRenderNode()
	{

	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	Buffer* ComputeRenderNode::find_output_buffer(const std::string& name)
	{
		if (m_output_buffers.find(name) != m_output_buffers.end())
			return m_output_buffers[name];
		else
			return nullptr;
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	Buffer* ComputeRenderNode::find_intermediate_buffer(const std::string& name)
	{
		if (m_intermediate_buffers.find(name) != m_intermediate_buffers.end())
			return m_intermediate_buffers[name];
		else
			return nullptr;
	}

	// -----------------------------------------------------------------------------------------------------------------------------------
} // namespace nimble
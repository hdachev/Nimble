#include "render_node.h"
#include "render_graph.h"
#include "profiler.h"
#include "view.h"
#include "scene.h"
#include "shader_cache.h"
#include "shader_library.h"

namespace nimble
{
	static uint32_t g_last_node_id = 0;

	// -----------------------------------------------------------------------------------------------------------------------------------

	RenderNode::RenderNode(RenderNodeType type, RenderGraph* graph) : m_enabled(true), m_render_node_type(type), m_graph(graph), m_id(g_last_node_id++), m_total_time_cpu(0.0f), m_total_time_gpu(0.0f)
	{

	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	RenderNode::~RenderNode()
	{

	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	RenderTarget* RenderNode::render_target_by_name(const std::string& name)
	{
		if (m_render_targets.find(name) != m_render_targets.end())
			return m_render_targets[name];
		else
			return nullptr;
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	RenderTarget* RenderNode::render_target_dependecy_by_name(const std::string& name)
	{
		if (m_rt_dependecies.find(name) != m_rt_dependecies.end())
			return m_rt_dependecies[name];
		else
			return nullptr;
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	Buffer* RenderNode::buffer_dependecy_by_name(const std::string& name)
	{
		if (m_buffer_dependecies.find(name) != m_buffer_dependecies.end())
			return m_buffer_dependecies[name];
		else
			return nullptr;
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	void RenderNode::set_dependency(const std::string& name, RenderTarget* rt)
	{
		rt->last_dependent_node_id = id();
		m_rt_dependecies[name] = rt;
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	void RenderNode::set_dependency(const std::string& name, Buffer* buffer)
	{
		m_buffer_dependecies[name] = buffer;
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

	std::shared_ptr<RenderTarget> RenderNode::register_render_target(const std::string& name, const uint32_t& w, const uint32_t& h, GLenum target, GLenum internal_format, GLenum format, GLenum type, uint32_t num_samples, uint32_t array_size, uint32_t mip_levels)
	{
		std::shared_ptr<RenderTarget> rt = GlobalGraphicsResources::request_render_target(m_graph->id(), id(), w, h, target, internal_format, format, type, num_samples, array_size, mip_levels);
		m_render_targets[name] = rt.get();

		return rt;
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	std::shared_ptr<RenderTarget> RenderNode::register_scaled_render_target(const std::string& name, const float& w, const float& h, GLenum target, GLenum internal_format, GLenum format, GLenum type, uint32_t num_samples, uint32_t array_size, uint32_t mip_levels)
	{
		std::shared_ptr<RenderTarget> rt = GlobalGraphicsResources::request_scaled_render_target(m_graph->id(), id(), w, h, target, internal_format, format, type, num_samples, array_size, mip_levels);
		m_render_targets[name] = rt.get();

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
		m_library = ShaderCache::load_library(vs_template_path(), fs_template_path());

		return status && m_library != nullptr;
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	void SceneRenderNode::render_scene(const Params& params)
	{
		if (params.rt_views || params.depth_views)
			GlobalGraphicsResources::bind_render_targets(params.num_rt_views, params.rt_views, params.depth_views);
		else
			glBindFramebuffer(GL_FRAMEBUFFER, 0);

		glViewport(params.x, params.y, params.w, params.h);

		glEnable(GL_DEPTH_TEST);
		glEnable(GL_CULL_FACE);
		glCullFace(GL_BACK);

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

		if (params.view->m_scene)
		{
			Scene* scene = params.view->m_scene;
			Entity* entities = scene->entities();

			// Bind buffers
			if (HAS_BIT_FLAG(flags(), NODE_USAGE_PER_VIEW_UBO))
				GlobalGraphicsResources::per_view_ubo()->bind_range(0, sizeof(PerViewUniforms) * params.view->m_id, sizeof(PerViewUniforms));

			if (HAS_BIT_FLAG(flags(), NODE_USAGE_POINT_LIGHTS))
				GlobalGraphicsResources::per_scene_point_lights_ubo()->bind_base(1);

			if (HAS_BIT_FLAG(flags(), NODE_USAGE_SPOT_LIGHTS))
				GlobalGraphicsResources::per_scene_spot_lights_ubo()->bind_base(2);

			if (HAS_BIT_FLAG(flags(), NODE_USAGE_DIRECTIONAL_LIGHTS))
				GlobalGraphicsResources::per_scene_directional_lights_ubo()->bind_base(3);

			for (uint32_t i = 0; i < scene->entity_count(); i++)
			{
				Entity& e = entities[i];

				if (!params.view->m_culling || (params.view->m_culling && e.visibility(params.view->m_id)))
				{
					// Bind mesh VAO
					e.mesh->bind();

					for (uint32_t j = 0; j < e.mesh->submesh_count(); j++)
					{
						SubMesh& s = e.mesh->submesh(j);

						int32_t tex_unit = 0;

#ifdef ENABLE_SUBMESH_CULLING
						if (!params.view->m_culling || (params.view->m_culling && e.submesh_visibility(j, params.view->m_id)))
						{
#endif
							ProgramKey& key = s.material->program_key();

							key.set_mesh_type(e.mesh->type());

							// Lookup shader program from library
							Program* program = m_library->lookup_program(key);

							if (!program)
								program = m_library->create_program(e.mesh->type(), flags(), s.material);

							program->use();

							// Bind material
							if (HAS_BIT_FLAG(flags(), NODE_USAGE_MATERIAL_ALBEDO) && !s.material->surface_texture(TEXTURE_TYPE_ALBEDO))
								program->set_uniform("u_Albedo", s.material->uniform_albedo());

							if (HAS_BIT_FLAG(flags(), NODE_USAGE_MATERIAL_EMISSIVE) && !s.material->surface_texture(TEXTURE_TYPE_EMISSIVE))
								program->set_uniform("u_Emissive", s.material->uniform_emissive());

							if ((HAS_BIT_FLAG(flags(), NODE_USAGE_MATERIAL_ROUGH_SMOOTH) && !s.material->surface_texture(TEXTURE_TYPE_ROUGH_SMOOTH)) || 
								(HAS_BIT_FLAG(flags(), NODE_USAGE_MATERIAL_METAL_SPEC) && !s.material->surface_texture(TEXTURE_TYPE_METAL_SPEC)))
								program->set_uniform("u_MetalRough", glm::vec4(s.material->uniform_metallic(), s.material->uniform_roughness(), 0.0f, 0.0f));

							s.material->bind(program, tex_unit);

							if (HAS_BIT_FLAG(flags(), NODE_USAGE_PER_OBJECT_UBO))
								GlobalGraphicsResources::per_entity_ubo()->bind_range(4, sizeof(PerEntityUniforms) * i, sizeof(PerEntityUniforms));

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
			GlobalGraphicsResources::bind_render_targets(params.num_rt_views, params.rt_views, nullptr);
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

		GlobalGraphicsResources::per_view_ubo()->bind_range(0, sizeof(PerViewUniforms) * params.view->m_id, sizeof(PerViewUniforms));

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
} // namespace nimble
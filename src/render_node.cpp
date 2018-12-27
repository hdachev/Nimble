#include "render_node.h"
#include "render_graph.h"
#include "profiler.h"
#include "view.h"
#include "scene.h"
#include "shader_library.h"

namespace nimble
{
	static uint32_t g_last_node_id = 0;

	// -----------------------------------------------------------------------------------------------------------------------------------

	RenderNode::RenderNode(RenderNodeType type, RenderGraph* graph) : m_enabled(true), m_render_node_type(type), m_graph(graph), m_id(g_last_node_id++), m_total_time_cpu(0.0f), m_total_time_gpu(0.0f)
	{
		m_passthrough_name = name();
		m_passthrough_name += "_Passthrough";
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

	void RenderNode::passthrough()
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

	SceneRenderNode::SceneRenderNode(RenderGraph* graph) : RenderNode(RENDER_NODE_SCENE, graph)
	{

	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	SceneRenderNode::~SceneRenderNode()
	{

	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	void SceneRenderNode::execute(const View& view)
	{

	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	void SceneRenderNode::render_scene(const Params& params)
	{
		if (params.rt_views || params.depth_views)
			GlobalGraphicsResources::bind_render_targets(params.num_rt_views, params.rt_views, params.depth_views);
		else
			glBindFramebuffer(GL_FRAMEBUFFER, 0);

		glViewport(params.x, params.y, params.h, params.w);

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

		if (params.scene)
		{
			Entity* entities = params.scene->entities();

			for (uint32_t i = 0; i < params.scene->entity_count(); i++)
			{
				Entity& e = entities[i];

				if (!params.view->m_culling || (params.view->m_culling && e.visibility(params.view->m_id)))
				{
					// Bind mesh VAO
					e.m_mesh->bind();

					for (uint32_t j = 0; j < e.m_mesh->submesh_count(); j++)
					{
						SubMesh& s = e.m_mesh->submesh(j);

						int32_t tex_unit = 0;

#ifdef ENABLE_SUBMESH_CULLING
						if (e.submesh_visibility(j, params.view->m_id))
						{
#endif
							ProgramKey& key = s.material->program_key();

							// Only static meshes for now
							key.set_mesh_type(MESH_TYPE_STATIC);

							// Lookup shader program from library
							Program* program = params.library->lookup_program(key);

							program->use();

							// Bind material
							s.material->bind(program, tex_unit);

							// Bind uniform buffers
							if (HAS_BIT_FLAG(flags(), NODE_USAGE_PER_VIEW_UBO))
								GlobalGraphicsResources::per_frame_ubo()->bind_base(0);

							if (HAS_BIT_FLAG(flags(), NODE_USAGE_PER_OBJECT_UBO))
								GlobalGraphicsResources::per_entity_ubo()->bind_base(1);

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
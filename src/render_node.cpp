#include "render_node.h"
#include "render_graph.h"
#include "profiler.h"
#include "view.h"

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

	void SceneRenderNode::execute(View* view)
	{

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

	void MultiPassRenderNode::execute(View* view)
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

	ComputeRenderNode::ComputeRenderNode(RenderGraph* graph) : MultiPassRenderNode(RENDER_NODE_COMPUTE, graph)
	{

	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	ComputeRenderNode::~ComputeRenderNode()
	{

	}

	// -----------------------------------------------------------------------------------------------------------------------------------
} // namespace nimble
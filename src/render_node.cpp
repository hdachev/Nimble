#include "render_node.h"
#include "render_graph.h"

namespace nimble
{
	static uint32_t g_last_node_id = 0;

	// -----------------------------------------------------------------------------------------------------------------------------------

	RenderNode::RenderNode(RenderGraph* graph) : m_graph(graph), m_id(g_last_node_id++)
	{
		
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	uint32_t RenderNode::id()
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
}
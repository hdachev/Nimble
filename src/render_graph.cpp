#include "render_graph.h"
#include "renderer.h"

namespace nimble
{
	// -----------------------------------------------------------------------------------------------------------------------------------

	RenderGraph::RenderGraph(Renderer* renderer) : m_renderer(renderer)
	{
		
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	RenderGraph::~RenderGraph()
	{

	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	bool RenderGraph::initialize()
	{
		for (auto& node : m_nodes)
		{
			if (!node->initialize())
				return false;
		}

		return true;
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	void RenderGraph::shutdown()
	{
		for (auto& node : m_nodes)
			node->shutdown();

		clear();
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	void RenderGraph::clear()
	{
		m_nodes.clear();
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	void RenderGraph::execute(const View& view)
	{
		for (auto& node : m_nodes)
			node->execute(view);
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	bool RenderGraph::attach_and_initialize_node(std::shared_ptr<RenderNode> node)
	{
		m_nodes.push_back(node);

		return node->initialize_internal() && node->register_resources();
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	std::shared_ptr<RenderNode> RenderGraph::node_by_name(const std::string& name)
	{
		for (const auto& node : m_nodes)
		{
			if (node->name() == name)
				return node;
		}

		return nullptr;
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	void RenderGraph::on_window_resized(const uint32_t& w, const uint32_t& h)
	{
		m_window_width = w;
		m_window_height = h;

		for (auto& node : m_nodes)
			node->on_window_resized(w, h);
	}

	// -----------------------------------------------------------------------------------------------------------------------------------
} // namespace nimble
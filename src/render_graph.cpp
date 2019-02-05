#include "render_graph.h"
#include "renderer.h"
#include "utility.h"
#include "logger.h"

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

	RenderGraphType RenderGraph::type()
	{
		return RENDER_GRAPH_STANDARD;
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

	void RenderGraph::execute(const View* view)
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

	ShadowRenderGraph::ShadowRenderGraph(Renderer* renderer) : RenderGraph(renderer)
	{

	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	bool ShadowRenderGraph::initialize()
	{
		bool result = RenderGraph::initialize();

		std::string includes;
		std::string defines;

		if (!utility::read_shader_separate(utility::path_for_resource("assets/" + sampling_source_path()), includes, m_sampling_source, defines))
		{
			NIMBLE_LOG_ERROR("Failed load Sampling Source: " + sampling_source_path());
			return false;
		}

		return true;
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	RenderGraphType ShadowRenderGraph::type()
	{
		return RENDER_GRAPH_SHADOW;
	}

	// -----------------------------------------------------------------------------------------------------------------------------------
} // namespace nimble
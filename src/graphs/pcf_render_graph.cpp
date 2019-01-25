#include "pcf_render_graph.h"
#include "../nodes/standard_depth_node.h"

namespace nimble
{
	// -----------------------------------------------------------------------------------------------------------------------------------

	PCFRenderGraph::PCFRenderGraph(Renderer* renderer) : RenderGraph(renderer)
	{

	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	std::string PCFRenderGraph::name()
	{
		return "PCF Render Graph";
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	bool PCFRenderGraph::build()
	{
		std::shared_ptr<StandardDepthNode> depth_node = std::make_shared<StandardDepthNode>(this);

		return attach_and_initialize_node(depth_node);
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	void PCFRenderGraph::refresh()
	{

	}

	// -----------------------------------------------------------------------------------------------------------------------------------
}
#include "forward_render_graph.h"
#include "../nodes/forward_render_node.h"

namespace nimble
{
	// -----------------------------------------------------------------------------------------------------------------------------------

	ForwardRenderGraph::ForwardRenderGraph(Renderer* renderer) : RenderGraph(renderer)
	{

	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	std::string ForwardRenderGraph::name()
	{
		return "Forward Render Graph";
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	bool ForwardRenderGraph::build()
	{
		std::shared_ptr<ForwardRenderNode> forward_node = std::make_shared<ForwardRenderNode>(this);

		return attach_and_initialize_node(forward_node);
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	void ForwardRenderGraph::refresh()
	{

	}

	// -----------------------------------------------------------------------------------------------------------------------------------
}
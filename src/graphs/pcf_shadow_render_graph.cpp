#include "pcf_shadow_render_graph.h"
#include "../nodes/standard_depth_node.h"

namespace nimble
{
	// -----------------------------------------------------------------------------------------------------------------------------------

	PCFShadowRenderGraph::PCFShadowRenderGraph(Renderer* renderer) : ShadowRenderGraph(renderer)
	{

	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	std::string PCFShadowRenderGraph::name()
	{
		return "PCF Shadow Render Graph";
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	bool PCFShadowRenderGraph::build()
	{
		std::shared_ptr<StandardDepthNode> depth_node = std::make_shared<StandardDepthNode>(this);

		return attach_and_initialize_node(depth_node);
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	void PCFShadowRenderGraph::refresh()
	{

	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	std::string PCFShadowRenderGraph::sampling_source()
	{
		return "shader/shadows/sampling/pcf.glsl";
	}

	// -----------------------------------------------------------------------------------------------------------------------------------
}
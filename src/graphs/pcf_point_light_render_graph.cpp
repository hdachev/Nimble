#include "pcf_point_light_render_graph.h"
#include "../nodes/pcf_point_light_depth_node.h"

namespace nimble
{
	// -----------------------------------------------------------------------------------------------------------------------------------

	PCFPointLightRenderGraph::PCFPointLightRenderGraph(Renderer* renderer) : ShadowRenderGraph(renderer)
	{

	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	std::string PCFPointLightRenderGraph::name()
	{
		return "PCF Point Light Shadow Render Graph";
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	bool PCFPointLightRenderGraph::build()
	{
		std::shared_ptr<PCFPointLightDepthNode> depth_node = std::make_shared<PCFPointLightDepthNode>(this);

		return attach_and_initialize_node(depth_node);
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	void PCFPointLightRenderGraph::refresh()
	{

	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	std::string PCFPointLightRenderGraph::sampling_source_path()
	{
		return "shader/shadows/point_light/sampling/pcf_point_light.glsl";
	}

	// -----------------------------------------------------------------------------------------------------------------------------------
}
#include "pcf_spot_light_render_graph.h"
#include "../nodes/pcf_directional_light_depth_node.h"

namespace nimble
{
	// -----------------------------------------------------------------------------------------------------------------------------------

	PCFSpotLightRenderGraph::PCFSpotLightRenderGraph(Renderer* renderer) : ShadowRenderGraph(renderer)
	{

	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	std::string PCFSpotLightRenderGraph::name()
	{
		return "PCF Spot Light Shadow Render Graph";
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	bool PCFSpotLightRenderGraph::build()
	{
		std::shared_ptr<PCFDirectionalLightDepthNode> depth_node = std::make_shared<PCFDirectionalLightDepthNode>(this);

		return attach_and_initialize_node(depth_node);
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	void PCFSpotLightRenderGraph::refresh()
	{

	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	std::string PCFSpotLightRenderGraph::sampling_source_path()
	{
		return "shader/shadows/spot_light/sampling/pcf_spot_light.glsl";
	}

	// -----------------------------------------------------------------------------------------------------------------------------------
}
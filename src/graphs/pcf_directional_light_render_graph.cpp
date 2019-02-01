#include "pcf_directional_light_render_graph.h"
#include "../nodes/pcf_directional_light_depth_node.h"

namespace nimble
{
	// -----------------------------------------------------------------------------------------------------------------------------------

	PCFDirectionalLightRenderGraph::PCFDirectionalLightRenderGraph(Renderer* renderer) : ShadowRenderGraph(renderer)
	{

	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	std::string PCFDirectionalLightRenderGraph::name()
	{
		return "PCF Directional Light Shadow Render Graph";
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	bool PCFDirectionalLightRenderGraph::build()
	{
		std::shared_ptr<PCFDirectionalLightDepthNode> depth_node = std::make_shared<PCFDirectionalLightDepthNode>(this);

		return attach_and_initialize_node(depth_node);
	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	void PCFDirectionalLightRenderGraph::refresh()
	{

	}

	// -----------------------------------------------------------------------------------------------------------------------------------

	std::string PCFDirectionalLightRenderGraph::sampling_source_path()
	{
		return "shader/shadows/directional_light/sampling/pcf_directional_light.glsl";
	}

	// -----------------------------------------------------------------------------------------------------------------------------------
}
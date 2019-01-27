#pragma once

#include "../render_graph.h"

namespace nimble
{
	class PCFPointLightRenderGraph : public ShadowRenderGraph
	{
	public:
		PCFPointLightRenderGraph(Renderer* renderer);
		std::string name() override;
		bool build() override;
		void refresh() override;
		std::string sampling_source_path() override;
	};
}
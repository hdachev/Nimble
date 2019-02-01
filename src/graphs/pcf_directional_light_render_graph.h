#pragma once

#include "../render_graph.h"

namespace nimble
{
	class PCFDirectionalLightRenderGraph : public ShadowRenderGraph
	{
	public:
		PCFDirectionalLightRenderGraph(Renderer* renderer);
		std::string name() override;
		bool build() override;
		void refresh() override;
		std::string sampling_source_path() override;
	};
}
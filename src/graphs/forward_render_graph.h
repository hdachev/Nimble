#pragma once

#include "../render_graph.h"

namespace nimble
{
	class ForwardRenderGraph : public RenderGraph
	{
	public:
		ForwardRenderGraph(Renderer* renderer);
		std::string name() override;
		bool build() override;
		void refresh() override;
	};
}
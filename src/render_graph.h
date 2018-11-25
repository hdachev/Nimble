#pragma once

#include "render_node.h"

namespace nimble
{
	class RenderGraph
	{
	public:
		RenderGraph();
		~RenderGraph();

		uint32_t id();

		virtual std::string name() = 0;
	};
}